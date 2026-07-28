# Architecture and decisions for roles 5 and 6

## 1. What was present before this implementation

The initial repository contained only:

- a small `README.md`;
- `checklist.md`;
- `preprocess.py` with a local OpenCV preprocessing function.

There was no FastAPI application, task state storage, Redis Streams contract,
Ray adapter, worker orchestration, Docker Compose topology, CI, monitoring,
or tests. The implementation in this repository adds that missing platform
layer without pretending that TransNetV2, Co-DETR, or ByteTrack are ready.

## 2. Required order of processing

The enforced path is:

```text
upload or URL
  -> RGB-only validation and decoding
  -> Ray ObjectRef for a frame window
  -> cut detection result
  -> coordinator verifies the cut result
  -> tracking job
  -> tracking result
  -> aggregation
  -> Ray object release
  -> final JSON
```

Tracking never reads `cv:video_chunks` directly. It reads only
`cv:tracking_jobs`, which can be created only from a validated
`CutResultMessage`. This removes the race condition in the draft architecture
where cut detection and tracking could consume the same raw chunk in parallel.

The pipeline is asynchronous per chunk. It does not need to wait for cut
detection of the entire video before starting tracking, but each individual
chunk must have its cut result first. Cut results may arrive out of order; the
coordinator stores them and publishes tracking jobs only as a contiguous
sequence `0, 1, 2, ...`. A per-task Redis lock prevents two coordinator
instances from dispatching the same sequence concurrently.

## 3. Audio is explicitly outside the pipeline

The backend:

- rejects an uploaded or remote resource whose MIME type is `audio/*`;
- opens only the RGB video stream through Decord;
- never extracts, stores, queues, or analyzes an audio track;
- never changes stage from video processing to an audio stage because such a
  stage does not exist in the state machine.

A container may contain FFmpeg because it is useful for video diagnostics and
future codec support. Its presence does not mean audio is processed.

## 4. Chunk boundary correctness

A cut detector needs temporal context around a batch boundary. A plain split
such as `0..63`, `64..127` can miss a transition around frames 63/64.
Therefore ingestion produces two ranges:

- **context range** — all frames physically stored in the Ray object;
- **valid range** — the unique, non-overlapping range owned by this chunk.

With `CHUNK_SIZE_FRAMES=64` and `CHUNK_OVERLAP_FRAMES=16`:

```text
chunk 0: context 0..63,   valid 0..63
chunk 1: context 48..127, valid 64..127
chunk 2: context 112..191, valid 128..191
```

For a tensor-local index `i`, the global frame is:

```text
global_frame = context_start_frame + i
```

Output ownership rule:

- a hard cut is emitted only by the chunk whose valid range contains its
  `frame`;
- a gradual transition is emitted only by the chunk whose valid range contains
  its `end_frame`;
- scene intervals returned for tracking must be clipped to the valid range.

These rules prevent both missed boundary events and duplicate final events.
The final aggregator still deduplicates equal transitions defensively.

## 5. FPS decision

The existing legacy `preprocess.py` defaults to 25 FPS, while the team pipeline
document describes a 30 FPS standardized dataset. Silently applying either
value inside the API would shift frame numbers and timestamps.

The backend therefore preserves source FPS and original global frame numbers.
A model-specific worker may resample internally, but it must map every output
back to the original video coordinate system before publishing its result.

## 6. Redis Streams and delivery semantics

Streams:

| Stream | Producer | Consumer |
|---|---|---|
| `cv:ingest_jobs` | API | ingest worker |
| `cv:video_chunks` | ingest worker | cut worker |
| `cv:cut_results` | cut worker | coordinator |
| `cv:tracking_jobs` | coordinator | tracking worker |
| `cv:tracking_results` | tracking worker | aggregator |
| `cv:dead_letters` | common worker runtime | operator/debugging |

Consumers acknowledge a message only after successful handling. Stale pending
messages are reclaimed. After the configured attempt limit, the message is
copied to the dead-letter stream and acknowledged so one poison message cannot
block the whole consumer group.

Chunk writes are idempotent in Redis Lua scripts:

- a duplicate cut result cannot increment progress twice or create a second
  tracking job;
- a duplicate tracking result cannot increment progress twice or release the
  same Ray object twice.

This is required because Redis Streams consumer groups provide at-least-once,
not exactly-once, processing.

## 7. Ray object lifetime

Redis stores only an opaque token and metadata. It does not store image bytes
or a pickled Ray Client `ObjectRef`. A detached named registry actor owns the
real ObjectRef and exchanges it with workers through Ray's native nested-reference
protocol. This is required because independently connected Ray Client processes
cannot safely reconstruct each other's out-of-band pickled ClientObjectRefs.

The aggregator asks the registry to release the object only after both cut and
tracking output exist for the chunk. The registry uses an explicit Ray free call
where available and falls back to Ray reference-counted garbage collection if
that private API changes. Cleanup is isolated in one adapter so it can be
replaced without changing domain code.

### Important production limitation

The included Compose profile uses Ray Client between separate containers. It
is suitable for validating APIs, contracts, ordering, retries, and cleanup,
but it is not enough to claim strict local zero-copy for every transfer.
Ray can expose a NumPy object without copying when the consumer reads an object
that is already local to the same Ray node; a remote object must first be
transferred to that node.

For the H200 deployment, decoding, TransNetV2, and Co-DETR/ByteTrack should be
Ray actors scheduled on the same physical Ray node whenever strict local
shared-memory access is required. The upload volume must also be visible on
that node. Do not market the development Compose topology as guaranteed
zero-copy across arbitrary containers or machines.

## 8. Failure behavior

- Validation errors are returned before a task is queued.
- Worker errors remain pending for retry and then move to `cv:dead_letters`.
- A task has a terminal `failed` state and cannot silently resume from it.
- A tracking result arriving without a stored cut result is rejected.
- A final result is created only after every chunk has a tracking result.
- Remote URLs are disabled by default. When enabled, each redirect target is
  revalidated to block redirects into loopback/private/reserved networks.

A complete production retry for a partially decoded video requires a durable
chunk manifest or deterministic actor restart strategy. The present scaffold
makes downstream handlers idempotent, but a full multi-gigabyte resumable
decoder is intentionally not faked.

## 9. Role 6 topology

The default Compose file provides:

- FastAPI backend;
- Redis with AOF and `noeviction`;
- Ray head/object store with configurable `/dev/shm` and object-store size;
- ingest, coordinator, and aggregator workers;
- optional mock cut/tracking workers;
- optional Prometheus, Grafana, Node Exporter, Redis Exporter;
- optional NVIDIA DCGM Exporter.

Only FastAPI is required to be exposed publicly. Redis and Ray use internal
Docker DNS names and `expose`, not host `ports`.

The ML Dockerfile is multi-stage. Roles 3 and 4 must add their exact pinned
PyTorch, CUDA/TensorRT-compatible libraries, model code, and weights in derived
images. Do not install arbitrary latest GPU libraries at runtime.

## 10. What the mocks do and do not do

`mock_cut` returns one scene covering only the chunk valid range and no
transitions. `mock_tracking` returns an empty track list. Their lightweight
validation runs inside named Ray Actors, while Redis consumers remain
orchestrators. The mock tracking actor also rejects future out-of-order chunks
per task. They prove that the pipeline wiring works without inventing model
accuracy or detections.

They must never be enabled in production.
