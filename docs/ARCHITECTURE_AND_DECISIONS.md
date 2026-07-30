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

## 3. ZIP batch ingestion

`POST /api/v1/process/archive` accepts one ZIP file and creates one independent
pipeline task for every supported video entry. The existing single-video
`POST /api/v1/process` response contract remains unchanged.

The archive is first streamed to a temporary file with a compressed-size limit.
It is then inspected and extracted entry by entry without loading whole videos
into RAM. Every accepted member is written as `source.<ext>` inside its own
UUID task directory. All Redis task hashes and ingest stream events are created
in one transaction, so a queue failure cannot leave a partially accepted batch.

Archive hardening includes:

- rejection of absolute paths, `..` traversal, Windows drive paths, symlinks,
  encrypted entries, invalid ZIP data, and duplicate member names;
- a primary 4 GiB combined-video budget independent of video count, plus
  configurable compressed upload, total uncompressed, per-video, member-safety,
  free-disk reserve, and compression-ratio limits;
- unsupported non-video files are skipped and reported to the caller;
- temporary ZIP files and extracted task directories are removed on failure.

The ZIP itself is not a pipeline task and is deleted after extraction. Each
video continues through the normal upload source contract and has its own
status and result endpoints. The admission response reports both compressed
archive bytes and accepted video bytes, so capacity is observable without
using video count as the primary constraint.

Status reads use bounded retries around Redis access. A persistent temporary
read failure is translated to HTTP 503 with `Retry-After`, preventing a
short-lived backend dependency hiccup from surfacing as an opaque HTTP 500
while the task itself continues normally.

## 4. Audio is explicitly outside the pipeline

The backend:

- rejects an uploaded or remote resource whose MIME type is `audio/*`;
- opens only the first RGB video stream through FFmpeg;
- never extracts, stores, queues, or analyzes an audio track;
- never changes stage from video processing to an audio stage because such a
  stage does not exist in the state machine.

FFmpeg is the bounded streaming decoder. The command maps only `0:v:0` and
explicitly disables audio, subtitle and data streams, so audio is never queued
or analyzed.

## 5. Chunk boundary correctness

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

### Local throughput preset

The Docker Desktop mock profile keeps source FPS and source resolution intact.
It improves throughput by reducing orchestration overhead rather than changing
the input data:

- decoded RGB tensors are capped at 80 MiB;
- FFmpeg writes one chunk at a time into a preallocated NumPy buffer and exits
  after each video, preventing native decoder caches from growing with duration;
- archive entries are queued and one video is decoded at a time on Docker
  Desktop, while downstream stages remain concurrent;
- at most two tensors are in flight, so the theoretical Ray payload stays below
  the 256 MiB local object-store budget;
- ingest waits for downstream capacity before reading the next tensor;
- mock actors add no artificial sleep;
- human-facing progress is persisted every two chunks instead of on every
  stream hop, while durable completion counters are still updated atomically
  for every chunk.

For the reference 511-frame 2560x1440 and 302-frame 1280x720 videos, the
adaptive chunker produces 73 and 12 chunks respectively, compared with 171 and
34 under the former 32 MiB budget. This is a local orchestration optimization,
not a claim about real model inference speed. Production values must be tuned against model VRAM, GPU batch size, Ray node
memory, and the number of separately scaled ingest processes. Increasing
coroutine concurrency inside one decoder process is not the recommended scaling
mechanism.

## 6. FPS decision

The existing legacy `preprocess.py` defaults to 25 FPS, while the team pipeline
document describes a 30 FPS standardized dataset. Silently applying either
value inside the API would shift frame numbers and timestamps.

The backend therefore preserves source FPS and original global frame numbers.
A model-specific worker may resample internally, but it must map every output
back to the original video coordinate system before publishing its result.

## 7. Redis Streams and delivery semantics

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

## 8. Ray object lifetime

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

## 9. Failure behavior

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

## 10. Role 6 topology

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

## 11. What the mocks do and do not do

`mock_cut` returns one scene covering only the chunk valid range and no
transitions. `mock_tracking` returns an empty track list. Both validate the
strict stream metadata without opening another Ray Client connection. Ingest
owns the frame registry, and after Redis records tracking completion it releases
the corresponding ObjectRef through the same client that created it. Strict
per-task result ordering is enforced by an atomic Redis sequence plus `XADD`,
which also makes a redelivery idempotent.

They must never be enabled in production.

## 11. Task-partitioned concurrency and reduced orchestration

The mock profile processes different task IDs concurrently but never executes
two messages from the same task partition at once inside a worker batch. This
keeps stateful tracking order while allowing independent archive videos to
advance together. Ingest dynamically limits each active video to one in-flight
Ray tensor when several videos are running; a lone video may still use the
configured two-slot pipeline.

The coordinator no longer acquires a distributed lock and performs repeated
read/publish/advance calls for every chunk. A single Redis Lua script persists
the cut result, buffers its prepared tracking job, and publishes every newly
contiguous job. Tracking completion and progress use another atomic Lua update.
These scripts target the single Redis deployment in `compose.yaml`; a future
Redis Cluster migration must use hash tags or move the dynamic chunk lookup to
a cluster-safe data layout.

Mock workers keep the opaque frame token unchanged but do not dereference it.
They are stateless; per-task ordering is provided by the Redis worker partition.
Real model workers still resolve the actual ObjectRef and must retain any
required task-partitioned state.

Final result version `1.1` includes measured queue, decode, cut, tracking,
aggregation, orchestration/wait, and total wall-clock timings. Active stage
measurements can overlap when multiple tasks execute concurrently and should
not be interpreted as mutually exclusive wall-clock slices.


### Local mock actor safety

The local mock profile keeps the real bounded RGB tensor in Ray, but mock cut
and tracking validate only the strict message metadata. The ingest process that
created the ObjectRef also releases it after durable tracking completion. This
removes all cross-client actor calls from the local mock path while retaining
Object Store pressure and deterministic lifetime. Real ML workers continue to
resolve the actual ObjectRef. `RAY_ACTOR_CALL_TIMEOUT_SECONDS` bounds owner-side
registry calls.

## 12. Real model integration

The real-model overlay adds one GPU-enabled Ray worker node containing the
backend package and both participant model packages. Redis consumer processes
remain lightweight Ray clients. They resolve the opaque registry token but pass
the resulting ObjectRef to model actors as a nested reference, so the large RGB
tensor is materialized on the Ray worker node rather than in Redis or in the
consumer process.

AutoShot inference is shared by a named actor. Scene numbering is persisted per
task in Redis, and each cached chunk result is written before stream
publication, which makes redelivery idempotent. Only transitions whose global
anchor belongs to the chunk valid range are emitted.

YOLO-World uses a named actor per task, which is the state partition required by
ByteTrack. The actor processes only scene intervals inside the valid range,
resets on a changed `scene_id`, maps raw tracker IDs to scene-local IDs, and
checkpoints predictor state plus ID mappings after every chunk. The worker
publishes through the existing atomic Redis sequence and destroys the actor
after the last chunk.

The supplied AutoShot checkpoint is versioned at `models/weights_for_cut.pth`
with a SHA-256 manifest. Its compatible architecture is bundled in the cut
worker package, so production startup does not download executable Python
source. Loading is strict and fails instead of silently using random or
partially matched weights. The tracking checkpoint remains deployment-specific;
required paths are documented in `models/README.md`.
The `mock` and `ml` profiles must not be enabled together because they
intentionally share consumer groups.
