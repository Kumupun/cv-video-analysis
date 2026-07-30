# Contracts for roles 3 and 4

All payloads are Pydantic models in `backend/app/domain/schemas.py`. Workers should
import these models instead of maintaining a second handwritten JSON schema.
Unknown fields are rejected.

## Role 4: input `FrameBatchMetadata`

```json
{
  "task_id": "UUID",
  "chunk_id": "UUID:00000001",
  "chunk_index": 1,
  "object_ref": "opaque-Ray-token",
  "context_start_frame": 48,
  "context_end_frame": 127,
  "valid_start_frame": 64,
  "valid_end_frame": 127,
  "frame_count": 80,
  "fps": 30.0,
  "width": 1920,
  "height": 1080,
  "source_total_frames": 9000,
  "is_last": false
}
```

The tensor shape is `[T, H, W, C]`, dtype `uint8`, RGB. For local tensor index
`i`, use `global_frame = context_start_frame + i`.

Role 4 may use all context frames for inference, but may publish an event only
according to the valid-range ownership rule documented in
`ARCHITECTURE_AND_DECISIONS.md`.

## Role 4: output `CutResultMessage`

```json
{
  "task_id": "UUID",
  "chunk_id": "UUID:00000001",
  "chunk_index": 1,
  "object_ref": "same-opaque-Ray-token",
  "fps": 30.0,
  "context_start_frame": 48,
  "context_end_frame": 127,
  "valid_start_frame": 64,
  "valid_end_frame": 127,
  "transitions": [
    {
      "type": "hard_cut",
      "frame": 92,
      "timestamp": 3.0666667,
      "confidence": 0.97
    }
  ],
  "scenes": [
    {"scene_id": "task:scene:5", "start_frame": 64, "end_frame": 92},
    {"scene_id": "task:scene:6", "start_frame": 93, "end_frame": 127}
  ]
}
```

Rules:

- echo the same `task_id`, `chunk_id`, `chunk_index`, `object_ref`, FPS, and
  ranges;
- hard cut requires `frame`, `timestamp`, and confidence;
- gradual transition requires start/end frame and timestamp plus confidence;
- scenes are global-frame intervals clipped to the valid range;
- keep the same `scene_id` in adjacent chunks when there was no transition;
- change `scene_id` only after a verified hard/gradual transition;
- do not publish flashes, whip pans, subtitles, or lighting changes as cuts
  merely because pixel difference is high;
- do not release the Ray object.

## Role 3: input `TrackingJobMessage`

The coordinator creates this message only after validating and storing the cut
result. It contains the same frame mapping, scenes, transitions, and
`cut_verified=true`.

The coordinator buffers out-of-order cut results and emits tracking jobs in
strict `chunk_index` order. A real tracking service should still use one named
Ray Actor per task (or an equivalent partition) with sequential execution,
because ByteTrack state cannot safely be updated by two chunks concurrently.

Role 3 must:

- reject a message without `cut_verified=true`;
- process only frames/scenes inside the valid range;
- reset tracker state at every cut/scene boundary supplied by role 4;
- never carry a Track ID across a scene boundary;
- return global frame numbers, not local tensor indices;
- keep `scene_id` in every tracked object.

## Role 3: output `TrackingResultMessage`

```json
{
  "task_id": "UUID",
  "chunk_id": "UUID:00000001",
  "chunk_index": 1,
  "object_ref": "same-opaque-Ray-token",
  "tracks": [
    {
      "frame": 95,
      "scene_id": "task:scene:6",
      "track_id": 0,
      "class_name": "person",
      "confidence": 0.91,
      "bbox": {"x": 120.0, "y": 80.0, "width": 90.0, "height": 210.0}
    }
  ]
}
```

The aggregator owns Ray cleanup. Role 3 must not free the object.

## Consumer groups

- role 4 group: `cut-workers` on `cv:video_chunks`;
- role 3 group: `tracking-workers` on `cv:tracking_jobs`.

Use `XREADGROUP`; acknowledge only after output was successfully written.
The shared worker runtime in `backend/app/workers/common.py` already implements stale
message reclaim, immediate in-order retries, dead-lettering, timing metrics,
and acknowledgements. Tracking-result publication advances a durable Redis
chunk sequence in the same Lua operation as `XADD`, so a worker or actor
reconstruction cannot make later chunks overtake a failed earlier chunk. A real
worker can reuse this Redis-facing adapter, but model inference must be
submitted to a Ray Actor. For stateful tracking, the actor must be partitioned
by `task_id` and checkpoint/rebuild its model state after a restart.

## Contract additions in pipeline version 1.1

`FrameBatchMetadata` adds optional `decode_ms`. `CutResultMessage` echoes
`decode_ms`, adds `cut_processing_ms`, and carries `is_last`. The coordinator
copies `is_last` into `TrackingJobMessage`. `TrackingResultMessage` adds
`tracking_processing_ms`. These fields default to zero/false for compatibility,
but real workers should populate their own processing timing and preserve
`is_last` unchanged.

Workers may process different `task_id` partitions concurrently. Messages for
one task must remain sequential, and a real stateful tracking implementation
must use one actor/state partition per task rather than one global tracker.
