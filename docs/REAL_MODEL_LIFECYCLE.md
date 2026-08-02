# Real-model lifecycle and recovery

## 1. Startup

1. `storage-init` creates `/data/uploads` and `/data/results` and sets ownership
   to UID/GID `10001`. Backend and ML images use that same UID.
2. Redis and the Ray head become healthy.
3. `ml-ray-worker` joins Ray with one GPU, a bounded 256 MiB local Object Store,
   and the `cut-model` / `tracking-model` custom resources.
4. Backend and stream workers start only after their dependencies are ready.
5. Only `ml-ray-worker` owns the ML image build definition. Cut and tracking
   reuse the resulting image, preventing parallel-build tag collisions.

## 2. Single video

1. `POST /api/v1/process` validates and writes the upload.
2. One durable task and one `cv:ingest_jobs` message are created atomically.
3. FFmpeg probes the original FPS and resolution, then decodes sequential,
   memory-bounded RGB chunks.
4. Ingest keeps at most two tensors in Ray and stores only their opaque registry
   tokens in Redis. This lets AutoShot on chunk N+1 overlap ByteTrack on N.
5. AutoShot resolves the ObjectRef, downsizes source frames to `48x27 uint8`,
   pads only the small representation to its 100-frame window, runs both its
   one-hot boundary and many-hot transition-range heads, and publishes hard and
   gradual transitions to `cv:cut_results`. A gradual range still open at a
   chunk boundary is carried in durable cut state and emitted only after its
   true end is visible in a later chunk.
6. Coordinator publishes only contiguous, cut-verified `cv:tracking_jobs`.
7. YOLO-World detection and ByteTrack process each video's tracking jobs
   sequentially, preserving tracker state without copying the whole RGB chunk
   into another BGR batch. Each verified transition resets ByteTrack before the
   frame immediately following the transition.
8. Aggregation stores the final JSON. Ingest observes durable tracking progress
   and releases each Ray object from the client that created it.
9. The completed source directory is removed. Task/result metadata stays in
   Redis for the configured TTL.

## 3. ZIP archive

`POST /api/v1/process/archive` safely extracts supported videos and creates all
tasks in one Redis transaction. Every video receives an independent task ID,
cut scene state, tracking actor, status, and result. Ingest concurrency remains
one, so archive entries queue instead of opening several FFmpeg decoders and
large tensors simultaneously. The number of files is governed by safety and
combined-size limits, not by an artificial small video count.

## 4. Recovery rules

- Redis socket/connect timeouts use client retries with exponential backoff.
- A transient Redis connection/timeout inside a handler does not consume the
  message delivery budget and does not mark the video failed. The pending
  message resumes from durable chunk counters.
- Deterministic model/schema failures still use the normal bounded retry and
  dead-letter path.
- AutoShot is a versioned named Ray Actor with actor restart and task retry. The
  driver drops a failed local handle before the next message attempt.
- Chunk publishing, cut dispatch, tracking completion, and final aggregation are
  idempotent; a reclaimed message cannot duplicate a completed chunk.
- Downstream stall detection remains enabled. A task fails only when no durable
  progress is made for the configured interval, rather than on one short Redis
  pause.

## 5. Shutdown and rebuild

Graceful stop keeps named volumes:

```powershell
docker compose -f compose.yaml -f compose.ml.example.yaml --profile ml --profile mock stop
```

Rebuild and start cut-only validation:

```powershell
docker compose -f compose.yaml -f compose.ml.example.yaml --profile ml --profile mock up --build -d redis ray-head storage-init backend ingest-worker coordinator aggregator ml-ray-worker cut-worker mock-tracking-worker
```

A full reset is also safe because `storage-init` repairs volume ownership:

```powershell
docker compose -f compose.yaml -f compose.ml.example.yaml --profile ml --profile mock down -v --remove-orphans
```

## 6. Verification

Verify AutoShot and both locally supplied tracking checkpoints:

```powershell
Get-FileHash .\models\weights_for_cut.pth -Algorithm SHA256
Get-FileHash .\models\best.pt -Algorithm SHA256
Get-FileHash .\models\yoloe26l_military_assets.pt -Algorithm SHA256

docker compose -f compose.yaml -f compose.ml.example.yaml --profile ml run --rm `
  tracking-worker python -c "from ultralytics import YOLO; a=YOLO('/models/best.pt'); b=YOLO('/models/yoloe26l_military_assets.pt'); print(type(a.model).__name__, a.names); print(type(b.model).__name__, b.names)"
```

End-to-end video or archive:

```powershell
.\backend\scripts\smoke_test.ps1 "C:\path\sample.mp4"
.\backend\scripts\smoke_test.ps1 "C:\path\videos.zip"
```

A successful cut-only run must reach `completed` and contain
`AutoShot chunk completed` and `YOLO ensemble + ByteTrack tracking chunk completed`
in worker logs. Empty transitions are valid when the video contains no
confident hard or gradual transition. Tune gradual decoding with
`CUT_MODEL_GRADUAL_THRESHOLD`, `CUT_MODEL_GRADUAL_BOUNDARY_THRESHOLD`,
`CUT_MODEL_GRADUAL_MIN_FRAMES`, and `CUT_MODEL_GRADUAL_MERGE_GAP_FRAMES`;
rebuild/recreate the ML services after changing them.
