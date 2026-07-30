# Validation report

## Completed in this environment

- `python -m pytest -c backend/pyproject.toml backend/tests` — **72 passed**.
- `python -m compileall -q backend/app backend/tests participant_3_tracking participant_4_cut_detection` — passed.
- Python source line-length scan — no lines above 88 characters.
- The supplied `models/weights_for_cut.pth` passes its SHA-256 manifest,
  contains `checkpoint["net"]` with 90 state entries, and loads strictly into
  the bundled AutoShot architecture with no missing, unexpected, or mismatched
  tensor shapes.
- A CPU AutoShot smoke inference over 100 RGB frames returned 100 finite
  probabilities and completed without a random-weight fallback.
- Parsed successfully as YAML:
  - `compose.yaml`;
  - `compose.ml.example.yaml`;
  - Prometheus configuration and alerts;
  - Grafana provisioning;
  - GitHub Actions workflow.
- Grafana dashboard parsed successfully as JSON.
- FastAPI/Pydantic domain and API modules import and compile.
- ZIP batch tests cover safe extraction, skipped files, traversal rejection,
  compression-ratio limits, combined-video-size admission, disk-reserve
  checks, API task creation, and retryable status reads without HTTP 500.
- Worker resilience tests cover immediate same-message retry and durable,
  atomic tracking-result ordering across actor reconstruction/redelivery.
- Performance tests verify memory-bounded adaptive chunking and the 80 MiB
  local default: 86 chunks for the 511-frame 2560x1440 reference and 14 chunks
  for the 302-frame 1280x720 reference, or 100 chunks combined.
- Decoder tests verify sequential FFmpeg chunk streaming, overlap ownership,
  exact resume from a requested chunk and fallback frame counting.
- A direct FFmpeg integration run decoded three synthetic 720p/1080p/1440p
  videos (1,190 frames total) with a measured peak RSS below 500 MiB.
- Long-running Redis messages refresh their pending-entry idle time, preventing
  duplicate auto-claim while a multi-gigabyte video is still processing.

## Not executable in this environment

The execution container used to prepare this archive does not provide:

- Docker / Docker Compose;
- Redis server or `redis-py` runtime package;
- Ray runtime;
- NVIDIA GPU / H200 / Container Toolkit.

Therefore the following are prepared but not falsely reported as executed:

- Docker image builds;
- `docker compose up` integration run;
- real Redis Streams integration;
- Ray Actor/Object Store integration;
- full Docker/Ray archive processing on the target Docker Desktop host;
- Prometheus/Grafana/DCGM live scraping;
- H200 throughput, memory, temperature, or accuracy benchmarking.

Run these checks on a Docker/NVIDIA host before merging to production:

```bash
python -m pip install -r backend/requirements/dev.txt
python -m ruff check --config backend/pyproject.toml backend/app backend/tests
python -m black --check --config backend/pyproject.toml backend/app backend/tests
python -m pytest -c backend/pyproject.toml backend/tests
docker compose config --quiet
docker compose -f compose.yaml -f compose.ml.example.yaml config --quiet
docker compose --profile mock up --build
bash backend/scripts/smoke_test.sh /absolute/path/to/sample.mp4
```

For the H200 environment, also run:

```bash
docker compose --profile monitoring --profile gpu up --build
nvidia-smi
docker compose ps
docker compose logs --tail=200
```

Run the real `ml` profile with the supplied cut checkpoint and the agreed
YOLO-World checkpoint, then measure Precision/Recall/F1 for cuts and
mAP/MOTA/IDF1 for detection/tracking on the agreed test dataset.

## Throughput optimization validation

- 72 backend tests pass, including checkpoint compatibility, task-partitioned
  concurrency, atomic cut-to-tracking dispatch, atomic tracking progress, and
  final timing fields.
- Python bytecode compilation succeeds for backend, tests, and both participant
  worker packages.
- `compose.yaml` and the ML overlay parse as valid YAML.
- Changed Python files contain no lines longer than 88 characters.
- Docker is unavailable in the authoring environment, so the final wall-clock
  comparison must be measured with the local mock stack.
