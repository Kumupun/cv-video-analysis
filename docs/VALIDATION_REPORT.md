# Validation report

## Completed in this environment

- `python -m pytest -c backend/pyproject.toml backend/tests` — **54 passed**.
- `python -m compileall -q backend/app backend/tests` — passed.
- Python source line-length scan — no lines above 88 characters.
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
- Decoder tests verify direct random-access chunk decoding, allowing ingest to
  wait for capacity before allocating the next RGB tensor and to skip already
  published chunks during a retry.

## Not executable in this environment

The execution container used to prepare this archive does not provide:

- Docker / Docker Compose;
- Redis server or `redis-py` runtime package;
- Ray runtime;
- Decord;
- NVIDIA GPU / H200 / Container Toolkit.

Therefore the following are prepared but not falsely reported as executed:

- Docker image builds;
- `docker compose up` integration run;
- real Redis Streams integration;
- Ray Actor/Object Store integration;
- Decord decoding of a real uploaded video;
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

Finally replace mocks with roles 3 and 4, then measure Precision/Recall/F1 for
cuts and mAP/MOTA/IDF1 for detection/tracking on the agreed test dataset.

## Throughput optimization validation

- 54 backend tests pass, including task-partitioned concurrency, atomic
  cut-to-tracking dispatch, atomic tracking progress, and final timing fields.
- Python bytecode compilation succeeds for `backend/app` and `backend/tests`.
- `compose.yaml` and the ML overlay parse as valid YAML.
- Changed Python files contain no lines longer than 88 characters.
- Docker is unavailable in the authoring environment, so the final wall-clock
  comparison must be measured with the local mock stack.
