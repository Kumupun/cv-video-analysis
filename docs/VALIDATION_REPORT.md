# Validation report

## Completed in this environment

- `python -m pytest -c backend/pyproject.toml backend/tests` — **14 passed**.
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
