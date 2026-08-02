# Roles 5 and 6 acceptance checklist

## Role 5 — Backend / Pipeline

- [x] `POST /api/v1/process` accepts exactly one file or URL.
- [x] `POST /api/v1/process/archive` safely creates one task per ZIP video
      within a 4 GiB combined-video budget.
- [x] `GET /api/v1/status/{task_id}` returns stage and chunk progress with
      retryable 503 fallback instead of transient 500.
- [x] `GET /api/v1/results/{task_id}` returns 202 until ready and final JSON
      after completion.
- [x] Audio-only resources are rejected; decoder reads RGB frames only.
- [x] Source FPS and original global frame numbers are preserved.
- [x] Chunk overlap and valid ownership range are included in the contract.
- [x] Frames are held in Ray, not Redis/HTTP/JPEG files.
- [x] Tracking jobs are created only after cut verification.
- [x] Redis Streams consumer groups, stale reclaim, retries, and dead letters.
- [x] Duplicate stream deliveries do not double-count or double-release.
- [x] Ray object is released after cut plus tracking results.
- [x] Final output contains video metadata, transitions, tracks, warnings.
- [x] Mock contract workers allow end-to-end integration before ML readiness.
- [ ] Real TransNetV2 worker supplied by role 4.
- [ ] Real Co-DETR/ByteTrack worker supplied by role 3.
- [ ] End-to-end benchmark on the actual H200 host.

## Role 6 — MLOps

- [x] Separate backend and ML worker Dockerfiles.
- [x] Multi-stage images and pinned Python dependencies.
- [x] Compose brings the development pipeline up with one command.
- [x] Redis/Ray are internal-only; FastAPI is the public entry point.
- [x] Configurable Ray `/dev/shm` and object-store memory.
- [x] GPU reservation example for real ML workers.
- [x] Prometheus configuration and alerts.
- [x] Grafana provisioning and starter dashboard.
- [x] Node, Redis, and optional DCGM exporters.
- [x] Worker-specific metrics endpoint and processing-time/error metrics.
- [x] GitHub Actions for lint, format, tests, Compose validation, backend image.
- [ ] Slack/Telegram receiver credentials and routing supplied by the team.
- [ ] Container Registry credentials and production deployment target supplied.
- [ ] CUDA/PyTorch/TensorRT matrix finalized with roles 3 and 4.
- [ ] Load test, GPU profile, and alert thresholds calibrated on H200.

## Commands used for local verification

```bash
python -m pytest -c backend/pyproject.toml backend/tests
python -m compileall -q backend/app backend/tests
```

Also run on a machine with Docker:

```bash
docker compose config --quiet
docker compose -f compose.yaml -f compose.ml.example.yaml config --quiet
docker compose --profile mock up --build
docker compose --profile mock --profile monitoring up --build
bash backend/scripts/smoke_test.sh sample.mp4
```
