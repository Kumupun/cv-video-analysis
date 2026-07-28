# CV Video Analysis

Каркас ролей **5 (Backend / Pipeline Engineer)** та **6 (MLOps Engineer)** для пайплайна:

`upload/URL → RGB decode → Ray Object Store → cut detection → verified tracking job → tracking → aggregation → JSON`

## Критичні правила

1. Аудіо не є етапом цього CV-пайплайна. Код не читає аудіодоріжку, відхиляє audio-only файли й працює лише з RGB-кадрами.
2. Трекінг не може початися до результату cut detection. Coordinator буферизує результати й відправляє tracking jobs строго за `chunk_index`; схема `TrackingJobMessage(cut_verified=True)` не дозволяє обхід етапу.
3. Кадри не пишуться у `.jpg` і не передаються через Redis/HTTP. У Redis передається лише непрозорий токен та метадані; справжній Ray `ObjectRef` утримує named registry actor.
4. Кожен chunk має `context range` із перекриттям і окремий `valid range`, тому склейка на межі двох батчів не губиться та не дублюється.
5. `ObjectRef` звільняється лише після наявності результатів cut detection і tracking для конкретного chunk; повторна доставка Redis-повідомлення не спричиняє повторне очищення.
6. `backend/app/workers/mock_cut.py` та `mock_tracking.py` — лише development-симулятори контрактів. Вони не замінюють TransNetV2, Co-DETR або ByteTrack.

## Структура

```text
backend/
├── app/
│   ├── api/                 # FastAPI endpoints and health checks
│   ├── core/                # settings, JSON logging, Prometheus metrics
│   ├── domain/              # strict Pydantic contracts and state machine
│   ├── infrastructure/      # Redis Streams, Ray Object Store, task repository
│   ├── services/            # upload, URL fetch, RGB decoder, aggregation
│   └── workers/
│       ├── ingest.py        # video → RGB chunks → Ray → video_chunks
│       ├── coordinator.py   # cut_results → verified tracking_jobs
│       ├── aggregator.py    # tracking_results → release ObjectRef → final JSON
│       ├── mock_cut.py      # development only
│       └── mock_tracking.py # development only
├── docker/                  # backend and ML-worker base images
├── requirements/            # pinned Python dependencies
├── scripts/                 # smoke test
├── tests/                   # backend unit tests
└── pyproject.toml           # pytest, Ruff and Black settings
monitoring/                  # Prometheus alerts + Grafana provisioning/dashboard
docs/                        # рішення архітектури та контракти ролей 3/4
data/                        # upload/result placeholders for local work
compose.yaml                 # main development stack
compose.ml.example.yaml      # example real ML-worker overlay
```

## Запуск зараз, поки ML-модулі інших учасників не готові

```bash
cp .env.example .env
docker compose --profile mock up --build
```

API: `http://localhost:8000/docs`

```bash
curl -X POST http://localhost:8000/api/v1/process \
  -F "file=@sample.mp4"
```

Після отримання `task_id`:

```bash
curl http://localhost:8000/api/v1/status/<task_id>
curl http://localhost:8000/api/v1/results/<task_id>
```

Mock cut worker створює одну сцену на chunk, а mock tracking worker повертає порожній список треків. Обидві заглушки викликають named Ray Actors, тому перевіряється не лише Redis-проводка, а й actor-шлях без вигаданих ML-результатів.

## Підключення реальних ролей 3 і 4

Контракти, які вони повинні споживати/публікувати:

- роль 4 читає `cv:video_chunks` як consumer group `cut-workers` і публікує `CutResultMessage` у `cv:cut_results`;
- coordinator читає `cv:cut_results` і тільки після цього створює `TrackingJobMessage` у `cv:tracking_jobs`;
- роль 3 читає `cv:tracking_jobs` як consumer group `tracking-workers` і публікує `TrackingResultMessage` у `cv:tracking_results`;
- aggregator збирає відповіді та звільняє Ray-об’єкт.

Приклад GPU-сервісів знаходиться в `compose.ml.example.yaml`. Реальні пакети/команди ролей 3 і 4 треба підставити замість `cut_worker.main` та `tracking_worker.main`.

Точні поля, frame mapping і правила ownership описані в `docs/WORKER_CONTRACTS.md`. Повний аналіз рішень та обмежень — у `docs/ARCHITECTURE_AND_DECISIONS.md`.

## Моніторинг

```bash
docker compose --profile mock --profile monitoring up --build
```

- Grafana: `http://localhost:3000`
- Prometheus: `http://localhost:9090`

На NVIDIA-сервері з Container Toolkit:

```bash
docker compose --profile monitoring --profile gpu up --build
```

`dcgm-exporter` збирає GPU utilization, VRAM і температуру. Node Exporter збирає CPU/RAM; Redis Exporter — стан брокера.

## Межі chunk-ів

За замовчуванням backend формує 64 унікальні кадри на chunk і додає 16 попередніх кадрів як контекст. ML-воркер отримує `context_start_frame/context_end_frame` та `valid_start_frame/valid_end_frame`. Глобальний номер кадру дорівнює `context_start_frame + local_index`; фінальні події дозволено публікувати лише для valid range.

## Важлива примітка щодо FPS

Існуючий `preprocess.py` має default `target_fps=25`, тоді як командна документація для стандартизованого датасету вказує 30 FPS. Backend навмисно **не використовує цей legacy-файл і не змінює FPS вхідного відео**: він зберігає оригінальні FPS та frame indices, щоб timestamps не “з’їжджали”. Роль 4 має узгодити власний model-specific resampling і повернути результати в координатах оригінального відео.

## Перевірки

```bash
python -m pip install -r backend/requirements/dev.txt
python -m ruff check --config backend/pyproject.toml backend/app backend/tests
python -m black --check --config backend/pyproject.toml backend/app backend/tests
python -m pytest -c backend/pyproject.toml backend/tests
python -m compileall -q backend/app backend/tests
docker compose config --quiet
docker compose -f compose.yaml -f compose.ml.example.yaml config --quiet
```

## Production notes

- Compose з окремими процесами через Ray Client перевіряє контракти та порядок, але сам по собі не гарантує фізичний zero-copy між довільними контейнерами/вузлами. Для H200 decode і ML Ray Actors треба розмістити на одному Ray node; деталі є в `docs/ARCHITECTURE_AND_DECISIONS.md`.
- Змініть пароль Grafana у `.env`.
- Не вмикайте `mock` profile у production.
- `ALLOW_REMOTE_URLS=false` за замовчуванням. При увімкненні URL перевіряються проти private/loopback/reserved IP для зменшення SSRF-ризику.
- `RAY_SHM_SIZE` та `RAY_OBJECT_STORE_BYTES` треба підібрати під RAM сервера. Не задавайте їх “наосліп” як 30–50% без перевірки паралельності, розміру кадрів і запасу для ОС/воркерів.
- Redis і Ray не публікуються назовні; зовні доступний лише FastAPI, а monitoring-порти можна закрити reverse proxy/VPN.
