# CV Video Analysis

Інтегрований бекендовий і ML-пайплайн для ролей **3 (Tracking)**, **4 (Cut Detection)**, **5 (Backend / Pipeline)** та **6 (MLOps)**:

`upload/URL/ZIP batch → RGB decode → Ray Object Store → cut detection → verified tracking job → tracking → aggregation → JSON`

## Критичні правила

1. Аудіо не є етапом цього CV-пайплайна. Код не читає аудіодоріжку, відхиляє audio-only файли й працює лише з RGB-кадрами.
2. Трекінг не може початися до результату cut detection. Coordinator буферизує результати й відправляє tracking jobs строго за `chunk_index`; схема `TrackingJobMessage(cut_verified=True)` не дозволяє обхід етапу.
3. Кадри не пишуться у `.jpg` і не передаються через Redis/HTTP. У Redis передається лише непрозорий токен та метадані; справжній Ray `ObjectRef` утримує named registry actor.
4. Кожен chunk має `context range` із перекриттям і окремий `valid range`, тому склейка на межі двох батчів не губиться та не дублюється.
5. `ObjectRef` звільняється лише після наявності результатів cut detection і tracking для конкретного chunk; повторна доставка Redis-повідомлення не спричиняє повторне очищення.
6. `backend/app/workers/mock_cut.py` та `mock_tracking.py` — лише development-симулятори контрактів. Реальний ML-профіль використовує AutoShot та YOLO-World + ByteTrack.

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
├── docker/                  # backend and integrated ML image
├── requirements/            # pinned Python dependencies
├── scripts/                 # smoke test
├── tests/                   # backend unit tests
└── pyproject.toml           # pytest, Ruff and Black settings
participant_3_tracking/      # YOLO-World + ByteTrack actor and Redis adapter
participant_4_cut_detection/ # AutoShot actor and Redis adapter
models/                      # supplied AutoShot checkpoint + optional tracking artifact
monitoring/                  # Prometheus alerts + Grafana provisioning/dashboard
docs/                        # рішення архітектури та контракти ролей 3/4
data/                        # upload/result placeholders for local work
compose.yaml                 # main development stack
compose.ml.example.yaml      # real GPU model overlay
```

## Mock-запуск без model artifacts

```bash
cp .env.example .env
docker compose --profile mock up --build
```

API: `http://localhost:8000/docs`

### Локальний запуск при обмеженій пам’яті Docker Desktop

У mock-профілі Ray dashboard вимкнений, а локальний Ray memory monitor за
замовчуванням вимкнений через `RAY_MEMORY_MONITOR_REFRESH_MS=0`. Це запобігає
хибному `ingest_failed`, коли Docker VM уже використовує понад стандартні 95%
пам’яті, хоча Ray Object Store майже порожній. Для production поверніть
`RAY_MEMORY_MONITOR_REFRESH_MS=250`, залиште достатній запас RAM і налаштуйте
`RAY_MEMORY_USAGE_THRESHOLD`.

Локальний preset використовує послідовний FFmpeg decoder, RGB-chunk до
80 MiB і не більше двох Ray-об’єктів одночасно. ZIP може містити будь-яку
кількість відео в межах сумарного 4 GiB бюджету, але ingest ставить їх у чергу
та декодує по одному. Це не обмежує кількість задач: воно не дозволяє двом
великим native decoder-ам одночасно заповнити RAM Docker Desktop. Cut,
coordinator, tracking та aggregation продовжують працювати паралельно.

Після зміни Ray-параметрів перестворіть stack. `storage-init` автоматично
вирівнює права named volumes для UID `10001`, тому ручний `chown` більше не
потрібний навіть після `down -v`:

```bash
docker compose --profile mock down -v --remove-orphans
docker compose --profile mock up --build -d
```

```bash
curl -X POST http://localhost:8000/api/v1/process \
  -F "file=@sample.mp4"
```

Після отримання `task_id`:

```bash
curl http://localhost:8000/api/v1/status/<task_id>
curl http://localhost:8000/api/v1/results/<task_id>
```

### Завантаження ZIP з кількома відео

Окремий endpoint не змінює контракт одиночного `/process`:

```bash
curl -X POST http://localhost:8000/api/v1/process/archive \
  -F "file=@videos.zip"
```

Кожне підтримуване відео в архіві отримує власний `task_id`, `status_url` і
`results_url`. Сторонні файли (`.txt`, `.json`, системні файли macOS тощо)
не запускаються, а повертаються в `skipped_files` із причиною. Усі відеозадачі
створюються в Redis однією транзакцією.

ZIP-приймання тепер орієнтується насамперед на **сумарний розмір відео**, а
не на їх кількість. За замовчуванням дозволено до 4 GiB для самого ZIP і до
4 GiB сумарно для всіх підтримуваних відео всередині нього. Тому один великий
файл може використати весь бюджет або багато малих файлів можуть поділити його.
`MAX_ARCHIVE_MEMBERS=2000` залишається лише технічним anti-abuse запобіжником,
а не бізнес-лімітом кількості відео.

Відповідь endpoint також містить `archive_size_bytes` і
`accepted_size_bytes`, щоб клієнт бачив фактичне використання розмірного
бюджету. ZIP-перевірка захищає від path traversal, символічних посилань,
зашифрованих entries, zip-bomb сценаріїв, нестачі місця на диску та підозріло
високого compression ratio. Ліміти налаштовуються через `MAX_ARCHIVE_*` у
`.env`.

Короткі збої читання статусу з Redis повторюються автоматично. Якщо стан усе
ще недоступний, API повертає retryable `503` із `Retry-After: 1`, а не
неінформативний `500 Internal Server Error`.

Mock cut worker створює одну сцену на chunk, а mock tracking worker повертає
порожній список треків. Заглушки перевіряють лише строгі метадані повідомлень і
не відкривають додаткові Ray Client з’єднання. RGB-тензор усе одно зберігається
в Ray ingest-процесом; після підтвердженого tracking-result той самий ingest
процес звільняє ObjectRef. Це прибирає блокування першого chunk у Docker Desktop,
але зберігає реальний Object Store, backpressure та контроль часу життя даних.

## Реальні ролі 3 і 4

Контракти, які вони повинні споживати/публікувати:

- роль 4 читає `cv:video_chunks` як consumer group `cut-workers` і публікує `CutResultMessage` у `cv:cut_results`;
- coordinator читає `cv:cut_results` і тільки після цього створює `TrackingJobMessage` у `cv:tracking_jobs`;
- роль 3 читає `cv:tracking_jobs` як consumer group `tracking-workers` і публікує `TrackingResultMessage` у `cv:tracking_results`;
- aggregator збирає відповіді; ingest звільняє Ray-об’єкт після durable tracking completion.

Інтеграція вже реалізована в `participant_4_cut_detection` та
`participant_3_tracking`. Обидва воркери використовують спільний
`StreamWorker`, Pydantic-схеми бекенда, stale-message reclaim, retry/DLQ і
атомарну публікацію tracking-result. Сирий Ray `ObjectRef` не відновлюється з
рядка: воркери отримують його через named registry та передають вкладеним
посиланням у Ray Actor.

AutoShot checkpoint уже доданий у `models/weights_for_cut.pth`. Для повністю
offline tracking додайте `models/yolov8s-world.pt` згідно з `models/README.md`,
після чого запустіть **без** профілю `mock`:

```bash
cp .env.example .env
docker compose -f compose.yaml -f compose.ml.example.yaml \
  --profile ml up --build
```

`ml-ray-worker` є справжнім GPU Ray node. AutoShot і YOLO-World actors
плануються тільки на ньому через custom resources. Cut-worker читає
`cv:video_chunks`, а tracking-worker — лише `cv:tracking_jobs`; тому coordinator
не обходиться. Tracking actor розділений за `task_id`, кешує повторну обробку та
зберігає checkpoint ByteTrack-стану в Redis між chunk-ами.

Для стабільної роботи з 1080p/4K AutoShot спочатку зменшує кожен RGB-кадр до
власного входу `48x27` у форматі `uint8`, і лише після цього доповнює короткий
chunk до 100 кадрів та переводить дані у `float32`. Це усуває багатогігабайтне
тимчасове виділення пам’яті, через яке нативний Ray Actor міг завершуватися без
Python traceback. Реальний ML-профіль також тримає один in-flight chunk, один
cut-виклик і один tracking-виклик за замовчуванням; відео з ZIP залишаються
необмеженими за кількістю в межах розмірного бюджету, але проходять memory-bounded
чергу. Redis використовує 30-секундний socket timeout і п’ять повторів із
exponential backoff; короткий timeout залишає stream message pending замість
переведення всієї задачі у `failed`.

Точні поля, frame mapping і правила ownership описані в
`docs/WORKER_CONTRACTS.md`. Повний аналіз рішень та обмежень — у
`docs/ARCHITECTURE_AND_DECISIONS.md`.

## End-to-end smoke test у Windows PowerShell

Один скрипт перевіряє як окреме відео, так і ZIP-архів. Він завантажує файл,
очікує завершення кожного `task_id`, зупиняється на першій реальній помилці та
зберігає JSON у `smoke-results/`:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\backend\scripts\smoke_test.ps1 "C:\path\sample.mp4"
```

Для архіву команда така сама:

```powershell
.\backend\scripts\smoke_test.ps1 "C:\path\videos.zip"
```

Cut-only перевірка з реальною AutoShot і mock tracking, коли
`models/yolov8s-world.pt` ще відсутній:

```powershell
docker compose -f compose.yaml -f compose.ml.example.yaml --profile ml --profile mock up --build -d redis ray-head storage-init backend ingest-worker coordinator aggregator ml-ray-worker cut-worker mock-tracking-worker
.\backend\scripts\smoke_test.ps1 "C:\path\sample.mp4"
```

Повний реальний pipeline після додавання tracking checkpoint:

```powershell
docker compose -f compose.yaml -f compose.ml.example.yaml --profile ml up --build -d
.\backend\scripts\smoke_test.ps1 "C:\path\videos.zip"
```

Докладний життєвий цикл контейнерів, stream messages, Ray ObjectRef і повторного
запуску описано в `docs/REAL_MODEL_LIFECYCLE.md`.

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

Backend через окремий FFmpeg-процес послідовно формує до 64 унікальних
кадрів на chunk і додає до 16 попередніх кадрів як контекст. Фактичний розмір
автоматично зменшується, щоб RGB-тензор не перевищував
`MAX_DECODED_CHUNK_BYTES` (80 MiB у локальному preset). ML-воркер отримує
`context_start_frame/context_end_frame` та
`valid_start_frame/valid_end_frame`. Глобальний номер кадру дорівнює
`context_start_frame + local_index`; фінальні події дозволено публікувати лише
для valid range.

Ingest спочатку чекає вільного місця downstream і лише потім читає наступний
chunk із FFmpeg stdout у наперед виділений NumPy buffer. Після завершення задачі
FFmpeg-процес закривається, тому native codec buffers не накопичуються між
відео. Відновлення ingest запускає точний `trim=start_frame` для потрібного
`chunk_index`, не публікуючи вже збережені частини повторно. Redis pending-entry
heartbeat не дає іншому consumer-у забрати довге відео під час обробки.

## Важлива примітка щодо FPS

Існуючий `preprocess.py` має default `target_fps=25`, тоді як командна документація для стандартизованого датасету вказує 30 FPS. Backend навмисно **не використовує цей legacy-файл і не змінює FPS вхідного відео**: він зберігає оригінальні FPS та frame indices, щоб timestamps не “з’їжджали”. Роль 4 має узгодити власний model-specific resampling і повернути результати в координатах оригінального відео.

## Перевірки

```bash
python -m pip install -r backend/requirements/dev.txt
python -m ruff check --config backend/pyproject.toml backend/app backend/tests participant_3_tracking participant_4_cut_detection
python -m black --check --config backend/pyproject.toml backend/app backend/tests participant_3_tracking participant_4_cut_detection
python -m pytest -c backend/pyproject.toml backend/tests
python -m compileall -q backend/app backend/tests participant_3_tracking participant_4_cut_detection
docker compose config --quiet
docker compose -f compose.yaml -f compose.ml.example.yaml config --quiet
```

## Production notes

- Compose з окремими процесами через Ray Client перевіряє контракти та порядок, але сам по собі не гарантує фізичний zero-copy між довільними контейнерами/вузлами. Для H200 decode і ML Ray Actors треба розмістити на одному Ray node; деталі є в `docs/ARCHITECTURE_AND_DECISIONS.md`.
- Змініть пароль Grafana у `.env`.
- Не вмикайте `mock` profile у production. Перед ML-запуском перевірте bundled AutoShot checkpoint через `sha256sum -c models/SHA256SUMS` і додайте/налаштуйте tracking checkpoint згідно з `models/README.md`.
- `ALLOW_REMOTE_URLS=false` за замовчуванням. При увімкненні URL перевіряються проти private/loopback/reserved IP для зменшення SSRF-ризику.
- `RAY_SHM_SIZE` та `RAY_OBJECT_STORE_BYTES` треба підібрати під RAM сервера. Не задавайте їх “наосліп” як 30–50% без перевірки паралельності, розміру кадрів і запасу для ОС/воркерів.
- Redis і Ray не публікуються назовні; зовні доступний лише FastAPI, а monitoring-порти можна закрити reverse proxy/VPN.

## Throughput profile 1.1

The local mock pipeline keeps strict order inside one video while processing
independent task IDs concurrently downstream. Ingest decodes one video at a time;
cut, tracking, and aggregation default to two workers, while the lightweight
coordinator defaults to four. The real ML overlay deliberately reduces cut,
tracking, and global in-flight chunk concurrency to one on memory-constrained
Docker Desktop hosts.

The cut coordinator stores the cut payload, buffers out-of-order chunks, and
publishes all newly contiguous tracking jobs in one Redis Lua call. Tracking
completion and visible progress are also updated atomically. Final aggregation
fetches deterministic chunk hashes through one Redis pipeline instead of a
`SCAN` followed by one network request per chunk.

Local mock workers consume only the validated stream metadata and never call a
detached actor from a second Ray Client process. Ingest remains the owner of the
bounded Ray objects, observes durable tracking completion in Redis, and releases
completed chunks through its already-established registry handle. Every owner
registry call has a hard timeout, so failures become visible instead of blocking
forever at `cut_detection`.

Every final result now contains a `timings` object:

```json
{
  "queue_wait_ms": 0.0,
  "decoding_ms": 0.0,
  "cut_detection_ms": 0.0,
  "tracking_ms": 0.0,
  "aggregation_ms": 0.0,
  "orchestration_wait_ms": 0.0,
  "total_ms": 0.0
}
```

`total_ms` is wall-clock time from task creation. The decode, cut, and tracking
values are summed active chunk-processing times; with parallel work they may
overlap. `orchestration_wait_ms` is the non-negative remainder and helps reveal
Redis/Ray/backpressure overhead. The result contract version is now `1.1`.


### Why mock workers do not copy full RGB tensors

The local `mock` profile validates shape and frame-range metadata directly from
the strict Pydantic messages. Ingest still stores bounded frame tensors in Ray
and releases them only after Redis records tracking completion, so backpressure
and object lifetime remain covered. Avoiding Ray calls from mock cut, tracking,
and aggregation removes the cross-client blocking path that stalled Docker
Desktop. Owner-side registry calls are bounded by
`RAY_ACTOR_CALL_TIMEOUT_SECONDS`.
