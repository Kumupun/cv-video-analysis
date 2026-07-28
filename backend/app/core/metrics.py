from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

API_REQUESTS = Counter(
    "cv_api_requests_total",
    "Total API requests",
    ("method", "path", "status"),
)
TASKS_CREATED = Counter("cv_tasks_created_total", "Total accepted video tasks")
TASKS_FAILED = Counter("cv_tasks_failed_total", "Total failed video tasks", ("code",))
TASKS_COMPLETED = Counter("cv_tasks_completed_total", "Total completed video tasks")
TASK_STAGE = Gauge(
    "cv_tasks_by_stage",
    "Current number of tasks by stage",
    ("stage",),
)
CHUNKS_PUBLISHED = Counter(
    "cv_video_chunks_published_total", "Total decoded frame chunks published"
)
CHUNKS_RELEASED = Counter(
    "cv_ray_objects_released_total", "Total frame objects explicitly released"
)
WORKER_ERRORS = Counter(
    "cv_worker_errors_total", "Worker processing errors", ("worker",)
)
WORKER_UP = Gauge(
    "cv_worker_up", "Whether a pipeline worker process is running", ("worker",)
)
VIDEO_DECODE_SECONDS = Histogram(
    "cv_video_decode_seconds", "Time spent decoding one video"
)
CHUNK_PROCESS_SECONDS = Histogram(
    "cv_chunk_process_seconds",
    "Time spent processing one chunk",
    ("worker",),
)
