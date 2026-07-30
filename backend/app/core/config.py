from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "CV Video Analysis API"
    app_env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"

    redis_url: str = "redis://redis:6379/0"
    ray_address: str = "ray://ray-head:10001"

    upload_dir: Path = Path("/data/uploads")
    result_dir: Path = Path("/data/results")
    # Archive admission is primarily size-based. A single video may use the
    # whole 4 GiB batch budget, while many small videos may share it.
    max_upload_bytes: int = 4 * 1024 * 1024 * 1024
    max_archive_upload_bytes: int = 4 * 1024 * 1024 * 1024
    max_archive_video_bytes: int = 4 * 1024 * 1024 * 1024
    max_archive_uncompressed_bytes: int = 8 * 1024 * 1024 * 1024
    max_archive_members: int = Field(default=2_000, ge=1, le=10_000)
    max_archive_compression_ratio: float = Field(default=100.0, ge=1.0)
    archive_disk_reserve_bytes: int = Field(
        default=512 * 1024 * 1024,
        ge=0,
    )
    allowed_video_extensions: tuple[str, ...] = (
        ".mp4",
        ".mov",
        ".mkv",
        ".avi",
        ".webm",
        ".m4v",
    )
    allowed_video_mime_prefix: str = "video/"
    chunk_size_frames: int = Field(default=64, ge=8, le=1024)
    chunk_overlap_frames: int = Field(default=16, ge=0, le=512)
    max_decoded_chunk_bytes: int = Field(
        default=80 * 1024 * 1024,
        ge=8 * 1024 * 1024,
    )
    max_inflight_chunks_per_task: int = Field(default=2, ge=1, le=128)
    max_inflight_chunks_global: int = Field(default=2, ge=1, le=512)
    ingest_backpressure_poll_seconds: float = Field(default=0.1, gt=0.0, le=10.0)
    downstream_stall_timeout_seconds: float = Field(
        default=120.0,
        gt=0.0,
        le=3_600.0,
    )
    video_probe_timeout_seconds: float = Field(default=900.0, gt=0.0)
    decode_batch_timeout_seconds: float = Field(default=120.0, gt=0.0)
    ffmpeg_decode_threads: int = Field(default=1, ge=1, le=16)
    ray_put_timeout_seconds: float = Field(default=120.0, gt=0.0)
    ray_put_max_attempts: int = Field(default=3, ge=1, le=10)
    ray_put_retry_delay_seconds: float = Field(default=1.0, gt=0.0, le=30.0)
    ray_actor_call_timeout_seconds: float = Field(
        default=30.0,
        gt=0.0,
        le=600.0,
    )

    allow_remote_urls: bool = False
    remote_download_timeout_seconds: float = 60.0
    remote_download_max_redirects: int = 3

    stream_ingest_jobs: str = "cv:ingest_jobs"
    stream_video_chunks: str = "cv:video_chunks"
    stream_cut_results: str = "cv:cut_results"
    stream_tracking_jobs: str = "cv:tracking_jobs"
    stream_tracking_results: str = "cv:tracking_results"
    stream_dead_letters: str = "cv:dead_letters"

    group_ingest: str = "ingest-workers"
    group_cut: str = "cut-workers"
    group_coordinator: str = "pipeline-coordinator"
    group_tracking: str = "tracking-workers"
    group_aggregation: str = "result-aggregator"

    stream_block_ms: int = 2_000
    stream_batch_size: int = 8
    stream_claim_idle_ms: int = 60_000
    stream_processing_heartbeat_seconds: float = Field(
        default=15.0,
        gt=0.0,
        le=300.0,
    )
    stream_max_delivery_attempts: int = 3
    # Retry the same message before moving to later stream entries. This is
    # essential for stateful stages such as tracking, where processing chunk
    # N+1 after a temporary failure of chunk N would violate ordering.
    stream_retry_delay_seconds: float = Field(default=0.5, ge=0.0, le=30.0)
    stream_maxlen: int = 50_000

    # Workers process different task IDs concurrently while preserving strict
    # ordering inside each individual video partition.
    # Archive entries are admitted without a count limit, but local decoding is
    # intentionally queued one video at a time. This keeps RAM bounded for
    # multi-gigabyte archives while downstream cut/tracking workers remain
    # concurrent. Scale ingest as separate processes on larger hosts.
    ingest_worker_concurrency: int = Field(default=1, ge=1, le=16)
    cut_worker_concurrency: int = Field(default=2, ge=1, le=32)
    coordinator_worker_concurrency: int = Field(default=4, ge=1, le=64)
    tracking_worker_concurrency: int = Field(default=2, ge=1, le=32)
    aggregator_worker_concurrency: int = Field(default=2, ge=1, le=32)

    # Status reads are retried because a short Redis/network hiccup should not
    # surface as a transient HTTP 500 while the task keeps processing.
    status_read_max_attempts: int = Field(default=3, ge=1, le=10)
    status_read_retry_delay_seconds: float = Field(default=0.05, ge=0.0, le=5.0)

    task_ttl_seconds: int = 7 * 24 * 60 * 60
    result_ttl_seconds: int = 7 * 24 * 60 * 60
    worker_heartbeat_ttl_seconds: int = 30
    worker_metrics_port: int = Field(default=9101, ge=1024, le=65535)

    mock_cut_threshold: float = 0.85
    mock_worker_delay_ms: int = Field(default=0, ge=0, le=60_000)
    progress_update_every_chunks: int = Field(default=5, ge=1, le=1_000)

    prometheus_multiproc_dir: Path | None = None

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def health_dependencies(self) -> tuple[str, ...]:
        return ("redis", "ray")

    def ensure_directories(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
