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
    max_upload_bytes: int = 2 * 1024 * 1024 * 1024
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
    stream_max_delivery_attempts: int = 3
    stream_maxlen: int = 50_000

    task_ttl_seconds: int = 7 * 24 * 60 * 60
    result_ttl_seconds: int = 7 * 24 * 60 * 60
    worker_heartbeat_ttl_seconds: int = 30
    worker_metrics_port: int = Field(default=9101, ge=1024, le=65535)

    mock_cut_threshold: float = 0.85
    mock_worker_delay_ms: int = 50

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
