from app.core.config import PROJECT_ROOT, Settings


def test_local_performance_defaults_are_memory_bounded() -> None:
    settings = Settings(_env_file=None)

    assert settings.max_decoded_chunk_bytes == 80 * 1024 * 1024
    assert settings.max_inflight_chunks_per_task == 2
    assert settings.max_inflight_chunks_global == 2
    assert settings.mock_worker_delay_ms == 0
    assert settings.progress_update_every_chunks == 5
    assert settings.ingest_worker_concurrency == 1
    assert settings.cut_worker_concurrency == 2
    assert settings.tracking_worker_concurrency == 2
    assert settings.ray_actor_call_timeout_seconds == 30
    assert settings.downstream_stall_timeout_seconds == 120
    assert settings.ffmpeg_decode_threads == 1
    assert settings.video_probe_timeout_seconds == 900
    assert settings.stream_processing_heartbeat_seconds == 15


def test_local_env_queues_archive_decoding_safely() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        env_path = PROJECT_ROOT / ".env.example"

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()

    assert values["MAX_INFLIGHT_CHUNKS_PER_TASK"] == "1"
    assert values["MAX_INFLIGHT_CHUNKS_GLOBAL"] == "2"
    assert values["INGEST_WORKER_CONCURRENCY"] == "1"
    assert values["CUT_WORKER_CONCURRENCY"] == "2"
    assert values["TRACKING_WORKER_CONCURRENCY"] == "2"
    assert values["AGGREGATOR_WORKER_CONCURRENCY"] == "2"
