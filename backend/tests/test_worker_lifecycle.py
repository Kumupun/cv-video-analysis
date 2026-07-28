from __future__ import annotations

from types import SimpleNamespace

import prometheus_client
import pytest
from app.workers import common


@pytest.mark.asyncio
async def test_run_worker_connects_redis_before_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeStreams:
        def __init__(self, settings: object) -> None:
            events.append("streams_created")
            self.connected = False

        async def wait_until_ready(self) -> None:
            self.connected = True
            events.append("redis_connected")

        async def close(self) -> None:
            events.append("redis_closed")

    class FakeWorker:
        worker_name = "test-worker"

        async def run_forever(self) -> None:
            events.append("worker_started")

    def factory(settings: object, streams: FakeStreams) -> FakeWorker:
        assert streams.connected is True
        events.append("factory_called")
        return FakeWorker()

    monkeypatch.setattr(
        common,
        "get_settings",
        lambda: SimpleNamespace(log_level="INFO", worker_metrics_port=0),
    )
    monkeypatch.setattr(common, "configure_logging", lambda level: None)
    monkeypatch.setattr(common, "RedisStreams", FakeStreams)

    monkeypatch.setattr(prometheus_client, "start_http_server", lambda port: None)

    await common.run_worker(factory)

    assert events == [
        "streams_created",
        "redis_connected",
        "factory_called",
        "worker_started",
        "redis_closed",
    ]
