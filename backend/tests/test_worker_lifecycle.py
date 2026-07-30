from __future__ import annotations

import asyncio
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


def test_task_id_is_extracted_from_direct_and_serialized_messages() -> None:
    direct = common.StreamMessage(
        stream="cv:ingest_jobs",
        event_id="1-0",
        fields={"task_id": "direct-task"},
    )
    serialized = common.StreamMessage(
        stream="cv:video_chunks",
        event_id="2-0",
        fields={"payload": '{"task_id":"payload-task"}'},
    )

    assert common._extract_task_id(direct) == "direct-task"
    assert common._extract_task_id(serialized) == "payload-task"


@pytest.mark.asyncio
async def test_terminal_dead_letter_marks_task_failed() -> None:
    calls: list[tuple[str, str, str]] = []

    class RepositoryStub:
        async def fail(self, task_id: str, *, code: str, detail: str) -> None:
            calls.append((task_id, code, detail))

    worker = common.StreamWorker.__new__(common.StreamWorker)
    worker.worker_name = "mock-cut"
    worker.repository = RepositoryStub()
    message = common.StreamMessage(
        stream="cv:video_chunks",
        event_id="3-0",
        fields={"payload": '{"task_id":"failed-task"}'},
    )

    await worker._mark_task_failed(message, RuntimeError("Ray actor unavailable"))

    assert calls == [
        ("failed-task", "mock_cut_failed", "Ray actor unavailable"),
    ]


@pytest.mark.asyncio
async def test_worker_retries_transient_redis_poll_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep_calls: list[float] = []

    class StreamsStub:
        def __init__(self) -> None:
            self.claim_calls = 0

        async def wait_until_ready(self) -> None:
            return None

        async def ensure_group(self, stream: str, group: str) -> None:
            return None

        async def claim_stale(self, **kwargs):
            self.claim_calls += 1
            if self.claim_calls == 1:
                raise TimeoutError("temporary Redis timeout")
            raise asyncio.CancelledError

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    streams = StreamsStub()
    worker = common.StreamWorker.__new__(common.StreamWorker)
    worker.worker_name = "test-worker"
    worker.stream = "cv:test"
    worker.group = "test-group"
    worker.consumer = "test-consumer"
    worker.max_concurrency = 1
    worker.partition_by_task = False
    worker.streams = streams
    worker.settings = SimpleNamespace(
        stream_batch_size=8,
        stream_retry_delay_seconds=0.5,
    )

    monkeypatch.setattr(common.asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await worker.run_forever()

    assert streams.claim_calls == 2
    assert sleep_calls == [0.5]


@pytest.mark.asyncio
async def test_long_running_message_refreshes_pending_idle_timer() -> None:
    touched = asyncio.Event()
    calls: list[tuple[str, str, str]] = []

    class StreamsStub:
        async def touch_pending(
            self,
            message: common.StreamMessage,
            *,
            group: str,
            consumer: str,
        ) -> None:
            calls.append((message.event_id, group, consumer))
            touched.set()

    worker = common.StreamWorker.__new__(common.StreamWorker)
    worker.worker_name = "ingest"
    worker.group = "ingest-workers"
    worker.consumer = "consumer-1"
    worker.streams = StreamsStub()
    worker.settings = SimpleNamespace(
        stream_processing_heartbeat_seconds=0.001,
    )
    message = common.StreamMessage(
        stream="cv:ingest_jobs",
        event_id="9-0",
        fields={"task_id": "task-1"},
    )

    heartbeat = asyncio.create_task(worker._heartbeat_pending(message))
    await asyncio.wait_for(touched.wait(), timeout=1.0)
    heartbeat.cancel()
    await asyncio.gather(heartbeat, return_exceptions=True)

    expected_call = ("9-0", "ingest-workers", "consumer-1")
    assert calls
    assert all(call == expected_call for call in calls)
