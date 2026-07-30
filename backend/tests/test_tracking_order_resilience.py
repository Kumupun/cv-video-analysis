from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from app.core.config import Settings
from app.infrastructure.redis_streams import StreamMessage
from app.infrastructure.task_repository import (
    RedisTaskRepository,
    TrackingOrderError,
)
from app.workers.common import StreamWorker


class _RetryStreamsStub:
    def __init__(self) -> None:
        self.register_calls = 0
        self.ack_calls = 0
        self.clear_calls = 0
        self.dead_letters = 0

    async def register_failure(self, message: StreamMessage) -> int:
        self.register_calls += 1
        return self.register_calls

    async def ack(self, message: StreamMessage, group: str) -> None:
        self.ack_calls += 1

    async def clear_failure_counter(self, message: StreamMessage) -> None:
        self.clear_calls += 1

    async def publish_dead_letter(self, **kwargs: Any) -> None:
        self.dead_letters += 1


@pytest.mark.asyncio
async def test_worker_retries_same_message_before_acknowledging() -> None:
    calls: list[str] = []

    async def handler(message: StreamMessage) -> None:
        calls.append(message.event_id)
        if len(calls) == 1:
            raise RuntimeError("temporary actor reconstruction")

    streams = _RetryStreamsStub()
    worker = StreamWorker.__new__(StreamWorker)
    worker.worker_name = "mock-tracking"
    worker.group = "tracking-workers"
    worker.handler = handler
    worker.streams = streams
    worker.settings = SimpleNamespace(
        stream_max_delivery_attempts=3,
        stream_retry_delay_seconds=0.0,
    )
    message = StreamMessage(
        stream="cv:tracking_jobs",
        event_id="1-0",
        fields={"payload": '{"task_id":"task-1"}'},
    )

    await worker._process_message(message)

    assert calls == ["1-0", "1-0"]
    assert streams.register_calls == 1
    assert streams.ack_calls == 1
    assert streams.clear_calls == 1
    assert streams.dead_letters == 0


class _EvalRedisStub:
    def __init__(self, result: list[int]) -> None:
        self.result = result
        self.calls: list[tuple[Any, ...]] = []

    async def eval(self, *args: Any) -> list[int]:
        self.calls.append(args)
        return self.result


@pytest.mark.asyncio
async def test_tracking_result_publish_is_atomic_and_idempotent() -> None:
    redis = _EvalRedisStub([1, 8])
    repository = RedisTaskRepository(redis, Settings(_env_file=None))
    task_id = uuid4()

    published = await repository.publish_tracking_result_in_order(
        task_id,
        f"{task_id}:00000007",
        7,
        stream="cv:tracking_results",
        stream_fields={
            "event_type": "tracking_result",
            "schema_version": "1",
            "payload": "{}",
        },
    )

    assert published is True
    assert len(redis.calls) == 1
    assert "next_tracking_result_chunk" in redis.calls[0][0]
    assert "XADD" in redis.calls[0][0]


@pytest.mark.asyncio
async def test_future_tracking_result_is_rejected_by_durable_sequence() -> None:
    redis = _EvalRedisStub([-1, 4])
    repository = RedisTaskRepository(redis, Settings(_env_file=None))
    task_id = uuid4()

    with pytest.raises(
        TrackingOrderError,
        match="expected chunk 4, received 6",
    ):
        await repository.publish_tracking_result_in_order(
            task_id,
            f"{task_id}:00000006",
            6,
            stream="cv:tracking_results",
            stream_fields={
                "event_type": "tracking_result",
                "schema_version": "1",
                "payload": "{}",
            },
        )
