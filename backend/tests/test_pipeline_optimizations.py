from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest

from app.core.config import Settings
from app.domain.schemas import (
    CutResultMessage,
    TrackingResultMessage,
    VideoMetadata,
)
from app.infrastructure.redis_streams import StreamMessage
from app.infrastructure.task_repository import RedisTaskRepository
from app.services.result_aggregation import aggregate_result
from app.workers.common import StreamWorker


@pytest.mark.asyncio
async def test_worker_parallelizes_tasks_but_preserves_each_task_order() -> None:
    worker = StreamWorker.__new__(StreamWorker)
    worker.max_concurrency = 2
    worker.partition_by_task = True
    active_tasks: set[str] = set()
    max_parallel = 0
    completed: list[str] = []

    async def process(message: StreamMessage) -> None:
        nonlocal max_parallel
        task_id = message.fields["task_id"]
        assert task_id not in active_tasks
        active_tasks.add(task_id)
        max_parallel = max(max_parallel, len(active_tasks))
        await asyncio.sleep(0.01)
        completed.append(message.event_id)
        active_tasks.remove(task_id)

    worker._process_message = process  # type: ignore[method-assign]
    messages = [
        StreamMessage("stream", "1-0", {"task_id": "task-a"}),
        StreamMessage("stream", "2-0", {"task_id": "task-a"}),
        StreamMessage("stream", "3-0", {"task_id": "task-b"}),
    ]

    await worker._process_batch(messages)

    assert max_parallel == 2
    assert completed.index("1-0") < completed.index("2-0")


class _EvalRedisStub:
    def __init__(self, result: list[int | str]) -> None:
        self.result = result
        self.calls: list[tuple[Any, ...]] = []

    async def eval(self, *args: Any) -> list[int | str]:
        self.calls.append(args)
        return self.result


@pytest.mark.asyncio
async def test_cut_persistence_and_contiguous_dispatch_use_one_lua_call() -> None:
    redis = _EvalRedisStub([1, 2, 2, 5, 2])
    repository = RedisTaskRepository(redis, Settings(_env_file=None))
    task_id = uuid4()

    result = await repository.record_cut_and_dispatch(
        task_id,
        f"{task_id}:00000001",
        1,
        cut_payload="{}",
        object_ref="ray-object:1",
        tracking_stream="cv:tracking_jobs",
        tracking_stream_fields={
            "event_type": "TrackingJobMessage",
            "schema_version": "1",
            "payload": "{}",
        },
    )

    assert result == (True, 2, 2, 5)
    assert len(redis.calls) == 1
    script = str(redis.calls[0][0])
    assert "while true do" in script
    assert "tracking_job_payload" in script
    assert "XADD" in script

    # Regression: the persisted chunk key and the key reconstructed by the
    # contiguous-dispatch Lua loop must be byte-for-byte identical.  A prior
    # optimization stored ``task_id`` twice in the hash key while Lua used it
    # once, so cut detection completed but no tracking job was ever emitted.
    current_chunk_key = str(redis.calls[0][3])
    lua_chunk_prefix = str(redis.calls[0][-2])
    assert f"{lua_chunk_prefix}00000001" == current_chunk_key


def test_chunk_key_does_not_duplicate_task_id() -> None:
    task_id = uuid4()

    assert (
        RedisTaskRepository._chunk_key(
            task_id,
            f"{task_id}:00000000",
        )
        == f"cv:chunk:{task_id}:00000000"
    )
    assert RedisTaskRepository._chunk_key_prefix(task_id) == f"cv:chunk:{task_id}:"


@pytest.mark.asyncio
async def test_tracking_completion_returns_total_from_same_lua_call() -> None:
    redis = _EvalRedisStub(["1", 1, 7, 12])
    repository = RedisTaskRepository(redis, Settings(_env_file=None))
    task_id = uuid4()

    result = await repository.mark_chunk_tracking_done(
        task_id,
        f"{task_id}:00000006",
        "{}",
    )

    assert result == (True, True, 7, 12)
    assert len(redis.calls) == 1
    script = str(redis.calls[0][0])
    assert "tracking_completed_chunks" in script
    assert "target_stage" in script


def test_final_result_reports_stage_and_wall_clock_timings() -> None:
    task_id = uuid4()
    created = datetime(2026, 7, 29, 10, 0, tzinfo=UTC)
    started = created + timedelta(milliseconds=100)
    completed = created + timedelta(milliseconds=1_000)
    cut = CutResultMessage(
        task_id=task_id,
        chunk_id="chunk-0",
        chunk_index=0,
        object_ref="token",
        fps=30,
        context_start_frame=0,
        context_end_frame=9,
        valid_start_frame=0,
        valid_end_frame=9,
        transitions=[],
        scenes=[],
        decode_ms=120,
        cut_processing_ms=80,
    )
    tracking = TrackingResultMessage(
        task_id=task_id,
        chunk_id="chunk-0",
        chunk_index=0,
        object_ref="token",
        tracks=[],
        tracking_processing_ms=200,
    )

    result = aggregate_result(
        task_id=task_id,
        video=VideoMetadata(
            fps=30,
            frame_count=10,
            width=640,
            height=360,
            duration_seconds=1 / 3,
        ),
        chunk_payloads=[
            {
                "cut_payload": cut.model_dump_json(),
                "tracking_payload": tracking.model_dump_json(),
            }
        ],
        created_at=created,
        processing_started_at=started,
        completed_at=completed,
        aggregation_ms=50,
    )

    assert result.pipeline_version == "1.1"
    assert result.timings.queue_wait_ms == pytest.approx(100)
    assert result.timings.decoding_ms == pytest.approx(120)
    assert result.timings.cut_detection_ms == pytest.approx(80)
    assert result.timings.tracking_ms == pytest.approx(200)
    assert result.timings.aggregation_ms == pytest.approx(50)
    assert result.timings.orchestration_wait_ms == pytest.approx(450)
    assert result.timings.total_ms == pytest.approx(1_000)


class _ChunkPipelineStub:
    def __init__(self) -> None:
        self.keys: list[str] = []
        self.execute_calls = 0

    def hgetall(self, key: str) -> _ChunkPipelineStub:
        self.keys.append(key)
        return self

    async def execute(self) -> list[dict[str, str]]:
        self.execute_calls += 1
        return [{"chunk_index": str(index)} for index, _ in enumerate(self.keys)]


class _PipelineRedisStub:
    def __init__(self) -> None:
        self.pipe = _ChunkPipelineStub()

    def pipeline(self, *, transaction: bool) -> _ChunkPipelineStub:
        assert transaction is False
        return self.pipe


@pytest.mark.asyncio
async def test_final_chunk_fetch_uses_one_redis_pipeline() -> None:
    redis = _PipelineRedisStub()
    repository = RedisTaskRepository(redis, Settings(_env_file=None))
    task_id = uuid4()

    payloads = await repository.get_all_chunk_payloads(task_id, total_chunks=4)

    assert len(payloads) == 4
    assert redis.pipe.execute_calls == 1
    assert len(redis.pipe.keys) == 4
