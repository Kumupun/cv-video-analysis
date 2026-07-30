from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
from app.core.config import Settings
from app.domain.enums import SourceKind
from app.domain.schemas import VideoSource
from app.infrastructure.task_repository import RedisTaskRepository


class _PipelineStub:
    def __init__(self) -> None:
        self.commands: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.execute_calls = 0

    def hset(self, *args: Any, **kwargs: Any) -> _PipelineStub:
        self.commands.append(("hset", args, kwargs))
        return self

    def expire(self, *args: Any, **kwargs: Any) -> _PipelineStub:
        self.commands.append(("expire", args, kwargs))
        return self

    def xadd(self, *args: Any, **kwargs: Any) -> _PipelineStub:
        self.commands.append(("xadd", args, kwargs))
        return self

    async def execute(self) -> list[Any]:
        self.execute_calls += 1
        return []


class _RedisStub:
    def __init__(self) -> None:
        self.pipeline_instance = _PipelineStub()

    def pipeline(self, *, transaction: bool) -> _PipelineStub:
        assert transaction is True
        return self.pipeline_instance


@pytest.mark.asyncio
async def test_archive_tasks_are_enqueued_in_one_redis_transaction() -> None:
    redis = _RedisStub()
    repository = RedisTaskRepository(redis, Settings())
    first_id = uuid4()
    second_id = uuid4()
    first_source = VideoSource(
        kind=SourceKind.UPLOAD,
        upload_path="/data/uploads/first/source.mp4",
    )
    second_source = VideoSource(
        kind=SourceKind.UPLOAD,
        upload_path="/data/uploads/second/source.mp4",
    )

    statuses = await repository.create_many_and_enqueue(
        [
            (first_id, first_source, {"task_id": str(first_id)}),
            (second_id, second_source, {"task_id": str(second_id)}),
        ],
        "cv:ingest_jobs",
    )

    commands = redis.pipeline_instance.commands
    assert [status.task_id for status in statuses] == [first_id, second_id]
    assert redis.pipeline_instance.execute_calls == 1
    assert [name for name, _, _ in commands].count("hset") == 2
    assert [name for name, _, _ in commands].count("expire") == 2
    assert [name for name, _, _ in commands].count("xadd") == 2
