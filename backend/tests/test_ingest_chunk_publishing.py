from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from app.core.config import Settings
from app.infrastructure.task_repository import RedisTaskRepository
from app.workers.ingest import IngestHandler


class _RedisStub:
    def __init__(self, eval_results: list[int]) -> None:
        self.eval_results = eval_results
        self.eval_calls: list[tuple[Any, ...]] = []
        self.hashes: dict[tuple[str, str], str] = {}

    async def hget(self, key: str, field: str) -> str | None:
        return self.hashes.get((key, field))

    async def eval(self, script: str, *args: Any) -> int:
        self.eval_calls.append((script, *args))
        return self.eval_results.pop(0)


@pytest.mark.asyncio
async def test_ingest_chunk_is_enqueued_exactly_once() -> None:
    redis = _RedisStub([1, 0])
    repository = RedisTaskRepository(redis, Settings())
    task_id = uuid4()
    fields = {
        "event_type": "FrameBatchMetadata",
        "schema_version": "1",
        "payload": '{"chunk_index":0}',
    }

    first = await repository.publish_ingest_chunk_once(
        task_id,
        f"{task_id}:00000000",
        stream="cv:video_chunks",
        stream_fields=fields,
        object_ref="ray-object:first",
        chunk_index=0,
    )
    duplicate = await repository.publish_ingest_chunk_once(
        task_id,
        f"{task_id}:00000000",
        stream="cv:video_chunks",
        stream_fields=fields,
        object_ref="ray-object:duplicate",
        chunk_index=0,
    )

    assert first is True
    assert duplicate is False
    assert len(redis.eval_calls) == 2
    assert "XADD" in redis.eval_calls[0][0]


@pytest.mark.asyncio
async def test_ingest_completion_requires_all_chunks() -> None:
    redis = _RedisStub([0])
    repository = RedisTaskRepository(redis, Settings())
    task_id = uuid4()
    redis.hashes[(f"cv:task:{task_id}", "ingest_published_chunks")] = "1"

    with pytest.raises(RuntimeError, match="1/2 chunks"):
        await repository.mark_ingest_complete(task_id, 2)


def test_first_published_chunk_updates_visible_progress() -> None:
    handler = IngestHandler.__new__(IngestHandler)
    handler.settings = Settings(_env_file=None, progress_update_every_chunks=5)

    assert handler._should_publish_progress(1, 20) is True
    assert handler._should_publish_progress(2, 20) is False
    assert handler._should_publish_progress(5, 20) is True
