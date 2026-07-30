from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import numpy as np
import pytest

from app.core.config import Settings
from app.domain.enums import SourceKind, TaskStage
from app.domain.schemas import VideoMetadata, VideoSource
from app.infrastructure.task_repository import RedisTaskRepository
from app.services.video_decoder import DecodedBatch
from app.workers import ingest as ingest_module
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


@pytest.mark.asyncio
async def test_ingest_drops_local_batch_without_losing_chunk_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = uuid4()
    published_chunk_indexes: list[int] = []

    class RepositoryStub:
        async def get_status(self, current_task_id: object) -> SimpleNamespace:
            assert current_task_id == task_id
            return SimpleNamespace(stage=TaskStage.QUEUED)

        async def is_ingest_complete(self, current_task_id: object) -> bool:
            assert current_task_id == task_id
            return False

        async def get_source(self, current_task_id: object) -> VideoSource:
            assert current_task_id == task_id
            return VideoSource(
                kind=SourceKind.UPLOAD,
                upload_path=str(Path("video.mp4")),
            )

        async def save_video_metadata(
            self,
            current_task_id: object,
            metadata: VideoMetadata,
        ) -> None:
            assert current_task_id == task_id
            assert metadata.frame_count == 2

        async def get_ingest_published_chunks(
            self,
            current_task_id: object,
        ) -> int:
            assert current_task_id == task_id
            return 0

        async def publish_ingest_chunk_once(
            self,
            current_task_id: object,
            chunk_id: str,
            **kwargs: Any,
        ) -> bool:
            assert current_task_id == task_id
            assert chunk_id.endswith(":00000000")
            published_chunk_indexes.append(int(kwargs["chunk_index"]))
            return True

        async def mark_ingest_complete(
            self,
            current_task_id: object,
            total_chunks: int,
        ) -> None:
            assert current_task_id == task_id
            assert total_chunks == 1

        async def update_status(self, *args: Any, **kwargs: Any) -> None:
            return None

        async def fail(self, *args: Any, **kwargs: Any) -> None:
            pytest.fail("A valid one-chunk ingest must not fail")

    class DecoderStub:
        total_chunks = 1

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            return None

        def open(self) -> VideoMetadata:
            return VideoMetadata(
                fps=30.0,
                frame_count=2,
                width=2,
                height=2,
                duration_seconds=2 / 30,
            )

        def decode_batch(self, chunk_index: int) -> DecodedBatch:
            assert chunk_index == 0
            return DecodedBatch(
                chunk_index=0,
                context_start_frame=0,
                context_end_frame=1,
                valid_start_frame=0,
                valid_end_frame=1,
                frames_rgb=np.zeros((2, 2, 2, 3), dtype=np.uint8),
                is_last=True,
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr(ingest_module, "FFmpegVideoDecoder", DecoderStub)

    handler = IngestHandler.__new__(IngestHandler)
    handler.settings = Settings(_env_file=None)
    handler.repository = RepositoryStub()
    handler._advance_stage_if_needed = AsyncMock()
    handler._wait_for_downstream_capacity = AsyncMock(return_value=(True, 0))
    handler._wait_for_downstream_drain = AsyncMock()
    handler._put_frames_with_retry = AsyncMock(return_value="ray-object:test")

    await handler._process_task(
        task_id,
        SimpleNamespace(event_id="1-0"),
    )

    assert published_chunk_indexes == [0]
