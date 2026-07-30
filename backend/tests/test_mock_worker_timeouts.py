from __future__ import annotations

import time
from uuid import uuid4

import pytest

from app.core.config import Settings
from app.domain.schemas import FrameBatchMetadata
from app.infrastructure.redis_streams import StreamMessage
from app.infrastructure.serialization import model_to_stream_fields
from app.workers.mock_cut import MockCutHandler


class _Streams:
    def __init__(self) -> None:
        self.client = object()
        self.published: list[tuple[str, dict[str, str]]] = []

    async def publish(self, stream: str, fields: dict[str, str]) -> str:
        self.published.append((stream, fields))
        return "1-0"


def _message(task_id):
    batch = FrameBatchMetadata(
        task_id=task_id,
        chunk_id=f"{task_id}:00000000",
        chunk_index=0,
        object_ref="ray-object:test",
        context_start_frame=0,
        context_end_frame=7,
        valid_start_frame=0,
        valid_end_frame=7,
        frame_count=8,
        fps=24,
        width=1280,
        height=720,
        source_total_frames=8,
        is_last=True,
    )
    return StreamMessage(
        stream="cv:video_chunks",
        event_id="1-0",
        fields=model_to_stream_fields(batch),
    )


@pytest.mark.asyncio
async def test_mock_cut_does_not_open_a_second_ray_client() -> None:
    settings = Settings(_env_file=None, mock_worker_delay_ms=0)
    streams = _Streams()
    handler = MockCutHandler(settings, streams)  # type: ignore[arg-type]

    started = time.perf_counter()
    await handler(_message(uuid4()))

    assert time.perf_counter() - started < 0.1
    assert not hasattr(handler, "object_store")
    assert len(streams.published) == 1
    assert streams.published[0][0] == settings.stream_cut_results
