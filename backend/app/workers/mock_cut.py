from __future__ import annotations

import asyncio
import time

from app.core.config import Settings
from app.domain.schemas import CutResultMessage, FrameBatchMetadata, SceneInterval
from app.infrastructure.redis_streams import RedisStreams, StreamMessage
from app.infrastructure.serialization import (
    model_to_stream_fields,
    stream_fields_to_model,
)
from app.workers.common import StreamWorker, run_worker


def _validate_batch_metadata(batch: FrameBatchMetadata) -> None:
    """Validate the mock contract without opening a second Ray Client path.

    Ingest has already decoded the tensor, stored it in Ray and persisted the
    exact context range. The local mock worker does not run a model, so fetching
    the same tensor through a detached actor only adds a cross-client blocking
    point and provides no extra contract coverage.
    """

    expected_frame_count = batch.context_end_frame - batch.context_start_frame + 1
    if batch.frame_count != expected_frame_count:
        raise ValueError(
            "Frame metadata is inconsistent: "
            f"frame_count={batch.frame_count}, context={expected_frame_count}"
        )
    if not batch.object_ref.startswith("ray-object:"):
        raise ValueError("Expected an opaque Ray object token")


class MockCutHandler:
    """Development-only contract adapter; it is not a cut model."""

    def __init__(self, settings: Settings, streams: RedisStreams) -> None:
        self.settings = settings
        self.streams = streams

    async def __call__(self, message: StreamMessage) -> None:
        batch = stream_fields_to_model(message.fields, FrameBatchMetadata)
        started = time.perf_counter()
        _validate_batch_metadata(batch)

        if self.settings.mock_worker_delay_ms:
            await asyncio.sleep(self.settings.mock_worker_delay_ms / 1_000)
        cut_processing_ms = (time.perf_counter() - started) * 1_000
        result = CutResultMessage(
            task_id=batch.task_id,
            chunk_id=batch.chunk_id,
            chunk_index=batch.chunk_index,
            object_ref=batch.object_ref,
            fps=batch.fps,
            context_start_frame=batch.context_start_frame,
            context_end_frame=batch.context_end_frame,
            valid_start_frame=batch.valid_start_frame,
            valid_end_frame=batch.valid_end_frame,
            transitions=[],
            scenes=[
                SceneInterval(
                    scene_id=f"{batch.task_id}:scene:{batch.chunk_index}:0",
                    start_frame=batch.valid_start_frame,
                    end_frame=batch.valid_end_frame,
                )
            ],
            is_last=batch.is_last,
            decode_ms=batch.decode_ms,
            cut_processing_ms=cut_processing_ms,
        )
        await self.streams.publish(
            self.settings.stream_cut_results,
            model_to_stream_fields(result),
        )


def factory(settings: Settings, streams: RedisStreams) -> StreamWorker:
    return StreamWorker(
        worker_name="mock-cut",
        stream=settings.stream_video_chunks,
        group=settings.group_cut,
        handler=MockCutHandler(settings, streams),
        settings=settings,
        streams=streams,
        max_concurrency=settings.cut_worker_concurrency,
        partition_by_task=True,
    )


if __name__ == "__main__":
    asyncio.run(run_worker(factory))
