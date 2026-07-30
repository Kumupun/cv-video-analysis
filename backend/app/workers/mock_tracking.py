from __future__ import annotations

import asyncio
import time

from app.core.config import Settings
from app.domain.schemas import TrackingJobMessage, TrackingResultMessage
from app.infrastructure.redis_streams import RedisStreams, StreamMessage
from app.infrastructure.serialization import (
    model_to_stream_fields,
    stream_fields_to_model,
)
from app.infrastructure.task_repository import RedisTaskRepository
from app.workers.common import StreamWorker, run_worker


def _validate_job_metadata(job: TrackingJobMessage) -> None:
    expected_frame_count = job.context_end_frame - job.context_start_frame + 1
    if expected_frame_count <= 0:
        raise ValueError("Tracking job has an empty frame context")
    if not job.object_ref.startswith("ray-object:"):
        raise ValueError("Expected an opaque Ray object token")


class MockTrackingHandler:
    """Development-only contract adapter; it returns no detections."""

    def __init__(self, settings: Settings, streams: RedisStreams) -> None:
        self.settings = settings
        self.streams = streams
        self.repository = RedisTaskRepository(streams.client, settings)

    async def __call__(self, message: StreamMessage) -> None:
        job = stream_fields_to_model(message.fields, TrackingJobMessage)
        if not job.cut_verified:
            raise RuntimeError("Tracking cannot run before cut detection")

        started = time.perf_counter()
        _validate_job_metadata(job)

        if self.settings.mock_worker_delay_ms:
            await asyncio.sleep(self.settings.mock_worker_delay_ms / 1_000)
        tracking_processing_ms = (time.perf_counter() - started) * 1_000
        result = TrackingResultMessage(
            task_id=job.task_id,
            chunk_id=job.chunk_id,
            chunk_index=job.chunk_index,
            object_ref=job.object_ref,
            tracks=[],
            tracking_processing_ms=tracking_processing_ms,
        )
        await self.repository.publish_tracking_result_in_order(
            job.task_id,
            job.chunk_id,
            job.chunk_index,
            stream=self.settings.stream_tracking_results,
            stream_fields=model_to_stream_fields(result),
        )


def factory(settings: Settings, streams: RedisStreams) -> StreamWorker:
    return StreamWorker(
        worker_name="mock-tracking",
        stream=settings.stream_tracking_jobs,
        group=settings.group_tracking,
        handler=MockTrackingHandler(settings, streams),
        settings=settings,
        streams=streams,
        max_concurrency=settings.tracking_worker_concurrency,
        partition_by_task=True,
    )


if __name__ == "__main__":
    asyncio.run(run_worker(factory))
