from __future__ import annotations

import asyncio

from app.core.config import Settings
from app.domain.schemas import CutResultMessage, TrackingJobMessage
from app.infrastructure.redis_streams import RedisStreams, StreamMessage
from app.infrastructure.task_repository import RedisTaskRepository
from app.infrastructure.serialization import (
    model_to_stream_fields,
    stream_fields_to_model,
)
from app.workers.common import StreamWorker, run_worker


class CutCoordinatorHandler:
    def __init__(self, settings: Settings, streams: RedisStreams) -> None:
        self.settings = settings
        self.streams = streams
        self.repository = RedisTaskRepository(streams.client, settings)

    async def __call__(self, message: StreamMessage) -> None:
        cut_result = stream_fields_to_model(message.fields, CutResultMessage)
        tracking_job = TrackingJobMessage(
            task_id=cut_result.task_id,
            chunk_id=cut_result.chunk_id,
            chunk_index=cut_result.chunk_index,
            object_ref=cut_result.object_ref,
            fps=cut_result.fps,
            context_start_frame=cut_result.context_start_frame,
            context_end_frame=cut_result.context_end_frame,
            valid_start_frame=cut_result.valid_start_frame,
            valid_end_frame=cut_result.valid_end_frame,
            scenes=cut_result.scenes,
            transitions=cut_result.transitions,
            is_last=cut_result.is_last,
            cut_verified=True,
        )
        await self.repository.record_cut_and_dispatch(
            cut_result.task_id,
            cut_result.chunk_id,
            cut_result.chunk_index,
            cut_payload=cut_result.model_dump_json(),
            object_ref=cut_result.object_ref,
            tracking_stream=self.settings.stream_tracking_jobs,
            tracking_stream_fields=model_to_stream_fields(tracking_job),
        )


def factory(settings: Settings, streams: RedisStreams) -> StreamWorker:
    return StreamWorker(
        worker_name="coordinator",
        stream=settings.stream_cut_results,
        group=settings.group_coordinator,
        handler=CutCoordinatorHandler(settings, streams),
        settings=settings,
        streams=streams,
        max_concurrency=settings.coordinator_worker_concurrency,
        partition_by_task=True,
    )


if __name__ == "__main__":
    asyncio.run(run_worker(factory))
