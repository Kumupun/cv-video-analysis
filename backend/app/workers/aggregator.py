from __future__ import annotations

import asyncio
from uuid import UUID

from app.core.config import Settings
from app.core.metrics import CHUNKS_RELEASED, TASKS_COMPLETED
from app.domain.enums import TaskStage
from app.domain.schemas import TrackingResultMessage
from app.infrastructure.ray_store import RayObjectStore
from app.infrastructure.redis_streams import RedisStreams, StreamMessage
from app.infrastructure.serialization import stream_fields_to_model
from app.infrastructure.task_repository import RedisTaskRepository
from app.services.result_aggregation import aggregate_result
from app.workers.common import StreamWorker, run_worker


class AggregationHandler:
    def __init__(self, settings: Settings, streams: RedisStreams) -> None:
        self.settings = settings
        self.streams = streams
        self.repository = RedisTaskRepository(streams.client, settings)
        self.object_store = RayObjectStore(settings)

    async def __call__(self, message: StreamMessage) -> None:
        tracking_result = stream_fields_to_model(message.fields, TrackingResultMessage)
        task_id = tracking_result.task_id
        (
            cut_done,
            is_new,
            completed_count,
        ) = await self.repository.mark_chunk_tracking_done(
            task_id,
            tracking_result.chunk_id,
            tracking_result.model_dump_json(),
        )
        if not cut_done:
            raise RuntimeError(
                "Tracking result arrived before a verified cut result; "
                "refusing to aggregate"
            )

        if not is_new:
            return

        await asyncio.to_thread(self.object_store.release, tracking_result.object_ref)
        CHUNKS_RELEASED.inc()

        status = await self.repository.get_status(task_id)
        total = max(status.total_chunks, 1)
        progress = 60.0 + 30.0 * min(completed_count / total, 1.0)
        await self.repository.update_status(
            task_id,
            stage=(
                TaskStage.AGGREGATING
                if completed_count >= total
                else TaskStage.TRACKING
            ),
            progress=progress,
            message=f"Tracking completed for {completed_count}/{total} chunks",
            tracking_completed_chunks=completed_count,
        )

        if completed_count < total:
            return

        chunk_payloads = await self.repository.get_all_chunk_payloads(task_id)
        video = await self.repository.get_video_metadata(task_id)
        result = aggregate_result(
            task_id=UUID(str(task_id)),
            video=video,
            chunk_payloads=chunk_payloads,
        )
        await self.repository.save_result(task_id, result)
        await self.repository.update_status(
            task_id,
            stage=TaskStage.COMPLETED,
            progress=100.0,
            message="Video analysis completed",
        )
        TASKS_COMPLETED.inc()


def factory(settings: Settings, streams: RedisStreams) -> StreamWorker:
    return StreamWorker(
        worker_name="aggregator",
        stream=settings.stream_tracking_results,
        group=settings.group_aggregation,
        handler=AggregationHandler(settings, streams),
        settings=settings,
        streams=streams,
    )


if __name__ == "__main__":
    asyncio.run(run_worker(factory))
