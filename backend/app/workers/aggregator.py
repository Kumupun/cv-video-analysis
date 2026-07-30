from __future__ import annotations

import asyncio
import time
from uuid import UUID

from app.core.config import Settings
from app.core.metrics import TASKS_COMPLETED
from app.domain.enums import TaskStage
from app.domain.schemas import TrackingResultMessage
from app.infrastructure.redis_streams import RedisStreams, StreamMessage
from app.infrastructure.task_repository import RedisTaskRepository
from app.infrastructure.serialization import stream_fields_to_model
from app.services.result_aggregation import aggregate_result
from app.workers.common import StreamWorker, run_worker


class AggregationHandler:
    def __init__(self, settings: Settings, streams: RedisStreams) -> None:
        self.settings = settings
        self.streams = streams
        self.repository = RedisTaskRepository(streams.client, settings)

    async def __call__(self, message: StreamMessage) -> None:
        tracking_result = stream_fields_to_model(message.fields, TrackingResultMessage)
        task_id = tracking_result.task_id
        (
            cut_done,
            is_new,
            completed_count,
            total,
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

        if total <= 0 or completed_count < total:
            return

        aggregation_started = time.perf_counter()
        status = await self.repository.get_status(task_id)
        chunk_payloads = await self.repository.get_all_chunk_payloads(
            task_id,
            total,
        )
        video = await self.repository.get_video_metadata(task_id)
        result = aggregate_result(
            task_id=UUID(str(task_id)),
            video=video,
            chunk_payloads=chunk_payloads,
            created_at=status.created_at,
            processing_started_at=status.processing_started_at,
        )
        aggregation_ms = (time.perf_counter() - aggregation_started) * 1_000
        timings = result.timings.model_copy(
            update={
                "aggregation_ms": aggregation_ms,
                "orchestration_wait_ms": max(
                    0.0,
                    result.timings.total_ms
                    - result.timings.queue_wait_ms
                    - result.timings.decoding_ms
                    - result.timings.cut_detection_ms
                    - result.timings.tracking_ms
                    - aggregation_ms,
                ),
            }
        )
        result = result.model_copy(update={"timings": timings})
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
        max_concurrency=settings.aggregator_worker_concurrency,
        partition_by_task=True,
    )


if __name__ == "__main__":
    asyncio.run(run_worker(factory))
