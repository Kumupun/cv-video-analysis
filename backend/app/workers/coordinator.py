from __future__ import annotations

import asyncio
from uuid import UUID

from app.core.config import Settings
from app.domain.enums import TaskStage
from app.domain.schemas import CutResultMessage, TrackingJobMessage
from app.infrastructure.redis_streams import RedisStreams, StreamMessage
from app.infrastructure.serialization import (
    model_to_stream_fields,
    stream_fields_to_model,
)
from app.infrastructure.task_repository import RedisTaskRepository
from app.workers.common import StreamWorker, run_worker


class CutCoordinatorHandler:
    def __init__(self, settings: Settings, streams: RedisStreams) -> None:
        self.settings = settings
        self.streams = streams
        self.repository = RedisTaskRepository(streams.client, settings)

    async def __call__(self, message: StreamMessage) -> None:
        cut_result = stream_fields_to_model(message.fields, CutResultMessage)
        task_id = cut_result.task_id
        _, cut_count = await self.repository.mark_chunk_cut_done(
            task_id,
            cut_result.chunk_id,
            cut_result.model_dump_json(),
            cut_result.object_ref,
        )

        dispatched_count = await self._dispatch_ready_chunks_in_order(task_id)
        status = await self.repository.get_status(task_id)
        total = max(status.total_chunks, 1)
        progress = 30.0 + 30.0 * min(cut_count / total, 1.0)
        target_stage = (
            TaskStage.TRACKING
            if dispatched_count > 0 and status.stage == TaskStage.CUT_DETECTION
            else status.stage
        )
        dispatch_detail = (
            f"tracking dispatched through chunk {dispatched_count - 1}"
            if dispatched_count > 0
            else "waiting for the first contiguous cut result"
        )
        await self.repository.update_status(
            task_id,
            stage=target_stage,
            progress=max(status.progress, progress),
            message=(
                f"Cut detection verified {cut_count}/{total} chunks; "
                f"{dispatch_detail}"
            ),
            cut_completed_chunks=cut_count,
        )

    async def _dispatch_ready_chunks_in_order(self, task_id: UUID) -> int:
        lock = self.streams.client.lock(
            f"cv:tracking_dispatch_lock:{task_id}",
            timeout=30,
            blocking_timeout=10,
        )
        async with lock:
            while True:
                next_index = await self.repository.get_next_tracking_dispatch_index(
                    task_id
                )
                payload = await self.repository.get_cut_payload_for_index(
                    task_id,
                    next_index,
                )
                if payload is None:
                    return next_index

                cut_result = CutResultMessage.model_validate_json(payload)
                tracking_job = TrackingJobMessage(
                    task_id=task_id,
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
                    cut_verified=True,
                )
                await self.streams.publish(
                    self.settings.stream_tracking_jobs,
                    model_to_stream_fields(tracking_job),
                )
                advanced = await self.repository.advance_tracking_dispatch(
                    task_id,
                    next_index,
                )
                if not advanced:
                    return await self.repository.get_next_tracking_dispatch_index(
                        task_id
                    )


def factory(settings: Settings, streams: RedisStreams) -> StreamWorker:
    return StreamWorker(
        worker_name="coordinator",
        stream=settings.stream_cut_results,
        group=settings.group_coordinator,
        handler=CutCoordinatorHandler(settings, streams),
        settings=settings,
        streams=streams,
    )


if __name__ == "__main__":
    asyncio.run(run_worker(factory))
