from __future__ import annotations

import asyncio

from app.core.config import Settings
from app.domain.schemas import TrackingJobMessage, TrackingResultMessage
from app.infrastructure.ray_store import RayObjectStore
from app.infrastructure.redis_streams import RedisStreams, StreamMessage
from app.infrastructure.serialization import (
    model_to_stream_fields,
    stream_fields_to_model,
)
from app.ray_runtime.mock_actors import get_mock_tracking_actor
from app.workers.common import StreamWorker, run_worker


class MockTrackingHandler:
    """Development-only actor adapter; it returns no detections."""

    def __init__(self, settings: Settings, streams: RedisStreams) -> None:
        self.settings = settings
        self.streams = streams
        self.object_store = RayObjectStore(settings)
        self._actor = None

    def _actor_handle(self):
        if self._actor is None:
            self._actor = get_mock_tracking_actor(self.object_store.ray)
        return self._actor

    async def __call__(self, message: StreamMessage) -> None:
        job = stream_fields_to_model(message.fields, TrackingJobMessage)
        if not job.cut_verified:
            raise RuntimeError("Tracking cannot run before cut detection")

        frame_ref = self.object_store.resolve_ref(job.object_ref)
        actor_result_ref = self._actor_handle().analyze.remote(
            str(job.task_id),
            job.chunk_index,
            frame_ref,
        )
        await asyncio.to_thread(self.object_store.ray.get, actor_result_ref)
        await asyncio.sleep(self.settings.mock_worker_delay_ms / 1_000)

        result = TrackingResultMessage(
            task_id=job.task_id,
            chunk_id=job.chunk_id,
            chunk_index=job.chunk_index,
            object_ref=job.object_ref,
            tracks=[],
        )
        await self.streams.publish(
            self.settings.stream_tracking_results,
            model_to_stream_fields(result),
        )


def factory(settings: Settings, streams: RedisStreams) -> StreamWorker:
    return StreamWorker(
        worker_name="mock-tracking",
        stream=settings.stream_tracking_jobs,
        group=settings.group_tracking,
        handler=MockTrackingHandler(settings, streams),
        settings=settings,
        streams=streams,
    )


if __name__ == "__main__":
    asyncio.run(run_worker(factory))
