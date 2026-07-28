from __future__ import annotations

import asyncio

from app.core.config import Settings
from app.domain.schemas import CutResultMessage, FrameBatchMetadata, SceneInterval
from app.infrastructure.ray_store import RayObjectStore
from app.infrastructure.redis_streams import RedisStreams, StreamMessage
from app.infrastructure.serialization import (
    model_to_stream_fields,
    stream_fields_to_model,
)
from app.ray_runtime.mock_actors import get_mock_cut_actor
from app.workers.common import StreamWorker, run_worker


class MockCutHandler:
    """Development-only actor adapter; it is not a cut-detection model."""

    def __init__(self, settings: Settings, streams: RedisStreams) -> None:
        self.settings = settings
        self.streams = streams
        self.object_store = RayObjectStore(settings)
        self._actor = None

    def _actor_handle(self):
        if self._actor is None:
            self._actor = get_mock_cut_actor(self.object_store.ray)
        return self._actor

    async def __call__(self, message: StreamMessage) -> None:
        batch = stream_fields_to_model(message.fields, FrameBatchMetadata)
        frame_ref = self.object_store.resolve_ref(batch.object_ref)
        actor_result_ref = self._actor_handle().analyze.remote(frame_ref)
        actor_result = await asyncio.to_thread(
            self.object_store.ray.get,
            actor_result_ref,
        )
        if actor_result["frame_count"] != batch.frame_count:
            raise ValueError("Ray Actor received an unexpected frame count")

        await asyncio.sleep(self.settings.mock_worker_delay_ms / 1_000)
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
    )


if __name__ == "__main__":
    asyncio.run(run_worker(factory))
