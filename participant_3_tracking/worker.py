from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from app.core.config import Settings
from app.domain.schemas import TrackingJobMessage, TrackingResultMessage
from app.infrastructure.ray_store import RayObjectStore
from app.infrastructure.redis_streams import RedisStreams, StreamMessage
from app.infrastructure.serialization import (
    model_to_stream_fields,
    stream_fields_to_model,
)
from app.infrastructure.task_repository import RedisTaskRepository
from app.workers.common import StreamWorker, run_worker

logger = logging.getLogger(__name__)

TRACKING_STATE_PREFIX = "cv:ml:yolo-ensemble-bytetrack-state:v3"
TRACKING_RESULT_CACHE_PREFIX = "cv:ml:yolo-ensemble-bytetrack-result:v3"


def _actor_name(task_id: Any) -> str:
    return f"yolo-ensemble-tracking-v3-{str(task_id).replace('-', '')}"


def get_tracking_actor(
    ray: Any,
    settings: Settings,
    task_id: Any,
    classes: tuple[str, ...],
) -> Any:
    actor_options: dict[str, Any] = {
        "num_gpus": settings.tracking_model_num_gpus,
        "max_restarts": 1,
        "max_task_retries": 0,
    }
    if settings.tracking_actor_resource:
        actor_options["resources"] = {settings.tracking_actor_resource: 0.001}

    @ray.remote(**actor_options)
    class TrackingInferenceActor:
        def __init__(
            self,
            model_id: str,
            ensemble_model_id: str | None,
            ensemble_iou_threshold: float,
            parallel_inference: bool,
            classes: tuple[str, ...],
            confidence: float,
            image_size: int,
            allow_download: bool,
            track_low_threshold: float,
            new_track_threshold: float,
            track_buffer: int,
            match_threshold: float,
            fuse_score: bool,
        ) -> None:
            from participant_3_tracking.yolo_world_tracker import YoloWorldTracker

            self.tracker = YoloWorldTracker(
                model_id=model_id,
                ensemble_model_id=ensemble_model_id,
                ensemble_iou_threshold=ensemble_iou_threshold,
                parallel_inference=parallel_inference,
                classes=classes,
                confidence=confidence,
                image_size=image_size,
                allow_download=allow_download,
                track_low_threshold=track_low_threshold,
                new_track_threshold=new_track_threshold,
                track_buffer=track_buffer,
                match_threshold=match_threshold,
                fuse_score=fuse_score,
            )
            self.restored = False
            self.cache: dict[int, dict[str, Any]] = {}

        def process(
            self,
            wrapped_ref: list[Any],
            job_payload: dict[str, Any],
            checkpoint: str | None,
        ) -> dict[str, Any]:
            import ray as actor_ray

            from app.domain.schemas import TrackingJobMessage

            job = TrackingJobMessage.model_validate(job_payload)
            cached = self.cache.get(job.chunk_index)
            if cached is not None:
                if "checkpoint" not in cached:
                    cached["checkpoint"] = self.tracker.export_state()
                return cached
            if len(wrapped_ref) != 1:
                raise ValueError("Expected one nested Ray ObjectRef")
            if not self.restored:
                if checkpoint:
                    self.tracker.restore_state(checkpoint)
                self.restored = True

            frames = actor_ray.get(wrapped_ref[0])
            started = time.perf_counter()
            tracks = self.tracker.process_job(frames, job)
            response = {
                "tracks": tracks,
                "processing_ms": (time.perf_counter() - started) * 1_000,
            }




            self.cache[job.chunk_index] = response
            response["checkpoint"] = self.tracker.export_state()
            if len(self.cache) > 4:
                del self.cache[min(self.cache)]
            return response

    return TrackingInferenceActor.options(
        name=_actor_name(task_id),
        namespace=settings.tracking_actor_namespace,
        lifetime="detached",
        get_if_exists=True,
    ).remote(
        settings.tracking_model_id,
        settings.tracking_ensemble_model_id,
        settings.tracking_ensemble_iou_threshold,
        settings.tracking_parallel_inference,
        classes,
        settings.tracking_model_confidence,
        settings.tracking_model_image_size,
        settings.tracking_model_allow_download,
        settings.tracking_track_low_threshold,
        settings.tracking_new_track_threshold,
        settings.tracking_track_buffer,
        settings.tracking_match_threshold,
        settings.tracking_fuse_score,
    )


class TrackingActorManager:
    def __init__(self, object_store: RayObjectStore, settings: Settings) -> None:
        self.object_store = object_store
        self.settings = settings
        self._actors: dict[str, Any] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def get(self, task_id: Any, classes: tuple[str, ...]) -> Any:
        key = str(task_id)
        existing = self._actors.get(key)
        if existing is not None:
            return existing
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            existing = self._actors.get(key)
            if existing is None:
                existing = await asyncio.to_thread(
                    get_tracking_actor,
                    self.object_store.ray,
                    self.settings,
                    task_id,
                    classes,
                )
                self._actors[key] = existing
        return existing

    async def release(self, task_id: Any) -> None:
        key = str(task_id)
        actor = self._actors.pop(key, None)
        self._locks.pop(key, None)
        if actor is None:
            try:
                actor = await asyncio.to_thread(
                    self.object_store.ray.get_actor,
                    _actor_name(task_id),
                    namespace=self.settings.tracking_actor_namespace,
                )
            except Exception:
                return
        await asyncio.to_thread(self.object_store.ray.kill, actor, no_restart=True)


class RealTrackingHandler:
    def __init__(self, settings: Settings, streams: RedisStreams) -> None:
        self.settings = settings
        self.streams = streams
        self.repository = RedisTaskRepository(streams.client, settings)
        self.object_store = RayObjectStore(settings)
        self.actors = TrackingActorManager(self.object_store, settings)

    @staticmethod
    def _state_key(task_id: Any) -> str:
        return f"{TRACKING_STATE_PREFIX}:{task_id}"

    @staticmethod
    def _cache_key(task_id: Any, chunk_index: int) -> str:
        return f"{TRACKING_RESULT_CACHE_PREFIX}:{task_id}:{chunk_index:08d}"

    async def _infer(
        self,
        job: TrackingJobMessage,
        checkpoint: str | None,
    ) -> dict[str, Any]:
        object_ref = await asyncio.to_thread(
            self.object_store.resolve_ref,
            job.object_ref,
        )
        classes = (
            job.tracking_classes
            if job.tracking_classes is not None
            else self.settings.tracking_model_classes
        )
        actor = await self.actors.get(job.task_id, classes)
        result_ref = actor.process.remote(
            [object_ref],
            job.model_dump(mode="json"),
            checkpoint,
        )
        return await asyncio.to_thread(
            self.object_store.ray.get,
            result_ref,
            timeout=self.settings.ml_inference_timeout_seconds,
        )

    async def _persist_internal_state(
        self,
        result: TrackingResultMessage,
        checkpoint: str,
    ) -> None:
        pipeline = self.streams.client.pipeline(transaction=True)
        pipeline.set(
            self._cache_key(result.task_id, result.chunk_index),
            result.model_dump_json(),
            ex=self.settings.task_ttl_seconds,
        )
        pipeline.set(
            self._state_key(result.task_id),
            checkpoint,
            ex=self.settings.task_ttl_seconds,
        )
        await pipeline.execute()

    async def __call__(self, message: StreamMessage) -> None:
        job = stream_fields_to_model(message.fields, TrackingJobMessage)
        if not job.cut_verified:
            raise RuntimeError("Tracking cannot run before verified cut detection")

        cached = await self.streams.client.get(
            self._cache_key(job.task_id, job.chunk_index)
        )
        if cached:
            result = TrackingResultMessage.model_validate_json(cached)
        elif job.tracking_classes == ():
            result = TrackingResultMessage(
                task_id=job.task_id,
                chunk_id=job.chunk_id,
                chunk_index=job.chunk_index,
                object_ref=job.object_ref,
                tracks=[],
            )
            await self.streams.client.set(
                self._cache_key(job.task_id, job.chunk_index),
                result.model_dump_json(),
                ex=self.settings.task_ttl_seconds,
            )
        else:
            checkpoint = await self.streams.client.get(self._state_key(job.task_id))
            inference = await self._infer(job, checkpoint)
            result = TrackingResultMessage(
                task_id=job.task_id,
                chunk_id=job.chunk_id,
                chunk_index=job.chunk_index,
                object_ref=job.object_ref,
                tracks=list(inference.get("tracks", [])),
                tracking_processing_ms=max(
                    0.0,
                    float(inference.get("processing_ms", 0.0)),
                ),
            )
            next_checkpoint = inference.get("checkpoint")
            if not isinstance(next_checkpoint, str) or not next_checkpoint:
                raise RuntimeError("Tracking actor did not return a state checkpoint")
            await self._persist_internal_state(result, next_checkpoint)

        await self.repository.publish_tracking_result_in_order(
            job.task_id,
            job.chunk_id,
            job.chunk_index,
            stream=self.settings.stream_tracking_results,
            stream_fields=model_to_stream_fields(result),
        )
        if job.is_last:
            await self.actors.release(job.task_id)
            await self.streams.client.delete(self._state_key(job.task_id))

        logger.info(
            "YOLO ensemble + ByteTrack tracking chunk completed",
            extra={
                "task_id": str(job.task_id),
                "chunk_index": job.chunk_index,
                "track_count": len(result.tracks),
            },
        )


def factory(settings: Settings, streams: RedisStreams) -> StreamWorker:
    return StreamWorker(
        worker_name="yolo-ensemble-bytetrack-tracking",
        stream=settings.stream_tracking_jobs,
        group=settings.group_tracking,
        handler=RealTrackingHandler(settings, streams),
        settings=settings,
        streams=streams,
        max_concurrency=settings.tracking_worker_concurrency,
        partition_by_task=True,
    )


if __name__ == "__main__":
    asyncio.run(run_worker(factory))
