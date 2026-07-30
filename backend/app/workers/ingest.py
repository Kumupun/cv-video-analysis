from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from uuid import UUID

from app.core.config import Settings
from app.core.metrics import (
    CHUNKS_PUBLISHED,
    CHUNKS_RELEASED,
    TASKS_FAILED,
    VIDEO_DECODE_SECONDS,
)
from app.domain.enums import SourceKind, TaskStage
from app.domain.schemas import FrameBatchMetadata, VideoSource
from app.infrastructure.ray_store import RayObjectStore
from app.infrastructure.redis_streams import RedisStreams, StreamMessage
from app.infrastructure.serialization import model_to_stream_fields
from app.infrastructure.task_repository import RedisTaskRepository
from app.services.remote_fetcher import RemoteVideoFetcher
from app.services.video_decoder import DecodedBatch, DecordVideoDecoder
from app.workers.common import StreamWorker, TerminalWorkerError, run_worker

logger = logging.getLogger(__name__)

_STAGE_ORDER = {
    TaskStage.QUEUED: 0,
    TaskStage.DOWNLOADING: 1,
    TaskStage.DECODING: 2,
    TaskStage.CUT_DETECTION: 3,
    TaskStage.TRACKING: 4,
    TaskStage.AGGREGATING: 5,
    TaskStage.COMPLETED: 6,
    TaskStage.FAILED: 6,
}


class DownstreamStallError(RuntimeError):
    """No downstream chunk finished while ingest was backpressured."""


class IngestHandler:
    def __init__(self, settings: Settings, streams: RedisStreams) -> None:
        self.settings = settings
        self.streams = streams
        self.repository = RedisTaskRepository(streams.client, settings)
        self.object_store = RayObjectStore(settings)
        self.fetcher = RemoteVideoFetcher(settings)
        self._active_tasks: set[UUID] = set()
        self._active_tasks_lock = asyncio.Lock()

    async def __call__(self, message: StreamMessage) -> None:
        task_id = UUID(message.fields["task_id"])
        if not hasattr(self, "_active_tasks_lock"):
            # Some focused unit tests construct the handler with ``__new__``.
            # Lazy initialization also makes the concurrency bookkeeping robust
            # after deserialization or unusual dependency-injection paths.
            self._active_tasks = set()
            self._active_tasks_lock = asyncio.Lock()
        async with self._active_tasks_lock:
            self._active_tasks.add(task_id)
        try:
            await self._process_task(task_id, message)
        finally:
            async with self._active_tasks_lock:
                self._active_tasks.discard(task_id)

    async def _process_task(
        self,
        task_id: UUID,
        message: StreamMessage,
    ) -> None:
        current = await self.repository.get_status(task_id)
        if current.stage in {TaskStage.COMPLETED, TaskStage.FAILED}:
            return
        if await self.repository.is_ingest_complete(task_id):
            logger.info(
                "Ignoring a completed ingest delivery",
                extra={
                    "task_id": str(task_id),
                    "event_id": message.event_id,
                    "stage": current.stage.value,
                },
            )
            return

        started = time.perf_counter()
        try:
            source = await self.repository.get_source(task_id)
            video_path = await self._resolve_video_path(task_id, source)
            await self._advance_stage_if_needed(
                task_id,
                TaskStage.DECODING,
                progress=5.0,
                message="Decoding RGB video frames",
            )

            decoder = DecordVideoDecoder(
                video_path,
                self.settings.chunk_size_frames,
                self.settings.chunk_overlap_frames,
                self.settings.max_decoded_chunk_bytes,
            )
            metadata_started = time.perf_counter()
            metadata = await asyncio.wait_for(
                asyncio.to_thread(decoder.open),
                timeout=self.settings.decode_batch_timeout_seconds,
            )
            metadata_probe_ms = (
                time.perf_counter() - metadata_started
            ) * 1_000
            await self.repository.save_video_metadata(task_id, metadata)
            total_chunks = decoder.total_chunks
            await self._advance_stage_if_needed(
                task_id,
                TaskStage.CUT_DETECTION,
                total_chunks=total_chunks,
                progress=8.0,
                message=(
                    f"Streaming {total_chunks} memory-bounded RGB chunks to "
                    "cut detection"
                ),
            )

            published_count = await self.repository.get_ingest_published_chunks(
                task_id
            )
            released_count = 0
            # Ingest publishes chunks in deterministic order. Starting at the
            # durable count avoids one Redis existence check per chunk on the
            # normal path and still keeps publish_ingest_chunk_once idempotent.
            for chunk_index in range(min(published_count, total_chunks), total_chunks):
                has_capacity, released_count = (
                    await self._wait_for_downstream_capacity(
                        task_id,
                        published_count,
                        released_count,
                    )
                )
                if not has_capacity:
                    return

                decode_started = time.perf_counter()
                batch = await asyncio.wait_for(
                    asyncio.to_thread(decoder.decode_batch, chunk_index),
                    timeout=self.settings.decode_batch_timeout_seconds,
                )
                decode_ms = (time.perf_counter() - decode_started) * 1_000
                if published_count == 0 and chunk_index == 0:
                    decode_ms += metadata_probe_ms
                object_ref = await self._put_frames_with_retry(task_id, batch)
                chunk_id = f"{task_id}:{chunk_index:08d}"
                payload = FrameBatchMetadata(
                    task_id=task_id,
                    chunk_id=chunk_id,
                    chunk_index=batch.chunk_index,
                    object_ref=object_ref,
                    context_start_frame=batch.context_start_frame,
                    context_end_frame=batch.context_end_frame,
                    valid_start_frame=batch.valid_start_frame,
                    valid_end_frame=batch.valid_end_frame,
                    frame_count=batch.frames_rgb.shape[0],
                    fps=metadata.fps,
                    width=metadata.width,
                    height=metadata.height,
                    source_total_frames=metadata.frame_count,
                    is_last=batch.is_last,
                    decode_ms=decode_ms,
                )
                is_new = False
                try:
                    is_new = await self.repository.publish_ingest_chunk_once(
                        task_id,
                        chunk_id,
                        stream=self.settings.stream_video_chunks,
                        stream_fields=model_to_stream_fields(payload),
                        object_ref=object_ref,
                        chunk_index=batch.chunk_index,
                    )
                finally:
                    if not is_new:
                        await asyncio.to_thread(
                            self.object_store.release,
                            object_ref,
                        )

                if not is_new:
                    continue

                CHUNKS_PUBLISHED.inc()
                published_count += 1
                if self._should_publish_progress(published_count, total_chunks):
                    progress = 8.0 + 22.0 * (
                        published_count / max(total_chunks, 1)
                    )
                    await self.repository.update_status(
                        task_id,
                        progress=progress,
                        message=(
                            f"Decoded chunk {published_count}/{total_chunks}; "
                            "downstream processing is running"
                        ),
                    )

            await self.repository.mark_ingest_complete(task_id, total_chunks)
            await self.repository.update_status(
                task_id,
                progress=30.0,
                message="All chunks decoded and safely queued",
            )
            await self._wait_for_downstream_drain(
                task_id,
                total_chunks,
                released_count,
            )
            VIDEO_DECODE_SECONDS.observe(time.perf_counter() - started)
        except DownstreamStallError as exc:
            TASKS_FAILED.labels("pipeline_stalled").inc()
            await self.repository.fail(
                task_id,
                code="pipeline_stalled",
                detail=str(exc),
            )
            raise TerminalWorkerError(str(exc)) from exc
        except Exception as exc:
            TASKS_FAILED.labels("ingest_failed").inc()
            await self.repository.fail(task_id, code="ingest_failed", detail=str(exc))
            raise TerminalWorkerError(str(exc)) from exc

    def _should_publish_progress(self, completed: int, total: int) -> bool:
        return (
            completed == 1
            or completed >= total
            or completed % self.settings.progress_update_every_chunks == 0
        )

    async def _resolve_video_path(
        self,
        task_id: UUID,
        source: VideoSource,
    ) -> Path:
        if source.kind == SourceKind.URL:
            await self._advance_stage_if_needed(
                task_id,
                TaskStage.DOWNLOADING,
                progress=2.0,
                message="Downloading remote video",
            )
            assert source.url is not None
            return await self.fetcher.fetch(task_id, str(source.url))

        assert source.upload_path is not None
        return Path(source.upload_path)

    async def _advance_stage_if_needed(
        self,
        task_id: UUID,
        target_stage: TaskStage,
        *,
        progress: float,
        message: str,
        total_chunks: int | None = None,
    ) -> None:
        current = await self.repository.get_status(task_id)
        if current.stage in {TaskStage.COMPLETED, TaskStage.FAILED}:
            return
        stage = (
            target_stage
            if _STAGE_ORDER[current.stage] < _STAGE_ORDER[target_stage]
            else current.stage
        )
        await self.repository.update_status(
            task_id,
            stage=stage,
            progress=progress,
            message=message,
            total_chunks=total_chunks,
        )

    async def _put_frames_with_retry(
        self,
        task_id: UUID,
        batch: DecodedBatch,
    ) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self.settings.ray_put_max_attempts + 1):
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(self.object_store.put, batch.frames_rgb),
                    timeout=self.settings.ray_put_timeout_seconds,
                )
            except Exception as exc:
                last_error = exc
                if attempt >= self.settings.ray_put_max_attempts:
                    break
                logger.warning(
                    "Ray rejected a decoded chunk; retrying after backpressure",
                    extra={
                        "task_id": str(task_id),
                        "chunk_index": batch.chunk_index,
                        "attempt": attempt,
                        "max_attempts": self.settings.ray_put_max_attempts,
                        "chunk_bytes": int(batch.frames_rgb.nbytes),
                    },
                    exc_info=True,
                )
                await asyncio.sleep(self.settings.ray_put_retry_delay_seconds)

        assert last_error is not None
        raise RuntimeError(
            "Ray could not store a decoded RGB chunk after "
            f"{self.settings.ray_put_max_attempts} attempts. "
            f"Chunk size: {batch.frames_rgb.nbytes} bytes. "
            f"Last error: {last_error}"
        ) from last_error

    async def _active_task_count(self) -> int:
        async with self._active_tasks_lock:
            return len(self._active_tasks)

    async def _release_completed_chunks(
        self,
        task_id: UUID,
        released_count: int,
        tracking_completed: int,
    ) -> int:
        """Release contiguous completed chunks from the owning Ray client.

        The local mock cut/tracking workers intentionally do not open a second
        Ray Client connection. Ingest created the registry handle, so it also
        performs deterministic release after Redis confirms tracking output.
        """

        for chunk_index in range(released_count, tracking_completed):
            object_ref = await self.repository.get_chunk_object_ref(
                task_id,
                chunk_index,
            )
            await asyncio.wait_for(
                asyncio.to_thread(
                    self.object_store.release,
                    object_ref,
                    suppress_errors=False,
                ),
                timeout=self.settings.ray_actor_call_timeout_seconds,
            )
            CHUNKS_RELEASED.inc()
            released_count = chunk_index + 1
        return released_count

    async def _wait_for_downstream_capacity(
        self,
        task_id: UUID,
        published_count: int,
        released_count: int,
    ) -> tuple[bool, int]:
        stalled_since = time.monotonic()
        last_tracking_completed = -1
        last_global_in_flight = -1

        while True:
            stage, tracking_completed = (
                await self.repository.get_backpressure_state(task_id)
            )
            if tracking_completed > released_count:
                released_count = await self._release_completed_chunks(
                    task_id,
                    released_count,
                    tracking_completed,
                )
                stalled_since = time.monotonic()

            if stage == TaskStage.FAILED:
                return False, released_count
            if stage == TaskStage.COMPLETED:
                return False, released_count

            active_tasks = await self._active_task_count()
            per_task_limit = self.settings.max_inflight_chunks_per_task
            if active_tasks > 1:
                # Reserve one of the two local Object Store slots for another
                # video. A single video still uses the configured pipelining.
                per_task_limit = min(per_task_limit, 1)

            task_in_flight = max(0, published_count - tracking_completed)
            global_in_flight = await asyncio.wait_for(
                asyncio.to_thread(self.object_store.registered_count),
                timeout=self.settings.ray_actor_call_timeout_seconds,
            )
            if (
                task_in_flight < per_task_limit
                and global_in_flight < self.settings.max_inflight_chunks_global
            ):
                return True, released_count

            if (
                tracking_completed > last_tracking_completed
                or (
                    last_global_in_flight >= 0
                    and global_in_flight < last_global_in_flight
                )
            ):
                stalled_since = time.monotonic()
            last_tracking_completed = tracking_completed
            last_global_in_flight = global_in_flight

            stalled_for = time.monotonic() - stalled_since
            if stalled_for >= self.settings.downstream_stall_timeout_seconds:
                raise DownstreamStallError(
                    "Downstream processing made no progress for "
                    f"{stalled_for:.1f}s while waiting for chunk capacity: "
                    f"published={published_count}, "
                    f"tracking_completed={tracking_completed}, "
                    f"released={released_count}, "
                    f"task_in_flight={task_in_flight}, "
                    f"registered_objects={global_in_flight}, "
                    f"stage={stage.value}. Check cut/tracking worker logs."
                )

            await asyncio.sleep(self.settings.ingest_backpressure_poll_seconds)

    async def _wait_for_downstream_drain(
        self,
        task_id: UUID,
        total_chunks: int,
        released_count: int,
    ) -> None:
        """Wait for final tracking results and free the last Ray objects."""

        stalled_since = time.monotonic()
        last_tracking_completed = -1
        while released_count < total_chunks:
            stage, tracking_completed = (
                await self.repository.get_backpressure_state(task_id)
            )
            if tracking_completed > released_count:
                released_count = await self._release_completed_chunks(
                    task_id,
                    released_count,
                    tracking_completed,
                )
                stalled_since = time.monotonic()

            if released_count >= total_chunks:
                return
            if stage == TaskStage.FAILED:
                return

            if tracking_completed > last_tracking_completed:
                stalled_since = time.monotonic()
            last_tracking_completed = tracking_completed

            stalled_for = time.monotonic() - stalled_since
            if stalled_for >= self.settings.downstream_stall_timeout_seconds:
                raise DownstreamStallError(
                    "Downstream processing made no progress for "
                    f"{stalled_for:.1f}s while draining final chunks: "
                    f"tracking_completed={tracking_completed}/{total_chunks}, "
                    f"released={released_count}/{total_chunks}, "
                    f"stage={stage.value}. Check cut/tracking worker logs."
                )
            await asyncio.sleep(self.settings.ingest_backpressure_poll_seconds)



def factory(settings: Settings, streams: RedisStreams) -> StreamWorker:
    return StreamWorker(
        worker_name="ingest",
        stream=settings.stream_ingest_jobs,
        group=settings.group_ingest,
        handler=IngestHandler(settings, streams),
        settings=settings,
        streams=streams,
        max_concurrency=settings.ingest_worker_concurrency,
        partition_by_task=True,
    )


if __name__ == "__main__":
    asyncio.run(run_worker(factory))
