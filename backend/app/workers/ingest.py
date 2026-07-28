from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from uuid import UUID

from app.core.config import Settings
from app.core.metrics import CHUNKS_PUBLISHED, TASKS_FAILED, VIDEO_DECODE_SECONDS
from app.domain.enums import SourceKind, TaskStage
from app.domain.schemas import FrameBatchMetadata
from app.infrastructure.ray_store import RayObjectStore
from app.infrastructure.redis_streams import RedisStreams, StreamMessage
from app.infrastructure.serialization import model_to_stream_fields
from app.infrastructure.task_repository import RedisTaskRepository
from app.services.remote_fetcher import RemoteVideoFetcher
from app.services.video_decoder import DecordVideoDecoder
from app.workers.common import StreamWorker, TerminalWorkerError, run_worker

logger = logging.getLogger(__name__)


class IngestHandler:
    def __init__(self, settings: Settings, streams: RedisStreams) -> None:
        self.settings = settings
        self.streams = streams
        self.repository = RedisTaskRepository(streams.client, settings)
        self.object_store = RayObjectStore(settings)
        self.fetcher = RemoteVideoFetcher(settings)

    async def __call__(self, message: StreamMessage) -> None:
        task_id = UUID(message.fields["task_id"])
        source = await self.repository.get_source(task_id)
        started = time.perf_counter()
        try:
            if source.kind == SourceKind.URL:
                await self.repository.update_status(
                    task_id,
                    stage=TaskStage.DOWNLOADING,
                    progress=2.0,
                    message="Downloading remote video",
                )
                assert source.url is not None
                video_path = await self.fetcher.fetch(task_id, str(source.url))
            else:
                assert source.upload_path is not None
                video_path = Path(source.upload_path)

            await self.repository.update_status(
                task_id,
                stage=TaskStage.DECODING,
                progress=5.0,
                message="Decoding RGB video frames",
            )
            decoder = DecordVideoDecoder(
                video_path,
                self.settings.chunk_size_frames,
                self.settings.chunk_overlap_frames,
            )
            metadata = await asyncio.to_thread(decoder.open)
            await self.repository.save_video_metadata(task_id, metadata)
            total_chunks = (
                metadata.frame_count + self.settings.chunk_size_frames - 1
            ) // self.settings.chunk_size_frames
            await self.repository.update_status(
                task_id,
                stage=TaskStage.CUT_DETECTION,
                total_chunks=total_chunks,
                progress=8.0,
                message=(
                    f"Streaming {total_chunks} RGB chunks to cut detection "
                    "while decoding continues"
                ),
            )

            batches = decoder.batches()
            for batch in batches:
                object_ref = await asyncio.to_thread(
                    self.object_store.put, batch.frames_rgb
                )
                chunk_id = f"{task_id}:{batch.chunk_index:08d}"
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
                )
                try:
                    await self.streams.publish(
                        self.settings.stream_video_chunks,
                        model_to_stream_fields(payload),
                    )
                except Exception:
                    await asyncio.to_thread(self.object_store.release, object_ref)
                    raise
                CHUNKS_PUBLISHED.inc()
                progress = 8.0 + 22.0 * ((batch.chunk_index + 1) / max(total_chunks, 1))
                await self.repository.update_status(
                    task_id,
                    progress=progress,
                    message=(
                        f"Decoded chunk {batch.chunk_index + 1}/{total_chunks}; "
                        "waiting for cut detection"
                    ),
                )

            current = await self.repository.get_status(task_id)
            await self.repository.update_status(
                task_id,
                stage=current.stage,
                progress=max(current.progress, 30.0),
                message="All chunks decoded; downstream processing is running",
            )
            VIDEO_DECODE_SECONDS.observe(time.perf_counter() - started)
        except Exception as exc:
            TASKS_FAILED.labels("ingest_failed").inc()
            await self.repository.fail(task_id, code="ingest_failed", detail=str(exc))
            raise TerminalWorkerError(str(exc)) from exc


def factory(settings: Settings, streams: RedisStreams) -> StreamWorker:
    return StreamWorker(
        worker_name="ingest",
        stream=settings.stream_ingest_jobs,
        group=settings.group_ingest,
        handler=IngestHandler(settings, streams),
        settings=settings,
        streams=streams,
    )


if __name__ == "__main__":
    asyncio.run(run_worker(factory))
