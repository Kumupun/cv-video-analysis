from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.domain.schemas import VideoMetadata


class VideoDecodeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DecodedBatch:
    chunk_index: int
    context_start_frame: int
    context_end_frame: int
    valid_start_frame: int
    valid_end_frame: int
    frames_rgb: np.ndarray
    is_last: bool


class DecordVideoDecoder:
    """Decode only RGB video frames; audio is never opened or processed.

    Every chunk contains a configurable prefix copied from the previous chunk.
    The context range gives cut detection enough temporal history at chunk
    boundaries, while the valid range identifies the non-overlapping interval
    for which that chunk is allowed to emit final events.

    ``max_decoded_chunk_bytes`` bounds the uncompressed RGB tensor size. This
    is important because a small compressed video can still expand into very
    large frame batches and otherwise exhaust or stall the Ray object store.
    """

    def __init__(
        self,
        path: Path,
        chunk_size: int,
        overlap_frames: int = 0,
        max_decoded_chunk_bytes: int | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap_frames < 0:
            raise ValueError("overlap_frames cannot be negative")
        if overlap_frames >= chunk_size:
            raise ValueError("overlap_frames must be smaller than chunk_size")
        if max_decoded_chunk_bytes is not None and max_decoded_chunk_bytes <= 0:
            raise ValueError("max_decoded_chunk_bytes must be positive")
        self._path = path
        self._configured_chunk_size = chunk_size
        self._configured_overlap_frames = overlap_frames
        self._max_decoded_chunk_bytes = max_decoded_chunk_bytes
        self._reader = None
        self._metadata: VideoMetadata | None = None

    def open(self) -> VideoMetadata:
        try:
            from decord import VideoReader, cpu
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("decord is required for video ingestion") from exc
        try:
            reader = VideoReader(str(self._path), ctx=cpu(0))
            frame_count = len(reader)
            if frame_count <= 0:
                raise VideoDecodeError("The file contains no decodable video frames")
            fps = float(reader.get_avg_fps())
            if fps <= 0:
                raise VideoDecodeError("The video FPS metadata is invalid")
            first_frame = reader[0].asnumpy()
            if first_frame.ndim != 3 or first_frame.shape[2] != 3:
                raise VideoDecodeError("Expected RGB frames with three channels")
            height, width = first_frame.shape[:2]
            metadata = VideoMetadata(
                fps=fps,
                frame_count=frame_count,
                width=width,
                height=height,
                duration_seconds=frame_count / fps,
            )
            self._reader = reader
            self._metadata = metadata
            return metadata
        except VideoDecodeError:
            raise
        except Exception as exc:
            raise VideoDecodeError(f"Could not decode video: {exc}") from exc

    @property
    def metadata(self) -> VideoMetadata:
        if self._metadata is None:
            return self.open()
        return self._metadata

    @property
    def effective_chunk_size(self) -> int:
        chunk_size, _ = self._effective_chunking()
        return chunk_size

    @property
    def effective_overlap_frames(self) -> int:
        _, overlap_frames = self._effective_chunking()
        return overlap_frames

    def _effective_chunking(self) -> tuple[int, int]:
        metadata = self.metadata
        if self._max_decoded_chunk_bytes is None:
            return self._configured_chunk_size, self._configured_overlap_frames

        bytes_per_rgb_frame = metadata.width * metadata.height * 3
        if bytes_per_rgb_frame > self._max_decoded_chunk_bytes:
            raise VideoDecodeError(
                "A single decoded RGB frame exceeds the configured chunk byte limit"
            )
        max_context_frames = max(
            1,
            self._max_decoded_chunk_bytes // max(bytes_per_rgb_frame, 1),
        )
        if max_context_frames == 1:
            return 1, 0

        # Reserve at most one quarter of the tensor for context. This keeps
        # overlap useful while ensuring valid frames still make progress.
        overlap_frames = min(
            self._configured_overlap_frames,
            max_context_frames // 4,
        )
        chunk_size = min(
            self._configured_chunk_size,
            max_context_frames - overlap_frames,
        )
        chunk_size = max(1, chunk_size)
        overlap_frames = min(overlap_frames, max(0, chunk_size - 1))
        return chunk_size, overlap_frames

    @property
    def total_chunks(self) -> int:
        metadata = self.metadata
        chunk_size = self.effective_chunk_size
        return (metadata.frame_count + chunk_size - 1) // chunk_size

    def decode_batch(self, chunk_index: int) -> DecodedBatch:
        """Decode one deterministic chunk without reading earlier chunks.

        Ingest checks downstream capacity before calling this method. That keeps
        the next large RGB tensor out of process memory while the previous Ray
        object is still in flight, and makes an ingest retry skip already
        published chunks without decoding them again.
        """

        if chunk_index < 0:
            raise IndexError("chunk_index cannot be negative")
        if self._reader is None:
            self.open()
        reader = self._reader
        metadata = self.metadata
        assert reader is not None

        chunk_size, overlap_frames = self._effective_chunking()
        valid_start = chunk_index * chunk_size
        if valid_start >= metadata.frame_count:
            raise IndexError("chunk_index is outside the video")

        valid_end_exclusive = min(
            valid_start + chunk_size,
            metadata.frame_count,
        )
        context_start = max(0, valid_start - overlap_frames)
        indices = list(range(context_start, valid_end_exclusive))
        frames = reader.get_batch(indices).asnumpy()
        frames = np.ascontiguousarray(frames, dtype=np.uint8)
        return DecodedBatch(
            chunk_index=chunk_index,
            context_start_frame=context_start,
            context_end_frame=valid_end_exclusive - 1,
            valid_start_frame=valid_start,
            valid_end_frame=valid_end_exclusive - 1,
            frames_rgb=frames,
            is_last=valid_end_exclusive == metadata.frame_count,
        )

    def batches(self) -> Iterator[DecodedBatch]:
        for chunk_index in range(self.total_chunks):
            yield self.decode_batch(chunk_index)
