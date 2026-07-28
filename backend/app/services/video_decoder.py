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
    """

    def __init__(self, path: Path, chunk_size: int, overlap_frames: int = 0) -> None:
        if overlap_frames >= chunk_size:
            raise ValueError("overlap_frames must be smaller than chunk_size")
        self._path = path
        self._chunk_size = chunk_size
        self._overlap_frames = overlap_frames
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

    def batches(self) -> Iterator[DecodedBatch]:
        if self._reader is None:
            self.open()
        reader = self._reader
        metadata = self.metadata
        assert reader is not None

        for chunk_index, valid_start in enumerate(
            range(0, metadata.frame_count, self._chunk_size)
        ):
            valid_end_exclusive = min(
                valid_start + self._chunk_size,
                metadata.frame_count,
            )
            context_start = max(0, valid_start - self._overlap_frames)
            indices = list(range(context_start, valid_end_exclusive))
            frames = reader.get_batch(indices).asnumpy()
            frames = np.ascontiguousarray(frames, dtype=np.uint8)
            yield DecodedBatch(
                chunk_index=chunk_index,
                context_start_frame=context_start,
                context_end_frame=valid_end_exclusive - 1,
                valid_start_frame=valid_start,
                valid_end_frame=valid_end_exclusive - 1,
                frames_rgb=frames,
                is_last=valid_end_exclusive == metadata.frame_count,
            )
