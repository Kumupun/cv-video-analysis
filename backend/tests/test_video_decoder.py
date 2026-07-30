from pathlib import Path

import numpy as np
import pytest
from app.domain.schemas import VideoMetadata
from app.services.video_decoder import DecordVideoDecoder, VideoDecodeError


class FakeReader:
    def __init__(self, frame_count: int) -> None:
        self.frames = np.zeros((frame_count, 2, 3, 3), dtype=np.uint8)

    def get_batch(self, indices: list[int]):
        frames = self.frames[indices]

        class Batch:
            def asnumpy(self) -> np.ndarray:
                return frames

        return Batch()


def test_decoder_adds_context_without_overlapping_valid_ranges() -> None:
    decoder = DecordVideoDecoder(
        Path("unused.mp4"),
        chunk_size=4,
        overlap_frames=2,
    )
    decoder._reader = FakeReader(10)  # type: ignore[attr-defined]
    decoder._metadata = VideoMetadata(  # type: ignore[attr-defined]
        fps=30.0,
        frame_count=10,
        width=3,
        height=2,
        duration_seconds=10 / 30,
    )

    batches = list(decoder.batches())

    assert [
        (
            item.context_start_frame,
            item.context_end_frame,
            item.valid_start_frame,
            item.valid_end_frame,
        )
        for item in batches
    ] == [
        (0, 3, 0, 3),
        (2, 7, 4, 7),
        (6, 9, 8, 9),
    ]
    assert [item.frames_rgb.shape[0] for item in batches] == [4, 6, 4]


def test_decoder_limits_uncompressed_rgb_batch_size() -> None:
    # 10 RGB frames fit in the configured byte budget. The decoder reserves
    # two frames for context and emits eight new frames per full chunk.
    bytes_per_frame = 100 * 100 * 3
    decoder = DecordVideoDecoder(
        Path("unused.mp4"),
        chunk_size=64,
        overlap_frames=16,
        max_decoded_chunk_bytes=bytes_per_frame * 10,
    )
    decoder._reader = FakeReader(25)  # type: ignore[attr-defined]
    decoder._reader.frames = np.zeros(  # type: ignore[attr-defined]
        (25, 100, 100, 3),
        dtype=np.uint8,
    )
    decoder._metadata = VideoMetadata(  # type: ignore[attr-defined]
        fps=30.0,
        frame_count=25,
        width=100,
        height=100,
        duration_seconds=25 / 30,
    )

    batches = list(decoder.batches())

    assert decoder.effective_chunk_size == 8
    assert decoder.effective_overlap_frames == 2
    assert max(batch.frames_rgb.nbytes for batch in batches) <= bytes_per_frame * 10
    assert [batch.valid_start_frame for batch in batches] == [0, 8, 16, 24]


def test_decoder_rejects_frame_larger_than_memory_budget() -> None:
    decoder = DecordVideoDecoder(
        Path("unused.mp4"),
        chunk_size=64,
        overlap_frames=16,
        max_decoded_chunk_bytes=1024,
    )
    decoder._reader = FakeReader(1)  # type: ignore[attr-defined]
    decoder._reader.frames = np.zeros(  # type: ignore[attr-defined]
        (1, 100, 100, 3),
        dtype=np.uint8,
    )
    decoder._metadata = VideoMetadata(  # type: ignore[attr-defined]
        fps=30.0,
        frame_count=1,
        width=100,
        height=100,
        duration_seconds=1 / 30,
    )

    with pytest.raises(VideoDecodeError, match="single decoded RGB frame"):
        _ = decoder.effective_chunk_size


def test_optimized_budget_reduces_chunks_for_reference_videos() -> None:
    budget = 80 * 1024 * 1024

    high_resolution = DecordVideoDecoder(
        Path("unused-1440p.mp4"),
        chunk_size=64,
        overlap_frames=16,
        max_decoded_chunk_bytes=budget,
    )
    high_resolution._reader = FakeReader(1)  # type: ignore[attr-defined]
    high_resolution._metadata = VideoMetadata(  # type: ignore[attr-defined]
        fps=24.0,
        frame_count=511,
        width=2560,
        height=1440,
        duration_seconds=511 / 24,
    )

    hd = DecordVideoDecoder(
        Path("unused-720p.mp4"),
        chunk_size=64,
        overlap_frames=16,
        max_decoded_chunk_bytes=budget,
    )
    hd._reader = FakeReader(1)  # type: ignore[attr-defined]
    hd._metadata = VideoMetadata(  # type: ignore[attr-defined]
        fps=30.0,
        frame_count=302,
        width=1280,
        height=720,
        duration_seconds=302 / 30,
    )

    assert high_resolution.effective_chunk_size == 6
    assert high_resolution.total_chunks == 86
    assert hd.effective_chunk_size == 23
    assert hd.total_chunks == 14
    assert high_resolution.total_chunks + hd.total_chunks == 100


def test_decode_batch_can_jump_directly_to_a_requested_chunk() -> None:
    decoder = DecordVideoDecoder(
        Path("unused.mp4"),
        chunk_size=4,
        overlap_frames=2,
    )
    decoder._reader = FakeReader(10)  # type: ignore[attr-defined]
    decoder._metadata = VideoMetadata(  # type: ignore[attr-defined]
        fps=30.0,
        frame_count=10,
        width=3,
        height=2,
        duration_seconds=10 / 30,
    )

    batch = decoder.decode_batch(2)

    assert batch.chunk_index == 2
    assert batch.context_start_frame == 6
    assert batch.valid_start_frame == 8
    assert batch.valid_end_frame == 9
    assert batch.is_last is True

    with pytest.raises(IndexError, match="outside the video"):
        decoder.decode_batch(3)
