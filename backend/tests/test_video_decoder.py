from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.domain.schemas import VideoMetadata
from app.services.video_decoder import FFmpegVideoDecoder, VideoDecodeError


class _FakeProcess:
    def __init__(self, payload: bytes) -> None:
        self.stdout = BytesIO(payload)
        self.stderr = BytesIO()
        self._return_code: int | None = None

    def poll(self) -> int | None:
        return self._return_code

    def wait(self, timeout: float | None = None) -> int:
        self._return_code = 0
        return 0

    def terminate(self) -> None:
        self._return_code = 0

    def kill(self) -> None:
        self._return_code = -9


class _InMemoryFFmpegDecoder(FFmpegVideoDecoder):
    def __init__(
        self,
        frames: np.ndarray,
        *,
        chunk_size: int,
        overlap_frames: int,
        max_decoded_chunk_bytes: int | None = None,
        fps: float = 30.0,
    ) -> None:
        super().__init__(
            Path("unused.mp4"),
            chunk_size=chunk_size,
            overlap_frames=overlap_frames,
            max_decoded_chunk_bytes=max_decoded_chunk_bytes,
        )
        self._frames = np.ascontiguousarray(frames, dtype=np.uint8)
        self._metadata = VideoMetadata(
            fps=fps,
            frame_count=int(frames.shape[0]),
            width=int(frames.shape[2]),
            height=int(frames.shape[1]),
            duration_seconds=float(frames.shape[0] / fps),
            codec="fake",
        )

    def _restart_stream(self, chunk_index: int) -> None:
        self.close()
        chunk_size, overlap_frames = self._effective_chunking()
        valid_start = chunk_index * chunk_size
        context_start = max(0, valid_start - overlap_frames)
        process = _FakeProcess(self._frames[context_start:].tobytes())
        self._process = process  # type: ignore[assignment]
        self._next_chunk_index = chunk_index
        self._previous_tail = None


def _frames(frame_count: int, height: int = 2, width: int = 3) -> np.ndarray:
    values = np.arange(frame_count, dtype=np.uint8)[:, None, None, None]
    return np.broadcast_to(values, (frame_count, height, width, 3)).copy()


def test_decoder_adds_context_without_overlapping_valid_ranges() -> None:
    decoder = _InMemoryFFmpegDecoder(
        _frames(10),
        chunk_size=4,
        overlap_frames=2,
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
    assert batches[1].frames_rgb[:, 0, 0, 0].tolist() == [2, 3, 4, 5, 6, 7]


def test_decoder_limits_uncompressed_rgb_batch_size() -> None:
    bytes_per_frame = 100 * 100 * 3
    decoder = _InMemoryFFmpegDecoder(
        _frames(25, height=100, width=100),
        chunk_size=64,
        overlap_frames=16,
        max_decoded_chunk_bytes=bytes_per_frame * 10,
    )

    batches = list(decoder.batches())

    assert decoder.effective_chunk_size == 8
    assert decoder.effective_overlap_frames == 2
    assert max(batch.frames_rgb.nbytes for batch in batches) <= bytes_per_frame * 10
    assert [batch.valid_start_frame for batch in batches] == [0, 8, 16, 24]


def test_decoder_rejects_frame_larger_than_memory_budget() -> None:
    decoder = _InMemoryFFmpegDecoder(
        _frames(1, height=100, width=100),
        chunk_size=64,
        overlap_frames=16,
        max_decoded_chunk_bytes=1024,
    )

    with pytest.raises(VideoDecodeError, match="single decoded RGB frame"):
        _ = decoder.effective_chunk_size


def test_optimized_budget_reduces_chunks_for_reference_videos() -> None:
    budget = 80 * 1024 * 1024

    high_resolution = _InMemoryFFmpegDecoder(
        np.zeros((1, 1440, 2560, 3), dtype=np.uint8),
        chunk_size=64,
        overlap_frames=16,
        max_decoded_chunk_bytes=budget,
        fps=24.0,
    )
    high_resolution._metadata = VideoMetadata(
        fps=24.0,
        frame_count=511,
        width=2560,
        height=1440,
        duration_seconds=511 / 24,
    )

    hd = _InMemoryFFmpegDecoder(
        np.zeros((1, 720, 1280, 3), dtype=np.uint8),
        chunk_size=64,
        overlap_frames=16,
        max_decoded_chunk_bytes=budget,
    )
    hd._metadata = VideoMetadata(
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
    decoder = _InMemoryFFmpegDecoder(
        _frames(10),
        chunk_size=4,
        overlap_frames=2,
    )

    batch = decoder.decode_batch(2)

    assert batch.chunk_index == 2
    assert batch.context_start_frame == 6
    assert batch.valid_start_frame == 8
    assert batch.valid_end_frame == 9
    assert batch.frames_rgb[:, 0, 0, 0].tolist() == [6, 7, 8, 9]
    assert batch.is_last is True

    with pytest.raises(IndexError, match="outside the video"):
        decoder.decode_batch(3)


def test_probe_uses_counted_frames_when_container_omits_nb_frames(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"not-read-by-test")
    decoder = FFmpegVideoDecoder(video, chunk_size=8)
    calls: list[bool] = []

    def fake_probe(*, count_frames: bool) -> dict[str, Any]:
        calls.append(count_frames)
        stream: dict[str, Any] = {
            "width": 1280,
            "height": 720,
            "avg_frame_rate": "30000/1001",
            "codec_name": "h264",
        }
        if count_frames:
            stream["nb_read_frames"] = "301"
        return {"streams": [stream], "format": {"duration": "10.043"}}

    monkeypatch.setattr(decoder, "_run_ffprobe", fake_probe)

    metadata = decoder.open()

    assert calls == [False, True]
    assert metadata.frame_count == 301
    assert metadata.width == 1280
    assert metadata.height == 720
    assert metadata.codec == "h264"
    assert metadata.fps == pytest.approx(30000 / 1001)
