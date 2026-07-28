from pathlib import Path

import numpy as np
from app.domain.schemas import VideoMetadata
from app.services.video_decoder import DecordVideoDecoder


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
