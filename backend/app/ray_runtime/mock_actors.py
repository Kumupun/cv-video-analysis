from __future__ import annotations

from typing import Any

MOCK_NAMESPACE = "cv-video-analysis"


def get_mock_cut_actor(ray: Any) -> Any:
    """Return a named development actor that validates an RGB tensor.

    The actor deliberately returns no detections. Its purpose is to exercise
    the same Ray Actor call path that a real TransNetV2 actor will use.
    """

    @ray.remote(max_restarts=2, max_task_retries=2)
    class MockCutActor:
        def analyze(self, frames: Any) -> dict[str, int]:
            if getattr(frames, "ndim", None) != 4:
                raise ValueError("Expected a [T,H,W,C] frame tensor")
            if frames.shape[-1] != 3:
                raise ValueError("Expected RGB frames")
            return {"frame_count": int(frames.shape[0])}

    return MockCutActor.options(
        name="mock-cut-actor",
        namespace=MOCK_NAMESPACE,
        get_if_exists=True,
    ).remote()


def get_mock_tracking_actor(ray: Any) -> Any:
    """Return a named development actor that validates a tracking tensor."""

    @ray.remote(max_restarts=2, max_task_retries=2)
    class MockTrackingActor:
        def __init__(self) -> None:
            self._next_chunk_by_task: dict[str, int] = {}

        def analyze(
            self,
            task_id: str,
            chunk_index: int,
            frames: Any,
        ) -> dict[str, int | bool]:
            if getattr(frames, "ndim", None) != 4:
                raise ValueError("Expected a [T,H,W,C] frame tensor")
            if frames.shape[-1] != 3:
                raise ValueError("Expected RGB frames")

            expected = self._next_chunk_by_task.get(task_id, 0)
            if chunk_index < expected:
                return {
                    "frame_count": int(frames.shape[0]),
                    "duplicate": True,
                }
            if chunk_index > expected:
                raise RuntimeError(
                    f"Out-of-order tracking chunk: expected {expected}, "
                    f"received {chunk_index}"
                )
            self._next_chunk_by_task[task_id] = expected + 1
            return {
                "frame_count": int(frames.shape[0]),
                "duplicate": False,
            }

    return MockTrackingActor.options(
        name="mock-tracking-actor",
        namespace=MOCK_NAMESPACE,
        get_if_exists=True,
    ).remote()
