from __future__ import annotations

from typing import Any

MOCK_NAMESPACE = "cv-video-analysis"


def _actor_name(prefix: str, task_id: str) -> str:
    safe_task_id = task_id.replace(":", "-")
    return f"{prefix}-{safe_task_id}"


def validate_frame_descriptor(descriptor: dict[str, Any]) -> dict[str, int]:
    """Validate small frame metadata without materializing the RGB tensor."""

    shape = descriptor.get("shape")
    if not isinstance(shape, list) or len(shape) != 4:
        raise ValueError("Expected a [T,H,W,C] frame tensor descriptor")
    if int(shape[-1]) != 3:
        raise ValueError("Expected RGB frames")
    if int(shape[0]) <= 0:
        raise ValueError("Expected at least one frame")
    return {"frame_count": int(shape[0])}


def _inspect_frame_object(
    ray: Any,
    object_token: str,
    *,
    validate_payload: bool,
) -> dict[str, int]:
    """Inspect a registered frame tensor from an application process.

    This helper is used outside remote actors. Remote actors receive only the
    already-resolved descriptor, avoiding actor-to-actor registry calls that
    can deadlock under Ray Client.
    """

    registry = ray.get_actor(
        "frame-object-registry",
        namespace=MOCK_NAMESPACE,
    )
    descriptor = ray.get(registry.describe.remote(object_token))
    if not isinstance(descriptor, dict):
        raise RuntimeError("Ray object registry returned invalid metadata")

    inspection = validate_frame_descriptor(descriptor)
    if validate_payload:
        wrapped_ref = ray.get(registry.resolve.remote(object_token))
        if not isinstance(wrapped_ref, list) or len(wrapped_ref) != 1:
            raise RuntimeError("Ray object registry returned an invalid reference")
        frames = ray.get(wrapped_ref[0])
        actual_shape = [int(dimension) for dimension in frames.shape]
        expected_shape = [int(dimension) for dimension in descriptor["shape"]]
        if actual_shape != expected_shape:
            raise ValueError("Frame tensor does not match its descriptor")

    return inspection


def get_mock_cut_actor(ray: Any, task_id: str) -> Any:
    """Return a task-partitioned development cut actor."""

    @ray.remote(max_restarts=2, max_task_retries=2, num_cpus=0)
    class MockCutActor:
        def analyze(self, descriptor: dict[str, Any]) -> dict[str, int]:
            # Keep this method fully self-contained. The stock Ray image does
            # not contain the application's ``app`` package.
            shape = descriptor.get("shape")
            if not isinstance(shape, list) or len(shape) != 4:
                raise ValueError("Expected a [T,H,W,C] frame tensor descriptor")
            if int(shape[-1]) != 3:
                raise ValueError("Expected RGB frames")
            if int(shape[0]) <= 0:
                raise ValueError("Expected at least one frame")
            return {"frame_count": int(shape[0])}

    return MockCutActor.options(
        name=_actor_name("mock-cut-actor", task_id),
        namespace=MOCK_NAMESPACE,
        get_if_exists=True,
    ).remote()


def get_mock_tracking_actor(ray: Any, task_id: str) -> Any:
    """Return one development tracking actor per video task."""

    @ray.remote(max_restarts=5, max_task_retries=3, num_cpus=0)
    class MockTrackingActor:
        def analyze(
            self,
            expected_task_id: str,
            chunk_index: int,
            descriptor: dict[str, Any],
        ) -> dict[str, int | str]:
            # Duplicated intentionally so remote deserialization never imports
            # any application module in the plain Ray head container.
            shape = descriptor.get("shape")
            if not isinstance(shape, list) or len(shape) != 4:
                raise ValueError("Expected a [T,H,W,C] frame tensor descriptor")
            if int(shape[-1]) != 3:
                raise ValueError("Expected RGB frames")
            if int(shape[0]) <= 0:
                raise ValueError("Expected at least one frame")
            return {
                "task_id": expected_task_id,
                "chunk_index": chunk_index,
                "frame_count": int(shape[0]),
            }

    return MockTrackingActor.options(
        name=_actor_name("mock-tracking-actor", task_id),
        namespace=MOCK_NAMESPACE,
        get_if_exists=True,
    ).remote()
