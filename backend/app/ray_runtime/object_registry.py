from __future__ import annotations

import logging
from typing import Any

OBJECT_REGISTRY_NAME = "frame-object-registry"
OBJECT_REGISTRY_NAMESPACE = "cv-video-analysis"

logger = logging.getLogger(__name__)


def _existing_registry_actor(ray: Any) -> Any | None:
    """Return the detached registry created by another Ray job, if present.

    Every Docker service connects through an independent Ray Client job.  A
    consumer must therefore retrieve the already-created named actor with
    ``ray.get_actor`` instead of redefining its class and calling
    ``get_if_exists`` again.  Re-registering the actor class from every client
    was the blocking call behind the first-chunk stall in local mock runs.
    """

    try:
        return ray.get_actor(
            OBJECT_REGISTRY_NAME,
            namespace=OBJECT_REGISTRY_NAMESPACE,
        )
    except ValueError:
        return None


def get_object_registry_actor(ray: Any) -> Any:
    """Return the detached actor that owns frame ObjectRefs.

    Redis carries only an opaque string token. The actual ObjectRef is passed
    to and from this actor through Ray's native nested-reference protocol. This
    avoids pickling a Ray Client ObjectRef outside Ray, which is not portable
    across independent Ray Client processes.

    The first ingest client creates the actor. All later clients retrieve it by
    name, which is the supported cross-job path for named Ray actors.
    """

    existing = _existing_registry_actor(ray)
    if existing is not None:
        return existing

    @ray.remote(max_restarts=3, max_task_retries=3, num_cpus=0)
    class ObjectRegistryActor:
        def __init__(self) -> None:
            self._entries: dict[str, dict[str, Any]] = {}

        def register(
            self,
            token: str,
            wrapped_ref: list[Any],
            descriptor: dict[str, Any],
            max_objects: int,
        ) -> str:
            if len(wrapped_ref) != 1:
                raise ValueError("Expected exactly one nested ObjectRef")
            if not isinstance(descriptor, dict):
                raise ValueError("Expected a frame descriptor mapping")
            if token in self._entries:
                return token
            if len(self._entries) >= max_objects:
                raise RuntimeError(
                    "Ray object registry capacity reached; retry after a chunk "
                    "is released"
                )
            self._entries[token] = {
                "object_ref": wrapped_ref[0],
                "descriptor": dict(descriptor),
            }
            return token

        def resolve(self, token: str) -> list[Any]:
            entry = self._get_entry(token)
            # Keep the ObjectRef nested so Ray does not automatically
            # dereference it when returning from the actor method.
            return [entry["object_ref"]]

        def describe(self, token: str) -> dict[str, Any]:
            entry = self._get_entry(token)
            return dict(entry["descriptor"])

        def validate_frames(
            self,
            token: str,
            validate_payload: bool = False,
        ) -> dict[str, int]:
            """Run the lightweight mock validation inside this Ray actor.

            The local mock profile needs to verify the Ray actor path but does
            not need another actor per task. Keeping validation in the already
            existing detached registry avoids a second dynamic actor creation
            from Ray Client, which can block indefinitely on Docker Desktop.
            """

            entry = self._get_entry(token)
            descriptor = entry["descriptor"]
            shape = descriptor.get("shape")
            if not isinstance(shape, list) or len(shape) != 4:
                raise ValueError("Expected a [T,H,W,C] frame tensor descriptor")
            if int(shape[-1]) != 3:
                raise ValueError("Expected RGB frames")
            if int(shape[0]) <= 0:
                raise ValueError("Expected at least one frame")

            if validate_payload:
                import ray

                frames = ray.get(entry["object_ref"])
                actual_shape = [int(dimension) for dimension in frames.shape]
                expected_shape = [int(dimension) for dimension in shape]
                if actual_shape != expected_shape:
                    raise ValueError("Frame tensor does not match its descriptor")

            return {"frame_count": int(shape[0])}

        def _get_entry(self, token: str) -> dict[str, Any]:
            try:
                return self._entries[token]
            except KeyError as exc:
                raise KeyError(
                    "Unknown Ray object token. The registry may have restarted "
                    "after memory pressure."
                ) from exc

        def release(self, token: str) -> bool:
            entry = self._entries.pop(token, None)
            if entry is None:
                return False
            object_ref = entry["object_ref"]

            try:
                from ray._private.internal_api import free

                free(object_ref, local_only=False)
            except Exception:
                logger.warning(
                    "Explicit Ray object release failed; reference will be "
                    "GC-managed",
                    exc_info=True,
                )
            return True

        def count(self) -> int:
            return len(self._entries)

    return ObjectRegistryActor.options(
        name=OBJECT_REGISTRY_NAME,
        namespace=OBJECT_REGISTRY_NAMESPACE,
        lifetime="detached",
        get_if_exists=True,
    ).remote()
