from __future__ import annotations

import logging
from typing import Any

OBJECT_REGISTRY_NAME = "frame-object-registry"
OBJECT_REGISTRY_NAMESPACE = "cv-video-analysis"

logger = logging.getLogger(__name__)


def get_object_registry_actor(ray: Any) -> Any:
    """Return the named actor that owns frame ObjectRefs.

    Redis carries only an opaque string token. The actual ObjectRef is passed
    to and from this actor through Ray's native nested-reference protocol. This
    avoids pickling a Ray Client ObjectRef outside Ray, which is not portable
    across independent Ray Client processes.
    """

    @ray.remote(max_restarts=0, max_task_retries=2, num_cpus=0)
    class ObjectRegistryActor:
        def __init__(self) -> None:
            self._refs: dict[str, Any] = {}

        def register(self, token: str, wrapped_ref: list[Any]) -> str:
            if len(wrapped_ref) != 1:
                raise ValueError("Expected exactly one nested ObjectRef")
            self._refs.setdefault(token, wrapped_ref[0])
            return token

        def resolve(self, token: str) -> list[Any]:
            try:
                object_ref = self._refs[token]
            except KeyError as exc:
                raise KeyError(f"Unknown Ray object token: {token}") from exc

            # Keep the ObjectRef nested so Ray does not automatically
            # dereference it when returning from the actor method.
            return [object_ref]

        def release(self, token: str) -> bool:
            object_ref = self._refs.pop(token, None)
            if object_ref is None:
                return False

            try:
                from ray._private.internal_api import free

                free(object_ref, local_only=False)
            except Exception:
                logger.warning(
                    "Explicit Ray object release failed; reference will be GC-managed",
                    exc_info=True,
                )
            return True

        def count(self) -> int:
            return len(self._refs)

    return ObjectRegistryActor.options(
        name=OBJECT_REGISTRY_NAME,
        namespace=OBJECT_REGISTRY_NAMESPACE,
        lifetime="detached",
        get_if_exists=True,
    ).remote()
