from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from app.core.config import Settings
from app.ray_runtime.object_registry import get_object_registry_actor

logger = logging.getLogger(__name__)


class RayObjectStore:
    """Adapter around Ray Object Store with an opaque Redis-safe token.

    A named Ray actor owns the real ObjectRefs. Redis receives only a UUID-like
    token, while ObjectRefs move between Ray clients through Ray's supported
    nested-reference protocol. This works across independent Docker workers and
    keeps the frame tensor pinned until the aggregator releases the token.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._ray: Any | None = None
        self._registry: Any | None = None

    def connect(self) -> None:
        if self._ray is not None:
            return
        try:
            import ray
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("ray package is required for production mode") from exc
        if not ray.is_initialized():
            ray.init(address=self._settings.ray_address, ignore_reinit_error=True)
        self._ray = ray

    @property
    def ray(self) -> Any:
        if self._ray is None:
            self.connect()
        return self._ray

    def _registry_handle(self) -> Any:
        if self._registry is None:
            self._registry = get_object_registry_actor(self.ray)
        return self._registry

    def put(self, value: Any) -> str:
        object_ref = self.ray.put(value)
        token = f"ray-object:{uuid4().hex}"

        # Nesting prevents Ray from replacing the ObjectRef with the frame
        # value when calling the registry actor.
        registration_ref = self._registry_handle().register.remote(
            token,
            [object_ref],
        )
        self.ray.get(registration_ref)
        return token

    def resolve_ref(self, token: str) -> Any:
        wrapped_ref = self.ray.get(self._registry_handle().resolve.remote(token))
        if not isinstance(wrapped_ref, list) or len(wrapped_ref) != 1:
            raise RuntimeError("Ray object registry returned an invalid reference")
        return wrapped_ref[0]

    def get(self, token: str) -> Any:
        return self.ray.get(self.resolve_ref(token))

    def release(self, token: str) -> None:
        try:
            release_ref = self._registry_handle().release.remote(token)
            self.ray.get(release_ref)
        except Exception:
            logger.warning(
                "Explicit Ray object release failed; reference will be GC-managed",
                exc_info=True,
            )

    def ping(self) -> bool:
        try:
            ref = self.ray.put({"health": "ok"})
            value = self.ray.get(ref)
            try:
                from ray._private.internal_api import free

                free(ref, local_only=False)
            except Exception:
                logger.debug("Ray health object will be GC-managed", exc_info=True)
            return value.get("health") == "ok"
        except Exception:
            return False
