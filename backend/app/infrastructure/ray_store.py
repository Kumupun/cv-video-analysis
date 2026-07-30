from __future__ import annotations

import logging
import threading
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
        self._connect_lock = threading.Lock()
        self._registry_lock = threading.Lock()

    def connect(self) -> None:
        if self._ray is not None:
            return
        with self._connect_lock:
            if self._ray is not None:
                return
            try:
                import ray
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise RuntimeError(
                    "ray package is required for production mode"
                ) from exc
            if not ray.is_initialized():
                ray.init(
                    address=self._settings.ray_address,
                    ignore_reinit_error=True,
                )
            self._ray = ray

    @property
    def ray(self) -> Any:
        if self._ray is None:
            self.connect()
        return self._ray

    def _registry_handle(self) -> Any:
        if self._registry is not None:
            return self._registry
        with self._registry_lock:
            if self._registry is None:
                self._registry = get_object_registry_actor(self.ray)
        return self._registry

    def _get(self, object_ref: Any) -> Any:
        return self.ray.get(
            object_ref,
            timeout=self._settings.ray_actor_call_timeout_seconds,
        )

    def put(self, value: Any) -> str:
        object_ref = self.ray.put(value)
        token = f"ray-object:{uuid4().hex}"

        try:
            # Nesting prevents Ray from replacing the ObjectRef with the frame
            # value when calling the registry actor.
            registration_ref = self._registry_handle().register.remote(
                token,
                [object_ref],
                self._describe_value(value),
                self._settings.max_inflight_chunks_global,
            )
            self._get(registration_ref)
        except Exception:
            self._free_unregistered_ref(object_ref)
            raise
        return token

    def registered_count(self) -> int:
        count_ref = self._registry_handle().count.remote()
        return int(self._get(count_ref))

    def resolve_ref(self, token: str) -> Any:
        wrapped_ref = self._get(self._registry_handle().resolve.remote(token))
        if not isinstance(wrapped_ref, list) or len(wrapped_ref) != 1:
            raise RuntimeError("Ray object registry returned an invalid reference")
        return wrapped_ref[0]

    def describe(self, token: str) -> dict[str, Any]:
        descriptor = self._get(self._registry_handle().describe.remote(token))
        if not isinstance(descriptor, dict):
            raise RuntimeError("Ray object registry returned invalid metadata")
        return dict(descriptor)

    def validate_frames(
        self,
        token: str,
        *,
        validate_payload: bool = False,
    ) -> dict[str, int]:
        result = self._get(
            self._registry_handle().validate_frames.remote(
                token,
                validate_payload,
            )
        )
        if not isinstance(result, dict):
            raise RuntimeError("Ray registry returned invalid validation data")
        return {"frame_count": int(result["frame_count"])}

    def get(self, token: str) -> Any:
        return self._get(self.resolve_ref(token))

    def release(self, token: str, *, suppress_errors: bool = True) -> bool:
        """Release one registered object token.

        The ingest process owns local mock-pipeline object lifetime and calls this
        method with ``suppress_errors=False`` so backpressure cannot silently
        deadlock behind an object that was never freed. Other cleanup paths may
        keep the historical best-effort behavior.
        """

        try:
            release_ref = self._registry_handle().release.remote(token)
            return bool(self._get(release_ref))
        except Exception:
            if not suppress_errors:
                raise
            logger.warning(
                "Explicit Ray object release failed; reference will be GC-managed",
                exc_info=True,
            )
            return False

    def ping(self) -> bool:
        try:
            ref = self.ray.put({"health": "ok"})
            value = self._get(ref)
            self._free_unregistered_ref(ref)
            return value.get("health") == "ok"
        except Exception:
            return False

    @staticmethod
    def _describe_value(value: Any) -> dict[str, Any]:
        shape = getattr(value, "shape", ())
        try:
            normalized_shape = [int(dimension) for dimension in shape]
        except (TypeError, ValueError):
            normalized_shape = []
        return {
            "shape": normalized_shape,
            "dtype": str(getattr(value, "dtype", "unknown")),
            "nbytes": int(getattr(value, "nbytes", 0) or 0),
        }

    @staticmethod
    def _free_unregistered_ref(object_ref: Any) -> None:
        try:
            from ray._private.internal_api import free

            free(object_ref, local_only=False)
        except Exception:
            logger.debug(
                "Ray object will be reference-counted after registration failure",
                exc_info=True,
            )
