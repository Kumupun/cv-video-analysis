from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from app.core.config import Settings
from app.infrastructure.ray_store import RayObjectStore


@dataclass
class FakeObjectRef:
    value: Any


class FakeRemoteMethod:
    def __init__(self, callback):
        self._callback = callback

    def remote(self, *args, **kwargs):
        return self._callback(*args, **kwargs)


class FakeRegistry:
    def __init__(self) -> None:
        self.refs: dict[str, FakeObjectRef] = {}
        self.descriptors: dict[str, dict[str, Any]] = {}
        self.register = FakeRemoteMethod(self._register)
        self.resolve = FakeRemoteMethod(self._resolve)
        self.describe = FakeRemoteMethod(self._describe)
        self.validate_frames = FakeRemoteMethod(self._validate_frames)
        self.release = FakeRemoteMethod(self._release)
        self.count = FakeRemoteMethod(self._count)

    def _register(
        self,
        token: str,
        wrapped_ref: list[FakeObjectRef],
        descriptor: dict[str, Any],
        max_objects: int,
    ) -> str:
        if len(self.refs) >= max_objects:
            raise RuntimeError("capacity reached")
        self.refs[token] = wrapped_ref[0]
        self.descriptors[token] = dict(descriptor)
        return token

    def _resolve(self, token: str) -> list[FakeObjectRef]:
        return [self.refs[token]]

    def _describe(self, token: str) -> dict[str, Any]:
        return dict(self.descriptors[token])

    def _validate_frames(
        self,
        token: str,
        validate_payload: bool,
    ) -> dict[str, int]:
        descriptor = self.descriptors[token]
        if validate_payload:
            assert list(self.refs[token].value.shape) == descriptor["shape"]
        return {"frame_count": int(descriptor["shape"][0])}

    def _release(self, token: str) -> bool:
        self.descriptors.pop(token, None)
        return self.refs.pop(token, None) is not None

    def _count(self) -> int:
        return len(self.refs)


class FakeRay:
    def __init__(self) -> None:
        self.timeouts: list[float] = []

    def put(self, value: Any) -> FakeObjectRef:
        return FakeObjectRef(value)

    def get(self, value: Any, *, timeout: float) -> Any:
        self.timeouts.append(timeout)
        if isinstance(value, FakeObjectRef):
            return value.value
        return value


def _store(*, max_objects: int = 4) -> tuple[RayObjectStore, FakeRegistry]:
    settings = Settings(
        _env_file=None,
        max_inflight_chunks_global=max_objects,
        ray_actor_call_timeout_seconds=12,
    )
    store = RayObjectStore(settings)
    fake_registry = FakeRegistry()
    store._ray = FakeRay()
    store._registry = fake_registry
    return store, fake_registry


def test_registry_token_round_trip() -> None:
    store, fake_registry = _store()

    class FrameArray:
        shape = (64, 720, 1280, 3)
        dtype = "uint8"
        nbytes = 64 * 720 * 1280 * 3

    frames = FrameArray()

    token = store.put(frames)

    assert token.startswith("ray-object:")
    assert store.get(token) is frames
    assert store.describe(token) == {
        "shape": [64, 720, 1280, 3],
        "dtype": "uint8",
        "nbytes": frames.nbytes,
    }
    assert store.validate_frames(token) == {"frame_count": 64}
    assert store.validate_frames(token, validate_payload=True) == {"frame_count": 64}
    assert store.registered_count() == 1
    assert token in fake_registry.refs
    assert set(store._ray.timeouts) == {12}

    store.release(token)

    assert store.registered_count() == 0
    assert token not in fake_registry.refs


def test_registry_rejects_more_than_global_limit() -> None:
    store, fake_registry = _store(max_objects=1)
    store.put({"chunk": 1})

    with pytest.raises(RuntimeError, match="capacity reached"):
        store.put({"chunk": 2})

    assert len(fake_registry.refs) == 1
