from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
        self.register = FakeRemoteMethod(self._register)
        self.resolve = FakeRemoteMethod(self._resolve)
        self.release = FakeRemoteMethod(self._release)

    def _register(self, token: str, wrapped_ref: list[FakeObjectRef]) -> str:
        self.refs[token] = wrapped_ref[0]
        return token

    def _resolve(self, token: str) -> list[FakeObjectRef]:
        return [self.refs[token]]

    def _release(self, token: str) -> bool:
        return self.refs.pop(token, None) is not None


class FakeRay:
    def put(self, value: Any) -> FakeObjectRef:
        return FakeObjectRef(value)

    def get(self, value: Any) -> Any:
        if isinstance(value, FakeObjectRef):
            return value.value
        return value


def test_registry_token_round_trip() -> None:
    store = RayObjectStore(Settings(_env_file=None))
    fake_ray = FakeRay()
    fake_registry = FakeRegistry()
    store._ray = fake_ray
    store._registry = fake_registry

    frames = {"shape": [64, 720, 1280, 3]}
    token = store.put(frames)

    assert token.startswith("ray-object:")
    assert store.get(token) == frames
    assert token in fake_registry.refs

    store.release(token)

    assert token not in fake_registry.refs
