from __future__ import annotations

from app.ray_runtime.object_registry import (
    OBJECT_REGISTRY_NAME,
    OBJECT_REGISTRY_NAMESPACE,
    get_object_registry_actor,
)


class _ExistingRay:
    def __init__(self) -> None:
        self.actor = object()
        self.remote_called = False

    def get_actor(self, name: str, *, namespace: str):
        assert name == OBJECT_REGISTRY_NAME
        assert namespace == OBJECT_REGISTRY_NAMESPACE
        return self.actor

    def remote(self, **kwargs):
        self.remote_called = True
        raise AssertionError("existing actor must not be redefined")


def test_existing_registry_is_retrieved_without_remote_creation() -> None:
    ray = _ExistingRay()

    result = get_object_registry_actor(ray)

    assert result is ray.actor
    assert ray.remote_called is False
