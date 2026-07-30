from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from app.ray_runtime.mock_actors import _inspect_frame_object


@dataclass
class _Ref:
    value: Any


class _Remote:
    def __init__(self, callback):
        self.callback = callback

    def remote(self, *args):
        return _Ref(self.callback(*args))


class _Registry:
    def __init__(self, descriptor: dict[str, Any]) -> None:
        self.describe = _Remote(lambda token: descriptor)
        self.resolve = _Remote(lambda token: [_Ref(_Frames())])


class _Frames:
    shape = (8, 720, 1280, 3)


class _Ray:
    def __init__(self, descriptor: dict[str, Any]) -> None:
        self.registry = _Registry(descriptor)

    def get_actor(self, name: str, *, namespace: str):
        assert name == "frame-object-registry"
        assert namespace == "cv-video-analysis"
        return self.registry

    def get(self, value):
        return value.value if isinstance(value, _Ref) else value


def test_mock_descriptor_path_does_not_materialize_frames() -> None:
    ray = _Ray({"shape": [8, 720, 1280, 3], "nbytes": 1})
    ray.registry.resolve = _Remote(
        lambda token: pytest.fail("payload should not be resolved")
    )

    result = _inspect_frame_object(
        ray,
        "token",
        validate_payload=False,
    )

    assert result == {"frame_count": 8}


def test_full_mock_validation_can_compare_the_real_tensor() -> None:
    ray = _Ray({"shape": [8, 720, 1280, 3], "nbytes": 1})

    result = _inspect_frame_object(
        ray,
        "token",
        validate_payload=True,
    )

    assert result == {"frame_count": 8}
