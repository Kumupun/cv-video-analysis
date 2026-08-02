from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import cloudpickle

from app.ray_runtime.mock_actors import (
    get_mock_cut_actor,
    get_mock_tracking_actor,
)


class _WrappedActorClass:
    def __init__(self, actor_class: type[object]) -> None:
        self.actor_class = actor_class

    def options(self, **kwargs: object) -> _WrappedActorClass:
        return self

    def remote(self) -> type[object]:
        return self.actor_class


class _CapturingRay:
    def remote(self, **kwargs: object):
        def decorator(actor_class: type[object]) -> _WrappedActorClass:
            return _WrappedActorClass(actor_class)

        return decorator


def test_mock_actors_deserialize_without_application_package(
    tmp_path: Path,
) -> None:
    fake_ray = _CapturingRay()
    actor_classes = (
        get_mock_cut_actor(fake_ray, "cut-task"),
        get_mock_tracking_actor(fake_ray, "tracking-task"),
    )

    pickle_paths: list[Path] = []
    for index, actor_class in enumerate(actor_classes):
        pickle_path = tmp_path / f"actor-{index}.pkl"
        pickle_path.write_bytes(cloudpickle.dumps(actor_class))
        pickle_paths.append(pickle_path)

    script = """
import cloudpickle
import pathlib
import sys

for raw_path in sys.argv[1:]:
    actor_class = cloudpickle.loads(pathlib.Path(raw_path).read_bytes())
    assert actor_class.__name__ in {"MockCutActor", "MockTrackingActor"}
"""
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    subprocess.run(
        [sys.executable, "-c", script, *(str(path) for path in pickle_paths)],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
