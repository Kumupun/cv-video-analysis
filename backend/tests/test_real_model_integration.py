from __future__ import annotations

import hashlib
import pickletools
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from uuid import uuid4
from zipfile import ZipFile

import pytest

from app.core.config import Settings
from app.domain.schemas import FrameBatchMetadata
from participant_3_tracking.worker import _actor_name
from participant_3_tracking.yolo_world_tracker import UltralyticsTracker
from participant_4_cut_detection.worker import CutSceneState, build_cut_result

TRACKING_CLASSES = (
    "camouflage_soldier",
    "weapon",
    "military_tank",
    "military_truck",
    "military_vehicle",
    "civilian",
    "soldier",
    "civilian_vehicle",
    "military_artillery",
    "trench",
    "military_aircraft",
    "military_warship",
)


def _batch(
    *,
    task_id,
    chunk_index: int,
    context_start: int,
    context_end: int,
    valid_start: int,
    valid_end: int,
) -> FrameBatchMetadata:
    return FrameBatchMetadata(
        task_id=task_id,
        chunk_id=f"{task_id}:{chunk_index:08d}",
        chunk_index=chunk_index,
        object_ref=f"ray-object:{chunk_index}",
        context_start_frame=context_start,
        context_end_frame=context_end,
        valid_start_frame=valid_start,
        valid_end_frame=valid_end,
        frame_count=context_end - context_start + 1,
        fps=25.0,
        width=640,
        height=360,
        source_total_frames=500,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _install_fake_ultralytics(
    monkeypatch: pytest.MonkeyPatch,
    names: dict[int, str],
) -> type:
    class FakeYOLO:
        instances: list[FakeYOLO] = []

        def __init__(self, model_id: str) -> None:
            self.model_id = model_id
            self.names = names
            self.predictor = None
            self.device = None
            self.__class__.instances.append(self)

        def to(self, device: str) -> None:
            self.device = device

    ultralytics = ModuleType("ultralytics")
    ultralytics.YOLO = FakeYOLO
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return FakeYOLO


def test_tracking_requirements_pin_bytetrack_assignment_solver() -> None:
    requirements_path = (
        Path(__file__).resolve().parents[2]
        / "participant_3_tracking"
        / "requirements.txt"
    )
    requirements = {
        line.strip()
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "lap==0.5.12" in requirements


def test_cut_adapter_maps_local_frames_and_ignores_overlap() -> None:
    task_id = uuid4()
    batch = _batch(
        task_id=task_id,
        chunk_index=1,
        context_start=48,
        context_end=127,
        valid_start=64,
        valid_end=127,
    )

    result, next_state = build_cut_result(
        batch,
        [
            {"local_frame": 2, "confidence": 0.99},
            {"local_frame": 20, "confidence": 0.91},
        ],
        CutSceneState(next_chunk_index=1, active_scene_number=3),
        processing_ms=12.5,
    )

    assert [transition.frame for transition in result.transitions] == [68]
    assert [scene.scene_id for scene in result.scenes] == [
        f"{task_id}:scene:3",
        f"{task_id}:scene:4",
    ]
    assert [(scene.start_frame, scene.end_frame) for scene in result.scenes] == [
        (64, 68),
        (69, 127),
    ]
    assert next_state == CutSceneState(
        next_chunk_index=2,
        active_scene_number=4,
    )


def test_cut_adapter_rejects_out_of_order_chunk() -> None:
    task_id = uuid4()
    batch = _batch(
        task_id=task_id,
        chunk_index=2,
        context_start=112,
        context_end=191,
        valid_start=128,
        valid_end=191,
    )

    try:
        build_cut_result(
            batch,
            [],
            CutSceneState(next_chunk_index=1, active_scene_number=0),
            processing_ms=0.0,
        )
    except RuntimeError as exc:
        assert "expected 1, got 2" in str(exc)
    else:
        raise AssertionError("out-of-order cut chunk was accepted")


def test_tracking_actor_name_is_stable_and_ray_safe() -> None:
    task_id = uuid4()

    assert _actor_name(task_id) == f"yolo-tracking-{str(task_id).replace('-', '')}"


def test_model_settings_accept_json_class_list(monkeypatch) -> None:
    monkeypatch.setenv("TRACKING_MODEL_CLASSES", '["soldier", "weapon"]')

    settings = Settings(_env_file=None)

    assert settings.tracking_model_classes == ("soldier", "weapon")
    assert settings.tracking_model_image_size == 960
    assert settings.tracking_model_allow_download is False
    assert settings.cut_model_num_gpus == 0.5
    assert settings.tracking_model_num_gpus == 0.5


def test_tracking_defaults_use_supplied_offline_checkpoint() -> None:
    settings = Settings(_env_file=None)

    assert settings.tracking_model_id == "/models/yoloe26l_military_assets.pt"
    assert settings.tracking_model_classes == ()
    assert settings.tracking_model_allow_download is False


def test_tracking_adapter_uses_embedded_vocabulary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"checkpoint")
    fake_yolo = _install_fake_ultralytics(
        monkeypatch,
        dict(enumerate(TRACKING_CLASSES)),
    )

    tracker = UltralyticsTracker(
        model_id=str(model_path),
        classes=(),
        confidence=0.25,
        image_size=960,
        allow_download=False,
    )

    assert tracker.class_ids is None
    assert tracker.model_names == dict(enumerate(TRACKING_CLASSES))
    assert fake_yolo.instances[-1].device == "cpu"


def test_tracking_adapter_filters_without_reprompting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"checkpoint")
    _install_fake_ultralytics(
        monkeypatch,
        dict(enumerate(TRACKING_CLASSES)),
    )

    tracker = UltralyticsTracker(
        model_id=str(model_path),
        classes=("soldier", "military_tank"),
        confidence=0.25,
        image_size=960,
        allow_download=False,
    )

    assert tracker.class_ids == (6, 2)
    assert tracker._inference_options()["classes"] == [6, 2]


def test_tracking_adapter_rejects_unknown_checkpoint_class(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"checkpoint")
    _install_fake_ultralytics(
        monkeypatch,
        dict(enumerate(TRACKING_CLASSES)),
    )

    with pytest.raises(ValueError, match="Unknown tracking classes"):
        UltralyticsTracker(
            model_id=str(model_path),
            classes=("person",),
            confidence=0.25,
            image_size=960,
            allow_download=False,
        )


def test_tracking_adapter_fails_before_ultralytics_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_yolo = _install_fake_ultralytics(
        monkeypatch,
        dict(enumerate(TRACKING_CLASSES)),
    )

    with pytest.raises(FileNotFoundError, match="runtime downloads are disabled"):
        UltralyticsTracker(
            model_id=str(tmp_path / "missing.pt"),
            classes=(),
            confidence=0.25,
            image_size=960,
            allow_download=False,
        )

    assert fake_yolo.instances == []


def test_supplied_tracking_checkpoint_metadata_and_hash() -> None:
    project_root = Path(__file__).resolve().parents[2]
    checkpoint_path = project_root / "models" / "yoloe26l_military_assets.pt"

    assert checkpoint_path.stat().st_size == 210_772_932
    assert _sha256(checkpoint_path) == (
        "490452b6652f6add750fc4ed4a70255006b91457555e5ffb69fd9c7e10724d4c"
    )

    with ZipFile(checkpoint_path) as archive:
        pickle_name = next(
            name for name in archive.namelist() if name.endswith("data.pkl")
        )
        pickle_data = archive.read(pickle_name)

    globals_found: set[str] = set()
    strings_found: set[str] = set()
    for opcode, argument, _ in pickletools.genops(pickle_data):
        if opcode.name == "GLOBAL" and isinstance(argument, str):
            globals_found.add(argument)
        if opcode.name in {"UNICODE", "BINUNICODE", "SHORT_BINUNICODE"}:
            if isinstance(argument, str):
                strings_found.add(argument)

    assert "ultralytics.nn.tasks YOLOEModel" in globals_found
    assert "yoloe-26l.yaml" in strings_found
    assert "8.4.112" in strings_found
    assert set(TRACKING_CLASSES).issubset(strings_found)


def test_supplied_cut_checkpoint_matches_bundled_architecture() -> None:
    torch = pytest.importorskip("torch")
    from participant_4_cut_detection.autoshot_architecture import (
        AutoShotSupernet,
    )

    project_root = Path(__file__).resolve().parents[2]
    checkpoint_path = project_root / "models" / "weights_for_cut.pth"
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    state = checkpoint["net"]
    model = AutoShotSupernet()
    assert len(state) == 90
    model.load_state_dict(state, strict=True)


def test_autoshot_resizes_before_temporal_padding() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("cv2")
    import numpy as np

    from participant_4_cut_detection.autoshot_detector import AutoShotDetector

    detector = AutoShotDetector.__new__(AutoShotDetector)
    detector.frame_width = 48
    detector.frame_height = 27
    source = np.zeros((3, 1080, 1920, 3), dtype=np.uint8)

    resized = detector._resize_rgb_frames(source)

    assert resized.shape == (3, 27, 48, 3)
    assert resized.dtype == np.uint8
    assert resized.nbytes == 3 * 27 * 48 * 3


def test_real_model_actor_name_is_versioned() -> None:
    settings = Settings(_env_file=None)

    assert settings.cut_model_actor_name == "autoshot-cut-inference-v2"


def test_tracking_runtime_pins_checkpoint_compatible_dependencies() -> None:
    project_root = Path(__file__).resolve().parents[2]
    tracking_requirements = (
        project_root / "participant_3_tracking" / "requirements.txt"
    ).read_text(encoding="utf-8")
    cut_requirements = (
        project_root / "participant_4_cut_detection" / "requirements.txt"
    ).read_text(encoding="utf-8")
    attributes = (project_root / ".gitattributes").read_text(encoding="utf-8")

    assert "ultralytics==8.4.112" in tracking_requirements
    assert "numpy==1.26.4" in tracking_requirements
    assert "opencv-python==4.10.0.84" in tracking_requirements
    assert "lap==0.5.12" in tracking_requirements
    assert "cloudpickle==3.1.2" in tracking_requirements
    assert cut_requirements.strip() == "opencv-python==4.10.0.84"
    assert "opencv-python-headless" not in tracking_requirements + cut_requirements
    assert (
        "models/yoloe26l_military_assets.pt filter=lfs diff=lfs merge=lfs -text"
        in attributes
    )


def test_ml_runtime_uses_one_build_owner_and_shared_uid() -> None:
    project_root = Path(__file__).resolve().parents[2]
    overlay = (project_root / "compose.ml.example.yaml").read_text(encoding="utf-8")
    base_compose = (project_root / "compose.yaml").read_text(encoding="utf-8")
    dockerfile = (
        project_root / "backend" / "docker" / "Dockerfile.ml-worker"
    ).read_text(encoding="utf-8")

    assert overlay.count("  build:") == 1
    assert "ML_MAX_INFLIGHT_CHUNKS_GLOBAL:-1" in overlay
    assert "--object-store-memory=${ML_RAY_OBJECT_STORE_BYTES:-268435456}" in overlay
    assert "TRACKING_MODEL_ALLOW_DOWNLOAD:-false" in overlay
    assert "/models/yoloe26l_military_assets.pt" in overlay
    assert "storage-init:" in base_compose
    assert "--uid 10001 worker" in dockerfile
    assert "--uid 10002 worker" not in dockerfile


def test_tracking_actor_caches_inference_before_checkpoint_export() -> None:
    project_root = Path(__file__).resolve().parents[2]
    worker_source = (project_root / "participant_3_tracking" / "worker.py").read_text(
        encoding="utf-8"
    )

    cache_position = worker_source.index("self.cache[job.chunk_index] = response")
    checkpoint_position = worker_source.index(
        'response["checkpoint"] = self.tracker.export_state()'
    )

    assert cache_position < checkpoint_position
    assert 'if "checkpoint" not in cached:' in worker_source
