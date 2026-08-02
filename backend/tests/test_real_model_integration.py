from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from uuid import uuid4

import pytest

from app.core.config import Settings
from app.domain.schemas import SUPPORTED_TRACKING_CLASSES, FrameBatchMetadata
from participant_3_tracking.worker import _actor_name
from participant_3_tracking.yolo_world_tracker import YoloWorldTracker
from participant_4_cut_detection.autoshot_detector import AutoShotDetector
from participant_4_cut_detection.worker import CutSceneState, build_cut_result

TRACKING_CLASSES = SUPPORTED_TRACKING_CLASSES


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


def _install_fake_ultralytics(
    monkeypatch: pytest.MonkeyPatch,
    names: dict[int, str],
    architecture_name: str = "WorldModel",
) -> type:
    class FakeBYTETracker:
        instances: list[FakeBYTETracker] = []

        def __init__(self, args) -> None:
            self.args = args
            self.update_calls: list[object] = []
            self.__class__.instances.append(self)

        def update(self, detections, _frame):
            import numpy as np

            self.update_calls.append(detections)
            rows = []
            for index, detection in enumerate(detections.data):
                if float(detection[4]) <= self.args.track_low_thresh:
                    continue
                rows.append(
                    [
                        *detection[:4],
                        index + 10,
                        detection[4],
                        detection[5],
                        index,
                    ]
                )
            return np.asarray(rows, dtype=np.float32)

    class FakeYOLO:
        instances: list[FakeYOLO] = []

        def __init__(self, model_id: str) -> None:
            self.model_id = model_id
            self.model = type(architecture_name, (), {})()
            self.names = names
            self.predictor = None
            self.device = None
            self.set_class_calls: list[tuple[str, ...]] = []
            self.predict_calls: list[dict[str, object]] = []
            self.predict_results: list[object] = []
            self.__class__.instances.append(self)

        def to(self, device: str) -> None:
            self.device = device

        def set_classes(self, classes: list[str]) -> None:
            selected = tuple(classes)
            self.set_class_calls.append(selected)
            self.names = dict(enumerate(selected))

        def predict(self, **kwargs):
            self.predict_calls.append(kwargs)
            return self.predict_results

    ultralytics = ModuleType("ultralytics")
    ultralytics.YOLO = FakeYOLO
    trackers = ModuleType("ultralytics.trackers")
    byte_tracker = ModuleType("ultralytics.trackers.byte_tracker")
    byte_tracker.BYTETracker = FakeBYTETracker
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)
    monkeypatch.setitem(sys.modules, "ultralytics.trackers", trackers)
    monkeypatch.setitem(sys.modules, "ultralytics.trackers.byte_tracker", byte_tracker)
    monkeypatch.setitem(sys.modules, "torch", torch)
    FakeYOLO.fake_byte_tracker = FakeBYTETracker
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


def test_autoshot_decodes_many_hot_range_without_duplicate_hard_cut() -> None:
    np = pytest.importorskip("numpy")
    detector = AutoShotDetector.__new__(AutoShotDetector)
    detector.threshold = 0.55
    detector.gradual_threshold = 0.55
    detector.gradual_boundary_threshold = 0.6
    detector.gradual_min_frames = 3
    detector.gradual_merge_gap_frames = 1
    boundary = np.asarray(
        [0.1, 0.1, 0.6, 0.9, 0.7, 0.2, 0.1, 0.1, 0.2, 0.85, 0.1],
        dtype=np.float32,
    )
    transition = np.asarray(
        [0.1, 0.1, 0.7, 0.8, 0.75, 0.65, 0.1, 0.1, 0.1, 0.8, 0.1],
        dtype=np.float32,
    )
    detector.predict_transition_probabilities = lambda _frames: (
        boundary,
        transition,
    )

    detections = detector.detect_transitions(object())

    assert detections == [
        {
            "type": "gradual_transition",
            "local_start_frame": 2,
            "local_end_frame": 5,
            "confidence": pytest.approx(np.sqrt(0.8 * 0.9)),
        },
        {
            "type": "hard_cut",
            "local_frame": 9,
            "confidence": pytest.approx(0.85),
        },
    ]


def test_autoshot_rejects_unconfirmed_many_hot_gradual_range() -> None:
    np = pytest.importorskip("numpy")
    detector = AutoShotDetector.__new__(AutoShotDetector)
    detector.threshold = 0.55
    detector.gradual_threshold = 0.75
    detector.gradual_boundary_threshold = 0.65
    detector.gradual_min_frames = 6
    detector.gradual_merge_gap_frames = 0
    boundary = np.asarray(
        [0.1, 0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1],
        dtype=np.float32,
    )
    transition = np.asarray(
        [0.1, 0.1, 0.9, 0.88, 0.91, 0.9, 0.87, 0.89, 0.1, 0.1],
        dtype=np.float32,
    )
    detector.predict_transition_probabilities = lambda _frames: (
        boundary,
        transition,
    )

    detections = detector.detect_transitions(object())

    assert detections == []


def test_cut_adapter_maps_gradual_transition_range_and_splits_after_end() -> None:
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
            {
                "type": "gradual_transition",
                "local_start_frame": 12,
                "local_end_frame": 20,
                "confidence": 0.88,
            },
            {
                "type": "hard_cut",
                "local_frame": 18,
                "confidence": 0.99,
            },
        ],
        CutSceneState(next_chunk_index=1, active_scene_number=3),
        processing_ms=12.5,
    )

    assert len(result.transitions) == 1
    transition = result.transitions[0]
    assert transition.type.value == "gradual_transition"
    assert transition.start_frame == 60
    assert transition.end_frame == 68
    assert transition.start_timestamp == pytest.approx(2.4)
    assert transition.end_timestamp == pytest.approx(2.72)
    assert [(scene.start_frame, scene.end_frame) for scene in result.scenes] == [
        (64, 68),
        (69, 127),
    ]
    assert next_state == CutSceneState(
        next_chunk_index=2,
        active_scene_number=4,
    )


def test_cut_adapter_defers_gradual_range_across_chunk_boundary() -> None:
    task_id = uuid4()
    first_batch = _batch(
        task_id=task_id,
        chunk_index=0,
        context_start=0,
        context_end=63,
        valid_start=0,
        valid_end=63,
    )

    first_result, pending_state = build_cut_result(
        first_batch,
        [
            {
                "type": "gradual_transition",
                "local_start_frame": 50,
                "local_end_frame": 63,
                "confidence": 0.81,
            }
        ],
        CutSceneState(next_chunk_index=0, active_scene_number=0),
        processing_ms=1.0,
    )

    assert first_result.transitions == []
    assert [(scene.start_frame, scene.end_frame) for scene in first_result.scenes] == [
        (0, 63)
    ]
    assert pending_state == CutSceneState(
        next_chunk_index=1,
        active_scene_number=0,
        pending_gradual_start_frame=50,
        pending_gradual_confidence=pytest.approx(0.81),
    )

    second_batch = _batch(
        task_id=task_id,
        chunk_index=1,
        context_start=48,
        context_end=127,
        valid_start=64,
        valid_end=127,
    )
    second_result, final_state = build_cut_result(
        second_batch,
        [
            {
                "type": "gradual_transition",
                "local_start_frame": 2,
                "local_end_frame": 32,
                "confidence": 0.9,
            }
        ],
        pending_state,
        processing_ms=1.0,
    )

    assert len(second_result.transitions) == 1
    transition = second_result.transitions[0]
    assert transition.start_frame == 50
    assert transition.end_frame == 80
    assert transition.confidence == pytest.approx(0.9)
    assert [(scene.start_frame, scene.end_frame) for scene in second_result.scenes] == [
        (64, 80),
        (81, 127),
    ]
    assert final_state == CutSceneState(
        next_chunk_index=2,
        active_scene_number=1,
    )


def test_cut_adapter_owns_deferred_gradual_ending_at_previous_boundary() -> None:
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
            {
                "type": "gradual_transition",
                "local_start_frame": 2,
                "local_end_frame": 15,
                "confidence": 0.86,
            }
        ],
        CutSceneState(
            next_chunk_index=1,
            active_scene_number=0,
            pending_gradual_start_frame=50,
            pending_gradual_confidence=0.81,
        ),
        processing_ms=1.0,
    )

    assert result.transitions[0].start_frame == 50
    assert result.transitions[0].end_frame == 63
    assert [
        (scene.scene_id, scene.start_frame, scene.end_frame) for scene in result.scenes
    ] == [(f"{task_id}:scene:1", 64, 127)]
    assert next_state == CutSceneState(
        next_chunk_index=2,
        active_scene_number=1,
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

    assert _actor_name(task_id) == (
        f"yolo-ensemble-tracking-v3-{str(task_id).replace('-', '')}"
    )


def test_model_settings_accept_json_class_list(monkeypatch) -> None:
    monkeypatch.setenv("TRACKING_MODEL_CLASSES", '["person", "car"]')

    settings = Settings(_env_file=None)

    assert settings.tracking_model_classes == ("person", "car")
    assert settings.cut_model_gradual_threshold == 0.75
    assert settings.cut_model_gradual_boundary_threshold == 0.65
    assert settings.cut_model_gradual_min_frames == 6
    assert settings.cut_model_gradual_merge_gap_frames == 0
    assert settings.tracking_ensemble_model_id == (
        "/models/yoloe26l_military_assets.pt"
    )
    assert settings.tracking_ensemble_iou_threshold == 0.55
    assert settings.tracking_parallel_inference is True
    assert settings.tracking_model_image_size == 960
    assert settings.tracking_model_allow_download is False
    assert settings.tracking_model_confidence == 0.50
    assert settings.tracking_track_low_threshold == 0.10
    assert settings.tracking_new_track_threshold == 0.55
    assert settings.tracking_track_buffer == 45
    assert settings.tracking_match_threshold == 0.75
    assert settings.tracking_fuse_score is True
    assert settings.cut_model_num_gpus == 0.5
    assert settings.tracking_model_num_gpus == 0.5


def test_tracking_defaults_use_local_detector_ensemble() -> None:
    settings = Settings(_env_file=None)

    assert settings.tracking_model_id == "/models/best.pt"
    assert settings.tracking_ensemble_model_id == (
        "/models/yoloe26l_military_assets.pt"
    )
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

    tracker = YoloWorldTracker(
        model_id=str(model_path),
        classes=(),
        confidence=0.25,
        image_size=960,
        allow_download=False,
    )

    assert tracker.model_names == dict(enumerate(TRACKING_CLASSES))
    assert fake_yolo.instances[-1].device == "cpu"
    assert fake_yolo.instances[-1].set_class_calls == []


def test_tracking_adapter_rejects_non_yolo_world_primary_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "yoloe26l.pt"
    model_path.write_bytes(b"checkpoint")
    _install_fake_ultralytics(
        monkeypatch,
        dict(enumerate(TRACKING_CLASSES)),
        architecture_name="YOLOEModel",
    )

    with pytest.raises(ValueError, match="must contain a YOLO-World WorldModel"):
        YoloWorldTracker(
            model_id=str(model_path),
            classes=(),
            confidence=0.25,
            image_size=960,
            allow_download=False,
        )


def test_tracking_adapter_sets_coco_and_military_prompts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"checkpoint")
    fake_yolo = _install_fake_ultralytics(
        monkeypatch,
        dict(enumerate(TRACKING_CLASSES)),
    )

    tracker = YoloWorldTracker(
        model_id=str(model_path),
        classes=("camouflage_soldier", "person"),
        confidence=0.25,
        image_size=960,
        allow_download=False,
    )

    assert tracker.model_names == {0: "camouflage_soldier", 1: "person"}
    assert fake_yolo.instances[-1].set_class_calls == [("camouflage_soldier", "person")]
    assert "classes" not in tracker._inference_options()


def test_tracking_ensemble_merges_duplicate_class_boxes_before_bytetrack(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    np = pytest.importorskip("numpy")

    class FakeBoxes:
        def __init__(self, rows) -> None:
            values = np.asarray(rows, dtype=np.float32)
            self.xyxy = values[:, :4]
            self.conf = values[:, 4]
            self.cls = values[:, 5]

        def __len__(self) -> int:
            return len(self.conf)

    primary_path = tmp_path / "best.pt"
    auxiliary_path = tmp_path / "yoloe26l_military_assets.pt"
    primary_path.write_bytes(b"world-checkpoint")
    auxiliary_path.write_bytes(b"yoloe-checkpoint")
    fake_yolo = _install_fake_ultralytics(
        monkeypatch,
        {0: "camouflage_soldier"},
    )
    tracker = YoloWorldTracker(
        model_id=str(primary_path),
        ensemble_model_id=str(auxiliary_path),
        classes=("camouflage_soldier",),
        confidence=0.50,
        image_size=960,
        allow_download=False,
        ensemble_iou_threshold=0.55,
        parallel_inference=True,
    )
    primary, auxiliary = fake_yolo.instances
    primary.predict_results = [
        SimpleNamespace(
            boxes=FakeBoxes([[0, 0, 6, 6, 0.80, 0]]),
            names={0: "camouflage_soldier"},
        )
    ]
    auxiliary.predict_results = [
        SimpleNamespace(
            boxes=FakeBoxes([[0.2, 0.2, 6.2, 6.2, 0.90, 0]]),
            names={0: "camouflage_soldier"},
        )
    ]

    tracks = tracker.process_interval(
        np.zeros((1, 8, 8, 3), dtype=np.uint8),
        global_start_frame=4,
        scene_id="scene-0",
    )

    assert tracker.parallel_inference is True
    assert len(primary.predict_calls) == 1
    assert len(auxiliary.predict_calls) == 1
    assert auxiliary.set_class_calls == []
    assert len(tracks) == 1
    assert tracks[0]["class_name"] == "camouflage_soldier"
    assert tracks[0]["confidence"] == pytest.approx(0.90)
    assert tracks[0]["track_id"] == 0


def test_tracking_adapter_builds_tuned_bytetrack_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"checkpoint")
    _install_fake_ultralytics(monkeypatch, dict(enumerate(TRACKING_CLASSES)))

    tracker = YoloWorldTracker(
        model_id=str(model_path),
        classes=(),
        confidence=0.50,
        image_size=960,
        allow_download=False,
        track_low_threshold=0.10,
        new_track_threshold=0.55,
        track_buffer=45,
        match_threshold=0.75,
        fuse_score=True,
    )

    arguments = tracker.byte_tracker.args
    assert arguments.tracker_type == "bytetrack"
    assert arguments.track_high_thresh == 0.5
    assert arguments.track_low_thresh == 0.1
    assert arguments.new_track_thresh == 0.55
    assert arguments.track_buffer == 45
    assert arguments.match_thresh == 0.75
    assert arguments.fuse_score is True
    assert tracker._inference_options()["conf"] == 0.10


def test_tracking_keeps_low_confidence_candidates_out_of_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    np = pytest.importorskip("numpy")

    class FakeTensor:
        def __init__(self, values) -> None:
            self.values = np.asarray(values)

        def detach(self):
            return self

        def cpu(self):
            return self

        def int(self):
            return FakeTensor(self.values.astype(int))

        def numpy(self):
            return self.values

    class FakeBoxes(SimpleNamespace):
        def __len__(self) -> int:
            return len(self.conf.values)

    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"checkpoint")
    _install_fake_ultralytics(monkeypatch, {0: "person"})
    tracker = YoloWorldTracker(
        model_id=str(model_path),
        classes=(),
        confidence=0.50,
        image_size=960,
        allow_download=False,
        track_low_threshold=0.10,
    )
    boxes = FakeBoxes(
        xyxy=FakeTensor([[0, 0, 2, 2], [1, 1, 5, 5]]),
        conf=FakeTensor([0.20, 0.80]),
        cls=FakeTensor([0, 0]),
    )
    tracker.model.predict_results = [SimpleNamespace(boxes=boxes, names={0: "person"})]

    tracks = tracker.process_interval(
        np.zeros((1, 8, 8, 3), dtype=np.uint8),
        global_start_frame=7,
        scene_id="scene-0",
    )

    assert len(tracks) == 1
    assert tracks[0]["frame"] == 7
    assert tracks[0]["confidence"] == pytest.approx(0.80)
    assert tracks[0]["track_id"] == 0


def test_tracking_resets_after_every_transition_even_if_scene_id_is_reused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    np = pytest.importorskip("numpy")
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"checkpoint")
    _install_fake_ultralytics(monkeypatch, dict(enumerate(TRACKING_CLASSES)))
    tracker = YoloWorldTracker(
        model_id=str(model_path),
        classes=(),
        confidence=0.50,
        image_size=960,
        allow_download=False,
    )
    reset_calls: list[str] = []
    original_reset = tracker._reset_for_scene

    def record_reset(scene_id: str) -> None:
        reset_calls.append(scene_id)
        original_reset(scene_id)

    monkeypatch.setattr(tracker, "_reset_for_scene", record_reset)
    job = SimpleNamespace(
        context_start_frame=0,
        context_end_frame=3,
        valid_start_frame=0,
        valid_end_frame=3,
        chunk_index=0,
        transitions=[SimpleNamespace(frame=1, end_frame=None)],
        scenes=[
            SimpleNamespace(scene_id="scene", start_frame=0, end_frame=1),
            SimpleNamespace(scene_id="scene", start_frame=2, end_frame=3),
        ],
    )

    tracker.process_job(np.zeros((4, 8, 8, 3), dtype=np.uint8), job)

    assert reset_calls == ["scene", "scene"]
    assert len(tracker.model.predict_calls) == 4
    assert all(
        call["conf"] == tracker.track_low_threshold
        for call in tracker.model.predict_calls
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
        YoloWorldTracker(
            model_id=str(tmp_path / "missing.pt"),
            classes=(),
            confidence=0.25,
            image_size=960,
            allow_download=False,
        )

    assert fake_yolo.instances == []


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

    assert settings.cut_model_actor_name == "autoshot-cut-inference-v4"


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
    assert (
        "clip @ https://github.com/openai/CLIP/archive/"
        "ded190a052fdf4585bd685cee5bc96e0310d2c93.zip" in tracking_requirements
    )
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
    assert "models/best.pt filter=lfs diff=lfs merge=lfs -text" in attributes
    assert "models/yolov8l-worldv2.pt filter=lfs diff=lfs merge=lfs -text" in attributes


def test_ml_runtime_uses_one_build_owner_and_shared_uid() -> None:
    project_root = Path(__file__).resolve().parents[2]
    overlay = (project_root / "compose.ml.example.yaml").read_text(encoding="utf-8")
    base_compose = (project_root / "compose.yaml").read_text(encoding="utf-8")
    dockerfile = (
        project_root / "backend" / "docker" / "Dockerfile.ml-worker"
    ).read_text(encoding="utf-8")

    assert overlay.count("  build:") == 1
    assert "ML_MAX_INFLIGHT_CHUNKS_PER_TASK:-2" in overlay
    assert "ML_MAX_INFLIGHT_CHUNKS_GLOBAL:-2" in overlay
    assert "--object-store-memory=${ML_RAY_OBJECT_STORE_BYTES:-268435456}" in overlay
    assert "TRACKING_MODEL_ALLOW_DOWNLOAD:-false" in overlay
    assert "TRACKING_MODEL_ID:-/models/best.pt" in overlay
    assert "TRACKING_ENSEMBLE_MODEL_ID:-/models/yoloe26l_military_assets.pt" in overlay
    assert "TRACKING_PARALLEL_INFERENCE:-true" in overlay
    assert "CUT_MODEL_GRADUAL_THRESHOLD:-0.75" in overlay
    assert "CUT_MODEL_GRADUAL_BOUNDARY_THRESHOLD:-0.65" in overlay
    assert "CUT_MODEL_GRADUAL_MIN_FRAMES:-6" in overlay
    assert "storage-init:" in base_compose
    assert "--uid 10001 worker" in dockerfile
    assert "--uid 10002 worker" not in dockerfile
    assert "clip.load('ViT-B/32', device='cpu', jit=False)" in dockerfile
    assert "hasattr(clip, 'load')" in dockerfile


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
