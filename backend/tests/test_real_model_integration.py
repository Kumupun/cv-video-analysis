from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from app.core.config import Settings
from app.domain.schemas import FrameBatchMetadata
from participant_3_tracking.worker import _actor_name
from participant_4_cut_detection.worker import CutSceneState, build_cut_result


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
    monkeypatch.setenv("TRACKING_MODEL_CLASSES", '["person", "car"]')

    settings = Settings(_env_file=None)

    assert settings.tracking_model_classes == ("person", "car")
    assert settings.cut_model_num_gpus == 0.5
    assert settings.tracking_model_num_gpus == 0.5


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
