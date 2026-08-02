from uuid import uuid4

from app.domain.enums import TransitionType
from app.domain.schemas import (
    CutResultMessage,
    SceneInterval,
    TrackingResultMessage,
    Transition,
    VideoMetadata,
)
from app.services.result_aggregation import aggregate_result


def test_aggregation_deduplicates_same_transition() -> None:
    task_id = uuid4()
    first = CutResultMessage(
        task_id=task_id,
        chunk_id="a",
        chunk_index=0,
        object_ref="ref-a",
        fps=30,
        context_start_frame=0,
        context_end_frame=127,
        valid_start_frame=0,
        valid_end_frame=127,
        transitions=[
            Transition(
                type=TransitionType.HARD_CUT,
                frame=100,
                timestamp=100 / 30,
                confidence=0.8,
            )
        ],
        scenes=[SceneInterval(scene_id="scene-0", start_frame=0, end_frame=127)],
    )
    second = first.model_copy(
        update={
            "chunk_id": "b",
            "chunk_index": 1,
            "object_ref": "ref-b",
            "context_end_frame": 255,
            "valid_end_frame": 255,
            "transitions": [
                Transition(
                    type=TransitionType.HARD_CUT,
                    frame=100,
                    timestamp=100 / 30,
                    confidence=0.95,
                )
            ],
            "scenes": [
                SceneInterval(scene_id="scene-0", start_frame=128, end_frame=255)
            ],
        }
    )
    tracking_a = TrackingResultMessage(
        task_id=task_id,
        chunk_id="a",
        chunk_index=0,
        object_ref="ref-a",
        tracks=[],
    )
    tracking_b = TrackingResultMessage(
        task_id=task_id,
        chunk_id="b",
        chunk_index=1,
        object_ref="ref-b",
        tracks=[],
    )
    chunks = [
        {
            "cut_payload": first.model_dump_json(),
            "tracking_payload": tracking_a.model_dump_json(),
        },
        {
            "cut_payload": second.model_dump_json(),
            "tracking_payload": tracking_b.model_dump_json(),
        },
    ]
    result = aggregate_result(
        task_id=task_id,
        video=VideoMetadata(
            fps=30,
            frame_count=300,
            width=1920,
            height=1080,
            duration_seconds=10,
        ),
        chunk_payloads=chunks,
    )
    assert len(result.transitions) == 1
    assert result.transitions[0].confidence == 0.95
    assert result.scenes[0].model_dump() == {
        "scene_id": "scene-0",
        "start_frame": 0,
        "end_frame": 255,
    }
    assert result.pipeline_version == "1.3"
