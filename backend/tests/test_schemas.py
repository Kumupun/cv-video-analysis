from uuid import uuid4

import pytest
from app.domain.enums import TransitionType
from app.domain.schemas import (
    CutResultMessage,
    FrameBatchMetadata,
    TrackingJobMessage,
    Transition,
)
from pydantic import ValidationError


def test_hard_cut_requires_frame_and_timestamp() -> None:
    with pytest.raises(ValidationError):
        Transition(type=TransitionType.HARD_CUT, confidence=0.9)


def test_gradual_transition_requires_interval() -> None:
    with pytest.raises(ValidationError):
        Transition(
            type=TransitionType.GRADUAL_TRANSITION,
            confidence=0.9,
            start_frame=10,
        )


def test_tracking_contract_forbids_unverified_cut_stage() -> None:
    with pytest.raises(ValidationError):
        TrackingJobMessage(
            task_id=uuid4(),
            chunk_id="chunk-1",
            chunk_index=0,
            object_ref="token",
            fps=30.0,
            context_start_frame=0,
            context_end_frame=63,
            valid_start_frame=0,
            valid_end_frame=63,
            scenes=[],
            transitions=[],
            cut_verified=False,
        )


def test_frame_batch_valid_range_must_be_inside_context() -> None:
    with pytest.raises(ValidationError):
        FrameBatchMetadata(
            task_id=uuid4(),
            chunk_id="chunk-1",
            chunk_index=0,
            object_ref="token",
            context_start_frame=10,
            context_end_frame=73,
            valid_start_frame=0,
            valid_end_frame=63,
            frame_count=64,
            fps=30.0,
            width=1920,
            height=1080,
            source_total_frames=300,
        )


def test_cut_event_must_be_owned_by_valid_range() -> None:
    with pytest.raises(ValidationError):
        CutResultMessage(
            task_id=uuid4(),
            chunk_id="chunk-1",
            chunk_index=0,
            object_ref="token",
            fps=30.0,
            context_start_frame=48,
            context_end_frame=127,
            valid_start_frame=64,
            valid_end_frame=127,
            transitions=[
                Transition(
                    type=TransitionType.HARD_CUT,
                    frame=60,
                    timestamp=2.0,
                    confidence=0.9,
                )
            ],
            scenes=[],
        )
