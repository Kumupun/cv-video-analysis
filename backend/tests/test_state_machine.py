from itertools import pairwise

import pytest
from app.domain.enums import TaskStage
from app.domain.state_machine import InvalidStageTransition, ensure_stage_transition


def test_pipeline_allows_required_order() -> None:
    path = [
        TaskStage.QUEUED,
        TaskStage.DECODING,
        TaskStage.CUT_DETECTION,
        TaskStage.TRACKING,
        TaskStage.AGGREGATING,
        TaskStage.COMPLETED,
    ]
    for current, target in pairwise(path):
        ensure_stage_transition(current, target)


def test_pipeline_rejects_tracking_before_cut_detection() -> None:
    with pytest.raises(InvalidStageTransition):
        ensure_stage_transition(TaskStage.DECODING, TaskStage.TRACKING)


def test_pipeline_rejects_reopening_completed_task() -> None:
    with pytest.raises(InvalidStageTransition):
        ensure_stage_transition(TaskStage.COMPLETED, TaskStage.TRACKING)
