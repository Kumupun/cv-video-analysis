from __future__ import annotations

from app.domain.enums import TaskStage

_ALLOWED_TRANSITIONS: dict[TaskStage, set[TaskStage]] = {
    TaskStage.QUEUED: {TaskStage.DOWNLOADING, TaskStage.DECODING, TaskStage.FAILED},
    TaskStage.DOWNLOADING: {TaskStage.DECODING, TaskStage.FAILED},
    TaskStage.DECODING: {TaskStage.CUT_DETECTION, TaskStage.FAILED},
    TaskStage.CUT_DETECTION: {TaskStage.TRACKING, TaskStage.FAILED},
    TaskStage.TRACKING: {TaskStage.AGGREGATING, TaskStage.FAILED},
    TaskStage.AGGREGATING: {TaskStage.COMPLETED, TaskStage.FAILED},
    TaskStage.COMPLETED: set(),
    TaskStage.FAILED: set(),
}


class InvalidStageTransition(ValueError):
    pass


def ensure_stage_transition(current: TaskStage, target: TaskStage) -> None:
    if current == target:
        return
    if target not in _ALLOWED_TRANSITIONS[current]:
        raise InvalidStageTransition(
            f"Invalid pipeline transition: {current.value} -> {target.value}"
        )
