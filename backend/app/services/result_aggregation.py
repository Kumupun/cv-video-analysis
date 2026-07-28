from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from app.domain.schemas import (
    CutResultMessage,
    FinalResult,
    TrackingResultMessage,
    VideoMetadata,
)


def aggregate_result(
    *,
    task_id: UUID,
    video: VideoMetadata,
    chunk_payloads: list[dict[str, str]],
) -> FinalResult:
    cuts: list[CutResultMessage] = []
    tracking: list[TrackingResultMessage] = []
    warnings: list[str] = []

    for chunk in chunk_payloads:
        cut_payload = chunk.get("cut_payload")
        tracking_payload = chunk.get("tracking_payload")
        if cut_payload:
            cuts.append(CutResultMessage.model_validate_json(cut_payload))
        else:
            warnings.append("A chunk was missing cut-detection output")
        if tracking_payload:
            tracking.append(TrackingResultMessage.model_validate_json(tracking_payload))
        else:
            warnings.append("A chunk was missing tracking output")

    cuts.sort(key=lambda item: item.chunk_index)
    tracking.sort(key=lambda item: item.chunk_index)

    transitions_by_key = {}
    for item in cuts:
        for transition in item.transitions:
            if transition.frame is not None:
                key = (transition.type.value, transition.frame, transition.frame)
            else:
                key = (
                    transition.type.value,
                    transition.start_frame,
                    transition.end_frame,
                )
            existing = transitions_by_key.get(key)
            if existing is None or transition.confidence > existing.confidence:
                transitions_by_key[key] = transition

    tracks = [track for item in tracking for track in item.tracks]
    tracks.sort(key=lambda item: (item.frame, item.scene_id, item.track_id))

    return FinalResult(
        task_id=task_id,
        video=video,
        transitions=list(transitions_by_key.values()),
        tracks=tracks,
        completed_at=datetime.now(UTC),
        warnings=warnings,
    )
