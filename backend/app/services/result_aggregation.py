from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from app.domain.schemas import (
    CutResultMessage,
    FinalResult,
    PipelineTimings,
    TrackingResultMessage,
    VideoMetadata,
)


def aggregate_result(
    *,
    task_id: UUID,
    video: VideoMetadata,
    chunk_payloads: list[dict[str, str]],
    created_at: datetime | None = None,
    processing_started_at: datetime | None = None,
    completed_at: datetime | None = None,
    aggregation_ms: float = 0.0,
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
    scenes_by_id = {}
    for item in cuts:
        for scene in item.scenes:
            existing_scene = scenes_by_id.get(scene.scene_id)
            if existing_scene is None:
                scenes_by_id[scene.scene_id] = scene
            else:
                scenes_by_id[scene.scene_id] = existing_scene.model_copy(
                    update={
                        "start_frame": min(
                            existing_scene.start_frame,
                            scene.start_frame,
                        ),
                        "end_frame": max(existing_scene.end_frame, scene.end_frame),
                    }
                )
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

    finished = completed_at or datetime.now(UTC)
    created = created_at or finished
    started = processing_started_at or created
    queue_wait_ms = max(0.0, (started - created).total_seconds() * 1_000)
    total_ms = max(0.0, (finished - created).total_seconds() * 1_000)
    decoding_ms = sum(item.decode_ms for item in cuts)
    cut_detection_ms = sum(item.cut_processing_ms for item in cuts)
    tracking_ms = sum(item.tracking_processing_ms for item in tracking)
    measured_ms = (
        queue_wait_ms + decoding_ms + cut_detection_ms + tracking_ms + aggregation_ms
    )
    orchestration_wait_ms = max(0.0, total_ms - measured_ms)

    return FinalResult(
        task_id=task_id,
        video=video,
        scenes=sorted(
            scenes_by_id.values(),
            key=lambda item: (item.start_frame, item.end_frame, item.scene_id),
        ),
        transitions=sorted(
            transitions_by_key.values(),
            key=lambda item: (
                item.frame if item.frame is not None else item.start_frame or 0,
                item.type.value,
            ),
        ),
        tracks=tracks,
        completed_at=finished,
        timings=PipelineTimings(
            queue_wait_ms=queue_wait_ms,
            decoding_ms=decoding_ms,
            cut_detection_ms=cut_detection_ms,
            tracking_ms=tracking_ms,
            aggregation_ms=aggregation_ms,
            orchestration_wait_ms=orchestration_wait_ms,
            total_ms=total_ms,
        ),
        warnings=warnings,
    )
