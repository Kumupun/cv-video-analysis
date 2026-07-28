from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from app.domain.enums import SourceKind, TaskStage, TransitionType


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VideoSource(StrictModel):
    kind: SourceKind
    upload_path: str | None = None
    url: HttpUrl | None = None
    original_filename: str | None = None
    content_type: str | None = None

    @model_validator(mode="after")
    def validate_source(self) -> VideoSource:
        if self.kind == SourceKind.UPLOAD and not self.upload_path:
            raise ValueError("upload_path is required for an uploaded video")
        if self.kind == SourceKind.URL and self.url is None:
            raise ValueError("url is required for a remote video")
        return self


class CreateTaskResponse(StrictModel):
    task_id: UUID
    status_url: str
    results_url: str


class TaskStatus(StrictModel):
    task_id: UUID
    stage: TaskStage
    progress: float = Field(ge=0.0, le=100.0)
    message: str
    created_at: datetime
    updated_at: datetime
    total_chunks: int = Field(default=0, ge=0)
    cut_completed_chunks: int = Field(default=0, ge=0)
    tracking_completed_chunks: int = Field(default=0, ge=0)
    error_code: str | None = None
    error_detail: str | None = None


class FrameBatchMetadata(StrictModel):
    task_id: UUID
    chunk_id: str
    chunk_index: int = Field(ge=0)
    object_ref: str
    context_start_frame: int = Field(ge=0)
    context_end_frame: int = Field(ge=0)
    valid_start_frame: int = Field(ge=0)
    valid_end_frame: int = Field(ge=0)
    frame_count: int = Field(gt=0)
    fps: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    source_total_frames: int = Field(gt=0)
    is_last: bool = False
    emitted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def validate_frame_window(self) -> FrameBatchMetadata:
        if self.context_end_frame < self.context_start_frame:
            raise ValueError("context frame range is invalid")
        expected = self.context_end_frame - self.context_start_frame + 1
        if expected != self.frame_count:
            raise ValueError("frame_count must match the inclusive context range")
        if not (
            self.context_start_frame
            <= self.valid_start_frame
            <= self.valid_end_frame
            <= self.context_end_frame
        ):
            raise ValueError("valid frame range must be inside the context range")
        if self.context_end_frame >= self.source_total_frames:
            raise ValueError("context range exceeds the source video")
        return self


class Transition(StrictModel):
    type: TransitionType
    confidence: float = Field(ge=0.0, le=1.0)
    frame: int | None = Field(default=None, ge=0)
    timestamp: float | None = Field(default=None, ge=0.0)
    start_frame: int | None = Field(default=None, ge=0)
    end_frame: int | None = Field(default=None, ge=0)
    start_timestamp: float | None = Field(default=None, ge=0.0)
    end_timestamp: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_transition_shape(self) -> Transition:
        if self.type == TransitionType.HARD_CUT:
            if self.frame is None or self.timestamp is None:
                raise ValueError("hard_cut requires frame and timestamp")
        else:
            required = (
                self.start_frame,
                self.end_frame,
                self.start_timestamp,
                self.end_timestamp,
            )
            if any(value is None for value in required):
                raise ValueError(
                    "gradual_transition requires start/end frames and timestamps"
                )
            if self.end_frame < self.start_frame:  # type: ignore[operator]
                raise ValueError("gradual transition end_frame precedes start_frame")
        return self


class SceneInterval(StrictModel):
    scene_id: str
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_interval(self) -> SceneInterval:
        if self.end_frame < self.start_frame:
            raise ValueError("scene end_frame precedes start_frame")
        return self


class CutResultMessage(StrictModel):
    task_id: UUID
    chunk_id: str
    chunk_index: int = Field(ge=0)
    object_ref: str
    fps: float = Field(gt=0)
    context_start_frame: int = Field(ge=0)
    context_end_frame: int = Field(ge=0)
    valid_start_frame: int = Field(ge=0)
    valid_end_frame: int = Field(ge=0)
    transitions: list[Transition]
    scenes: list[SceneInterval]
    processed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def validate_owned_output_range(self) -> CutResultMessage:
        if not (
            self.context_start_frame
            <= self.valid_start_frame
            <= self.valid_end_frame
            <= self.context_end_frame
        ):
            raise ValueError("valid range must be inside the context range")
        for transition in self.transitions:
            anchor = (
                transition.frame
                if transition.type == TransitionType.HARD_CUT
                else transition.end_frame
            )
            if anchor is None or not (
                self.valid_start_frame <= anchor <= self.valid_end_frame
            ):
                raise ValueError(
                    "each transition must be owned by this chunk's valid range"
                )
        for scene in self.scenes:
            if not (
                self.valid_start_frame
                <= scene.start_frame
                <= scene.end_frame
                <= self.valid_end_frame
            ):
                raise ValueError("scene intervals must be clipped to the valid range")
        return self


class TrackingJobMessage(StrictModel):
    task_id: UUID
    chunk_id: str
    chunk_index: int = Field(ge=0)
    object_ref: str
    fps: float = Field(gt=0)
    context_start_frame: int = Field(ge=0)
    context_end_frame: int = Field(ge=0)
    valid_start_frame: int = Field(ge=0)
    valid_end_frame: int = Field(ge=0)
    scenes: list[SceneInterval]
    transitions: list[Transition]
    cut_verified: bool = True

    @model_validator(mode="after")
    def prevent_pipeline_skip(self) -> TrackingJobMessage:
        if not self.cut_verified:
            raise ValueError("tracking is forbidden before cut detection is verified")
        if not (
            self.context_start_frame
            <= self.valid_start_frame
            <= self.valid_end_frame
            <= self.context_end_frame
        ):
            raise ValueError("valid range must be inside the context range")
        return self


class BoundingBox(StrictModel):
    x: float = Field(ge=0.0)
    y: float = Field(ge=0.0)
    width: float = Field(gt=0.0)
    height: float = Field(gt=0.0)


class TrackedObject(StrictModel):
    frame: int = Field(ge=0)
    scene_id: str
    track_id: int = Field(ge=0)
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox


class TrackingResultMessage(StrictModel):
    task_id: UUID
    chunk_id: str
    chunk_index: int = Field(ge=0)
    object_ref: str
    tracks: list[TrackedObject]
    processed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class VideoMetadata(StrictModel):
    fps: float = Field(gt=0)
    frame_count: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    codec: str | None = None


class FinalResult(StrictModel):
    task_id: UUID
    video: VideoMetadata
    transitions: list[Transition]
    tracks: list[TrackedObject]
    completed_at: datetime
    pipeline_version: str = "1.0"
    warnings: list[str] = Field(default_factory=list)


class ErrorResponse(StrictModel):
    detail: str
    code: str


class HealthResponse(StrictModel):
    status: str
    dependencies: dict[str, str]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class StreamEnvelope(StrictModel):
    event_type: str
    schema_version: str = "1"
    payload: dict[str, Any]
    emitted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
