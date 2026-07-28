from __future__ import annotations

from enum import StrEnum


class TaskStage(StrEnum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    DECODING = "decoding"
    CUT_DETECTION = "cut_detection"
    TRACKING = "tracking"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"


class TransitionType(StrEnum):
    HARD_CUT = "hard_cut"
    GRADUAL_TRANSITION = "gradual_transition"


class SourceKind(StrEnum):
    UPLOAD = "upload"
    URL = "url"


class WorkerKind(StrEnum):
    INGEST = "ingest"
    CUT = "cut"
    COORDINATOR = "coordinator"
    TRACKING = "tracking"
    AGGREGATOR = "aggregator"
