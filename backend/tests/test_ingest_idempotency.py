from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.domain.enums import TaskStage
from app.infrastructure.redis_streams import StreamMessage
from app.workers.common import TerminalWorkerError
from app.workers.ingest import IngestHandler


class _RepositoryStub:
    def __init__(self, stage: TaskStage, *, ingest_complete: bool) -> None:
        self.stage = stage
        self.ingest_complete = ingest_complete
        self.source_reads = 0

    async def get_status(self, task_id: object) -> SimpleNamespace:
        return SimpleNamespace(stage=self.stage)

    async def is_ingest_complete(self, task_id: object) -> bool:
        return self.ingest_complete

    async def get_source(self, task_id: object) -> object:
        self.source_reads += 1
        raise AssertionError(
            "Ingest should not read the source for a completed delivery"
        )


def _message() -> StreamMessage:
    return StreamMessage(
        stream="cv:ingest_jobs",
        event_id="1-0",
        fields={"task_id": str(uuid4())},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", [TaskStage.COMPLETED, TaskStage.FAILED])
async def test_ingest_replay_is_ignored_for_terminal_task(stage: TaskStage) -> None:
    handler = IngestHandler.__new__(IngestHandler)
    repository = _RepositoryStub(stage, ingest_complete=False)
    handler.repository = repository

    await handler(_message())

    assert repository.source_reads == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stage",
    [TaskStage.CUT_DETECTION, TaskStage.TRACKING, TaskStage.AGGREGATING],
)
async def test_ingest_replay_is_ignored_only_after_ingest_completed(
    stage: TaskStage,
) -> None:
    handler = IngestHandler.__new__(IngestHandler)
    repository = _RepositoryStub(stage, ingest_complete=True)
    handler.repository = repository

    await handler(_message())

    assert repository.source_reads == 0


@pytest.mark.asyncio
async def test_incomplete_cut_detection_delivery_is_resumed() -> None:
    class ResumeRepository(_RepositoryStub):
        async def get_source(self, task_id: object) -> object:
            self.source_reads += 1
            raise RuntimeError("resume attempted")

        async def fail(self, task_id: object, *, code: str, detail: str) -> None:
            return None

    handler = IngestHandler.__new__(IngestHandler)
    repository = ResumeRepository(TaskStage.CUT_DETECTION, ingest_complete=False)
    handler.repository = repository

    with pytest.raises(TerminalWorkerError, match="resume attempted"):
        await handler(_message())

    assert repository.source_reads == 1
