from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.core.config import Settings
from app.domain.enums import TaskStage
from app.workers.ingest import DownstreamStallError, IngestHandler


class _Repository:
    async def get_backpressure_state(self, task_id):
        return TaskStage.CUT_DETECTION, 0


@pytest.mark.asyncio
async def test_ingest_backpressure_fails_instead_of_waiting_forever() -> None:
    handler = IngestHandler.__new__(IngestHandler)
    handler.settings = Settings(
        _env_file=None,
        max_inflight_chunks_per_task=1,
        max_inflight_chunks_global=2,
        ingest_backpressure_poll_seconds=0.005,
        downstream_stall_timeout_seconds=0.03,
        ray_put_timeout_seconds=1,
    )
    handler.repository = _Repository()
    handler.object_store = SimpleNamespace(registered_count=lambda: 1)
    handler._active_tasks = {uuid4(), uuid4()}
    handler._active_tasks_lock = asyncio.Lock()

    with pytest.raises(DownstreamStallError, match="made no progress"):
        await handler._wait_for_downstream_capacity(
            uuid4(), published_count=1, released_count=0
        )
