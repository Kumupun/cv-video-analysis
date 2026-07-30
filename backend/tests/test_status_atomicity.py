from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest

from app.core.config import Settings
from app.domain.enums import TaskStage
from app.domain.schemas import TaskStatus
from app.infrastructure.task_repository import RedisTaskRepository


class _ConcurrentStatusRedis:
    """Small Redis stub that reproduces the lost-counter race from Docker.

    The first HGETALL is ingest's stale read. Immediately after returning that
    snapshot, the stub simulates coordinator/aggregator Lua increments. A safe
    status update must patch fields atomically and leave those increments intact.
    """

    def __init__(self, task_id: str) -> None:
        now = datetime.now(UTC)
        status = TaskStatus(
            task_id=task_id,
            stage=TaskStage.TRACKING,
            progress=75.0,
            message="Tracking completed for 43/86 chunks",
            created_at=now,
            updated_at=now,
            processing_started_at=now,
            total_chunks=86,
            cut_completed_chunks=18,
            tracking_completed_chunks=43,
        )
        self.data = RedisTaskRepository._status_mapping(status)
        self.hgetall_calls = 0
        self.eval_calls: list[tuple[Any, ...]] = []

    async def hgetall(self, key: str) -> dict[str, str]:
        self.hgetall_calls += 1
        snapshot = dict(self.data)
        if self.hgetall_calls == 1:
            # The downstream stages finish another chunk after ingest read the
            # old status but before ingest persists its progress message.
            self.data["cut_completed_chunks"] = "19"
            self.data["tracking_completed_chunks"] = "44"
        return snapshot

    async def eval(self, *args: Any) -> list[Any]:
        self.eval_calls.append(args)
        script = str(args[0])
        assert "hset_max('tracking_completed_chunks'" in script
        assert "A late ingest/cut update must never move a task backwards" in script

        (
            _script,
            _key_count,
            _task_key,
            requested_stage,
            requested_progress,
            requested_message,
            total_chunks,
            cut_completed,
            tracking_completed,
            error_code,
            error_detail,
            updated_at,
            _ttl,
        ) = args

        if requested_stage:
            current_rank = {
                "queued": 0,
                "downloading": 1,
                "decoding": 2,
                "cut_detection": 3,
                "tracking": 4,
                "aggregating": 5,
                "completed": 6,
                "failed": 6,
            }[self.data["stage"]]
            requested_rank = {
                "queued": 0,
                "downloading": 1,
                "decoding": 2,
                "cut_detection": 3,
                "tracking": 4,
                "aggregating": 5,
                "completed": 6,
                "failed": 6,
            }[str(requested_stage)]
            if requested_rank >= current_rank:
                self.data["stage"] = str(requested_stage)

        old_progress = float(self.data["progress"])
        if requested_progress != "":
            incoming_progress = float(requested_progress)
            if incoming_progress >= old_progress:
                self.data["progress"] = str(incoming_progress)
                if requested_message:
                    self.data["message"] = str(requested_message)
        elif requested_message:
            self.data["message"] = str(requested_message)

        for field, value in (
            ("total_chunks", total_chunks),
            ("cut_completed_chunks", cut_completed),
            ("tracking_completed_chunks", tracking_completed),
        ):
            if value != "":
                self.data[field] = str(max(int(self.data[field]), int(value)))

        if error_code:
            self.data["error_code"] = str(error_code)
        if error_detail:
            self.data["error_detail"] = str(error_detail)
        self.data["updated_at"] = str(updated_at)
        return [1, self.data["stage"]]

    async def hset(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError(
            "update_status must not perform a full read/modify/HSET write"
        )


@pytest.mark.asyncio
async def test_ingest_progress_cannot_erase_concurrent_stage_counters() -> None:
    task_id = uuid4()
    redis = _ConcurrentStatusRedis(str(task_id))
    repository = RedisTaskRepository(redis, Settings(_env_file=None))

    result = await repository.update_status(
        task_id,
        stage=TaskStage.CUT_DETECTION,
        progress=30.0,
        message="All chunks decoded and safely queued",
    )

    assert len(redis.eval_calls) == 1
    assert result.stage == TaskStage.TRACKING
    assert result.progress == 75.0
    assert result.message == "Tracking completed for 43/86 chunks"
    assert result.cut_completed_chunks == 19
    assert result.tracking_completed_chunks == 44
