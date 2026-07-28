from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from app.core.config import Settings
from app.domain.enums import TaskStage
from app.domain.schemas import FinalResult, TaskStatus, VideoMetadata, VideoSource
from app.domain.state_machine import ensure_stage_transition


class TaskNotFoundError(KeyError):
    pass


class RedisTaskRepository:
    def __init__(self, redis_client: Any, settings: Settings) -> None:
        self._redis = redis_client
        self._settings = settings

    @staticmethod
    def _task_key(task_id: UUID | str) -> str:
        return f"cv:task:{task_id}"

    @staticmethod
    def _result_key(task_id: UUID | str) -> str:
        return f"cv:result:{task_id}"

    @staticmethod
    def _chunk_key(task_id: UUID | str, chunk_id: str) -> str:
        return f"cv:chunk:{task_id}:{chunk_id}"

    @staticmethod
    def _metadata_key(task_id: UUID | str) -> str:
        return f"cv:video_metadata:{task_id}"

    async def create(self, task_id: UUID, source: VideoSource) -> TaskStatus:
        now = datetime.now(UTC)
        status = TaskStatus(
            task_id=task_id,
            stage=TaskStage.QUEUED,
            progress=0.0,
            message="Task accepted and queued",
            created_at=now,
            updated_at=now,
        )
        mapping = self._status_mapping(status)
        mapping["source"] = source.model_dump_json()
        key = self._task_key(task_id)
        await self._redis.hset(key, mapping=mapping)
        await self._redis.expire(key, self._settings.task_ttl_seconds)
        return status

    async def create_and_enqueue(
        self,
        task_id: UUID,
        source: VideoSource,
        stream: str,
        stream_fields: dict[str, str],
    ) -> TaskStatus:
        """Atomically create task state and append its ingest event."""

        now = datetime.now(UTC)
        status = TaskStatus(
            task_id=task_id,
            stage=TaskStage.QUEUED,
            progress=0.0,
            message="Task accepted and queued",
            created_at=now,
            updated_at=now,
        )
        mapping = self._status_mapping(status)
        mapping["source"] = source.model_dump_json()
        key = self._task_key(task_id)
        pipe = self._redis.pipeline(transaction=True)
        pipe.hset(key, mapping=mapping)
        pipe.expire(key, self._settings.task_ttl_seconds)
        pipe.xadd(
            stream,
            stream_fields,
            maxlen=self._settings.stream_maxlen,
            approximate=True,
        )
        await pipe.execute()
        return status

    async def get_status(self, task_id: UUID | str) -> TaskStatus:
        data = await self._redis.hgetall(self._task_key(task_id))
        if not data:
            raise TaskNotFoundError(str(task_id))
        return self._status_from_mapping(data)

    async def get_source(self, task_id: UUID | str) -> VideoSource:
        raw = await self._redis.hget(self._task_key(task_id), "source")
        if raw is None:
            raise TaskNotFoundError(str(task_id))
        return VideoSource.model_validate_json(raw)

    async def update_status(
        self,
        task_id: UUID | str,
        *,
        stage: TaskStage | None = None,
        progress: float | None = None,
        message: str | None = None,
        total_chunks: int | None = None,
        cut_completed_chunks: int | None = None,
        tracking_completed_chunks: int | None = None,
        error_code: str | None = None,
        error_detail: str | None = None,
    ) -> TaskStatus:
        current = await self.get_status(task_id)
        target_stage = stage or current.stage
        if current.stage in {TaskStage.COMPLETED, TaskStage.FAILED}:
            return current
        ensure_stage_transition(current.stage, target_stage)
        updated = current.model_copy(
            update={
                "stage": target_stage,
                "progress": (
                    current.progress
                    if progress is None
                    else max(current.progress, progress)
                ),
                "message": current.message if message is None else message,
                "updated_at": datetime.now(UTC),
                "total_chunks": (
                    current.total_chunks
                    if total_chunks is None
                    else max(current.total_chunks, total_chunks)
                ),
                "cut_completed_chunks": (
                    current.cut_completed_chunks
                    if cut_completed_chunks is None
                    else max(current.cut_completed_chunks, cut_completed_chunks)
                ),
                "tracking_completed_chunks": (
                    current.tracking_completed_chunks
                    if tracking_completed_chunks is None
                    else max(
                        current.tracking_completed_chunks,
                        tracking_completed_chunks,
                    )
                ),
                "error_code": error_code,
                "error_detail": error_detail,
            }
        )
        await self._redis.hset(
            self._task_key(task_id), mapping=self._status_mapping(updated)
        )
        return updated

    async def fail(self, task_id: UUID | str, *, code: str, detail: str) -> TaskStatus:
        current = await self.get_status(task_id)
        if current.stage in {TaskStage.COMPLETED, TaskStage.FAILED}:
            return current
        return await self.update_status(
            task_id,
            stage=TaskStage.FAILED,
            message="Task failed",
            error_code=code,
            error_detail=detail[:2_000],
        )

    async def save_video_metadata(
        self, task_id: UUID | str, metadata: VideoMetadata
    ) -> None:
        key = self._metadata_key(task_id)
        await self._redis.set(
            key, metadata.model_dump_json(), ex=self._settings.task_ttl_seconds
        )

    async def get_video_metadata(self, task_id: UUID | str) -> VideoMetadata:
        raw = await self._redis.get(self._metadata_key(task_id))
        if raw is None:
            raise TaskNotFoundError(f"metadata:{task_id}")
        return VideoMetadata.model_validate_json(raw)

    async def mark_chunk_cut_done(
        self,
        task_id: UUID | str,
        chunk_id: str,
        cut_payload: str,
        object_ref: str,
    ) -> tuple[bool, int]:
        """Persist cut output exactly once.

        Returns ``(is_new, completed_count)`` so duplicate Redis deliveries do
        not publish duplicate tracking jobs.
        """

        chunk_key = self._chunk_key(task_id, chunk_id)
        task_key = self._task_key(task_id)
        script = """
        local is_new = redis.call('HEXISTS', KEYS[1], 'cut_done') == 0
        redis.call('HSET', KEYS[1],
            'cut_done', '1',
            'cut_payload', ARGV[1],
            'object_ref', ARGV[2])
        redis.call('EXPIRE', KEYS[1], ARGV[3])
        local completed = tonumber(
            redis.call('HGET', KEYS[2], 'cut_completed_chunks') or '0'
        )
        if is_new then
            completed = redis.call(
                'HINCRBY', KEYS[2], 'cut_completed_chunks', 1
            )
        end
        return {is_new and 1 or 0, completed}
        """
        result = await self._redis.eval(
            script,
            2,
            chunk_key,
            task_key,
            cut_payload,
            object_ref,
            self._settings.task_ttl_seconds,
        )
        return int(result[0]) == 1, int(result[1])

    async def mark_chunk_tracking_done(
        self,
        task_id: UUID | str,
        chunk_id: str,
        tracking_payload: str,
    ) -> tuple[bool, bool, int]:
        """Persist tracking output exactly once.

        Returns ``(cut_done, is_new, completed_count)``. The caller releases
        the Ray ObjectRef only when ``is_new`` is true.
        """

        chunk_key = self._chunk_key(task_id, chunk_id)
        task_key = self._task_key(task_id)
        script = """
        local is_new = redis.call('HEXISTS', KEYS[1], 'tracking_done') == 0
        redis.call('HSET', KEYS[1],
            'tracking_done', '1',
            'tracking_payload', ARGV[1])
        redis.call('EXPIRE', KEYS[1], ARGV[2])
        local completed = tonumber(
            redis.call('HGET', KEYS[2], 'tracking_completed_chunks') or '0'
        )
        if is_new then
            completed = redis.call(
                'HINCRBY', KEYS[2], 'tracking_completed_chunks', 1
            )
        end
        local cut_done = redis.call('HGET', KEYS[1], 'cut_done') or '0'
        return {cut_done, is_new and 1 or 0, completed}
        """
        result = await self._redis.eval(
            script,
            2,
            chunk_key,
            task_key,
            tracking_payload,
            self._settings.task_ttl_seconds,
        )
        return str(result[0]) == "1", int(result[1]) == 1, int(result[2])

    async def get_next_tracking_dispatch_index(self, task_id: UUID | str) -> int:
        raw = await self._redis.hget(self._task_key(task_id), "next_tracking_chunk")
        return int(raw or 0)

    async def get_cut_payload_for_index(
        self, task_id: UUID | str, chunk_index: int
    ) -> str | None:
        chunk_id = f"{task_id}:{chunk_index:08d}"
        return await self._redis.hget(self._chunk_key(task_id, chunk_id), "cut_payload")

    async def advance_tracking_dispatch(
        self, task_id: UUID | str, expected_index: int
    ) -> bool:
        script = """
        local current = tonumber(
            redis.call('HGET', KEYS[1], 'next_tracking_chunk') or '0'
        )
        if current ~= tonumber(ARGV[1]) then
            return 0
        end
        redis.call('HSET', KEYS[1], 'next_tracking_chunk', current + 1)
        return 1
        """
        result = await self._redis.eval(
            script,
            1,
            self._task_key(task_id),
            expected_index,
        )
        return int(result) == 1

    async def get_all_chunk_payloads(self, task_id: UUID | str) -> list[dict[str, str]]:
        pattern = self._chunk_key(task_id, "*")
        cursor = 0
        chunks: list[dict[str, str]] = []
        while True:
            cursor, keys = await self._redis.scan(
                cursor=cursor, match=pattern, count=500
            )
            for key in keys:
                data = await self._redis.hgetall(key)
                if data:
                    chunks.append(data)
            if cursor == 0:
                break
        return chunks

    async def save_result(self, task_id: UUID | str, result: FinalResult) -> None:
        await self._redis.set(
            self._result_key(task_id),
            result.model_dump_json(),
            ex=self._settings.result_ttl_seconds,
        )

    async def get_result(self, task_id: UUID | str) -> FinalResult | None:
        raw = await self._redis.get(self._result_key(task_id))
        if raw is None:
            return None
        return FinalResult.model_validate_json(raw)

    @staticmethod
    def _status_mapping(status: TaskStatus) -> dict[str, str]:
        data = status.model_dump(mode="json")
        return {key: "" if value is None else str(value) for key, value in data.items()}

    @staticmethod
    def _status_from_mapping(data: dict[str, str]) -> TaskStatus:
        return TaskStatus(
            task_id=data["task_id"],
            stage=data["stage"],
            progress=float(data["progress"]),
            message=data["message"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            total_chunks=int(data.get("total_chunks", 0) or 0),
            cut_completed_chunks=int(data.get("cut_completed_chunks", 0) or 0),
            tracking_completed_chunks=int(
                data.get("tracking_completed_chunks", 0) or 0
            ),
            error_code=data.get("error_code") or None,
            error_detail=data.get("error_detail") or None,
        )
