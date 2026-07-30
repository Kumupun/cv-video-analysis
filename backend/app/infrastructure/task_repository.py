from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from app.core.config import Settings
from app.domain.enums import TaskStage
from app.domain.schemas import FinalResult, TaskStatus, VideoMetadata, VideoSource
from app.domain.state_machine import ensure_stage_transition


class TaskNotFoundError(KeyError):
    pass


class TaskStatusUnavailableError(RuntimeError):
    """A task exists, but its status could not be read reliably right now."""

    pass


class TrackingOrderError(RuntimeError):
    """A tracking result was produced ahead of the durable Redis sequence."""


_STAGE_RANK = {
    TaskStage.QUEUED: 0,
    TaskStage.DOWNLOADING: 1,
    TaskStage.DECODING: 2,
    TaskStage.CUT_DETECTION: 3,
    TaskStage.TRACKING: 4,
    TaskStage.AGGREGATING: 5,
    TaskStage.COMPLETED: 6,
    TaskStage.FAILED: 6,
}


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
        """Return one canonical Redis key for a task chunk.

        Pipeline messages use chunk IDs in the form ``<task_id>:<index>``.
        The Redis key already contains ``task_id`` as its own namespace, so
        storing that prefix a second time creates keys such as
        ``cv:chunk:<task>:<task>:00000000``.  The contiguous-dispatch Lua
        script reconstructs keys as ``cv:chunk:<task>:00000000``; the old
        double prefix therefore made every lookup miss and stopped the
        pipeline immediately after the first cut result.
        """

        task_text = str(task_id)
        chunk_text = str(chunk_id)
        message_prefix = f"{task_text}:"
        if chunk_text.startswith(message_prefix):
            chunk_text = chunk_text[len(message_prefix) :]
        return f"cv:chunk:{task_text}:{chunk_text}"

    @classmethod
    def _chunk_key_prefix(cls, task_id: UUID | str) -> str:
        """Prefix used by Lua to rebuild deterministic chunk keys by index."""

        return cls._chunk_key(task_id, "")

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
        """Atomically create one task state and append its ingest event."""

        statuses = await self.create_many_and_enqueue(
            [(task_id, source, stream_fields)],
            stream,
        )
        return statuses[0]

    async def create_many_and_enqueue(
        self,
        tasks: list[tuple[UUID, VideoSource, dict[str, str]]],
        stream: str,
    ) -> list[TaskStatus]:
        """Atomically create and enqueue all tasks from one archive upload."""

        if not tasks:
            raise ValueError("At least one task is required")

        now = datetime.now(UTC)
        statuses: list[TaskStatus] = []
        pipe = self._redis.pipeline(transaction=True)
        for task_id, source, stream_fields in tasks:
            task_status = TaskStatus(
                task_id=task_id,
                stage=TaskStage.QUEUED,
                progress=0.0,
                message="Task accepted and queued",
                created_at=now,
                updated_at=now,
            )
            statuses.append(task_status)
            mapping = self._status_mapping(task_status)
            mapping["source"] = source.model_dump_json()
            key = self._task_key(task_id)
            pipe.hset(key, mapping=mapping)
            pipe.expire(key, self._settings.task_ttl_seconds)
            pipe.xadd(
                stream,
                stream_fields,
                maxlen=self._settings.stream_maxlen,
                approximate=True,
            )
        await pipe.execute()
        return statuses

    async def get_status(self, task_id: UUID | str) -> TaskStatus:
        last_error: Exception | None = None
        for attempt in range(self._settings.status_read_max_attempts):
            try:
                data = await self._redis.hgetall(self._task_key(task_id))
                if not data:
                    raise TaskNotFoundError(str(task_id))
                return self._status_from_mapping(data)
            except TaskNotFoundError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self._settings.status_read_max_attempts:
                    await asyncio.sleep(self._settings.status_read_retry_delay_seconds)

        raise TaskStatusUnavailableError(
            f"Task status is temporarily unavailable: {task_id}"
        ) from last_error

    async def get_backpressure_state(
        self,
        task_id: UUID | str,
    ) -> tuple[TaskStage, int]:
        """Read only the fields needed by ingest backpressure polling."""

        stage, completed = await self._redis.hmget(
            self._task_key(task_id),
            "stage",
            "tracking_completed_chunks",
        )
        if stage is None:
            raise TaskNotFoundError(str(task_id))
        return TaskStage(str(stage)), int(completed or 0)

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
        """Atomically patch status without overwriting concurrent counters.

        Cut coordination, tracking aggregation and ingest progress all mutate the
        same Redis hash.  The previous read/modify/full-HSET implementation could
        read ``tracking_completed_chunks=43``, then an aggregator incremented it
        to 44, and finally ingest wrote the stale value 43 back.  The stream
        event was already acknowledged, so backpressure waited forever for a
        completion that had effectively been erased.

        This Lua patch updates only fields supplied by the caller, applies
        monotonic stage/progress/counter rules inside Redis, and therefore cannot
        clobber increments performed by the other pipeline Lua scripts.
        """

        current = await self.get_status(task_id)
        target_stage = stage or current.stage
        if current.stage in {TaskStage.COMPLETED, TaskStage.FAILED}:
            return current
        # A caller can read CUT_DETECTION, then coordinator advances the task to
        # TRACKING before this method runs. That late CUT_DETECTION update is a
        # harmless stale patch and must be ignored rather than treated as an
        # invalid backwards transition.
        if _STAGE_RANK[target_stage] >= _STAGE_RANK[current.stage]:
            ensure_stage_transition(current.stage, target_stage)

        script = """
        if redis.call('EXISTS', KEYS[1]) == 0 then
            return {-1, ''}
        end

        local current_stage = redis.call('HGET', KEYS[1], 'stage') or 'queued'
        if current_stage == 'completed' or current_stage == 'failed' then
            return {0, current_stage}
        end

        local ranks = {
            queued = 0,
            downloading = 1,
            decoding = 2,
            cut_detection = 3,
            tracking = 4,
            aggregating = 5,
            completed = 6,
            failed = 6
        }
        local allowed = {
            queued = {downloading = true, decoding = true, failed = true},
            downloading = {decoding = true, failed = true},
            decoding = {cut_detection = true, failed = true},
            cut_detection = {tracking = true, failed = true},
            tracking = {aggregating = true, failed = true},
            aggregating = {completed = true, failed = true}
        }

        local requested_stage = ARGV[1]
        local stale_stage = false
        if requested_stage ~= '' and requested_stage ~= current_stage then
            local current_rank = ranks[current_stage]
            local requested_rank = ranks[requested_stage]
            if requested_rank == nil then
                return {-2, current_stage}
            end
            if requested_stage == 'failed' then
                redis.call('HSET', KEYS[1], 'stage', requested_stage)
                current_stage = requested_stage
            elseif requested_rank < current_rank then
                -- A late ingest/cut update must never move a task backwards.
                stale_stage = true
            elseif allowed[current_stage] and allowed[current_stage][requested_stage] then
                redis.call('HSET', KEYS[1], 'stage', requested_stage)
                current_stage = requested_stage
            else
                return {-2, current_stage}
            end
        end

        local current_progress = tonumber(
            redis.call('HGET', KEYS[1], 'progress') or '0'
        )
        local requested_progress = nil
        if ARGV[2] ~= '' then
            requested_progress = tonumber(ARGV[2])
            if requested_progress > current_progress then
                redis.call('HSET', KEYS[1], 'progress', requested_progress)
                current_progress = requested_progress
            end
        end

        -- Do not let an old lower-progress message replace a newer stage text.
        if ARGV[3] ~= '' and not stale_stage and (
            requested_progress == nil or requested_progress >= current_progress
        ) then
            redis.call('HSET', KEYS[1], 'message', ARGV[3])
        end

        local function hset_max(field, raw_value)
            if raw_value == '' then
                return
            end
            local incoming = tonumber(raw_value)
            local existing = tonumber(redis.call('HGET', KEYS[1], field) or '0')
            if incoming > existing then
                redis.call('HSET', KEYS[1], field, incoming)
            end
        end

        hset_max('total_chunks', ARGV[4])
        hset_max('cut_completed_chunks', ARGV[5])
        hset_max('tracking_completed_chunks', ARGV[6])

        if ARGV[7] ~= '' then
            redis.call('HSET', KEYS[1], 'error_code', ARGV[7])
        end
        if ARGV[8] ~= '' then
            redis.call('HSET', KEYS[1], 'error_detail', ARGV[8])
        end

        if current_stage ~= 'queued' and (
            redis.call('HGET', KEYS[1], 'processing_started_at') or ''
        ) == '' then
            redis.call('HSET', KEYS[1], 'processing_started_at', ARGV[9])
        end
        redis.call('HSET', KEYS[1], 'updated_at', ARGV[9])
        redis.call('EXPIRE', KEYS[1], ARGV[10])
        return {1, current_stage}
        """

        now = datetime.now(UTC).isoformat()
        result = await self._redis.eval(
            script,
            1,
            self._task_key(task_id),
            "" if stage is None else stage.value,
            "" if progress is None else progress,
            "" if message is None else message,
            "" if total_chunks is None else total_chunks,
            "" if cut_completed_chunks is None else cut_completed_chunks,
            "" if tracking_completed_chunks is None else tracking_completed_chunks,
            "" if error_code is None else error_code,
            "" if error_detail is None else error_detail[:2_000],
            now,
            self._settings.task_ttl_seconds,
        )
        code = int(result[0])
        if code == -1:
            raise TaskNotFoundError(str(task_id))
        if code == -2:
            raise ValueError(
                "Invalid concurrent pipeline transition: "
                f"{result[1]} -> {target_stage.value}"
            )
        return await self.get_status(task_id)

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

    async def is_ingest_complete(self, task_id: UUID | str) -> bool:
        raw = await self._redis.hget(self._task_key(task_id), "ingest_complete")
        return str(raw or "0") == "1"

    async def get_ingest_published_chunks(self, task_id: UUID | str) -> int:
        raw = await self._redis.hget(
            self._task_key(task_id),
            "ingest_published_chunks",
        )
        return int(raw or 0)

    async def get_chunk_object_ref(
        self,
        task_id: UUID | str,
        chunk_index: int,
    ) -> str:
        """Return the Ray token persisted for a deterministic chunk index."""

        chunk_id = f"{task_id}:{chunk_index:08d}"
        raw = await self._redis.hget(
            self._chunk_key(task_id, chunk_id),
            "object_ref",
        )
        if raw is None or not str(raw):
            raise TaskNotFoundError(f"object_ref:{chunk_id}")
        return str(raw)

    async def is_ingest_chunk_published(
        self,
        task_id: UUID | str,
        chunk_id: str,
    ) -> bool:
        raw = await self._redis.hget(
            self._chunk_key(task_id, chunk_id),
            "ingest_published",
        )
        return str(raw or "0") == "1"

    async def publish_ingest_chunk_once(
        self,
        task_id: UUID | str,
        chunk_id: str,
        *,
        stream: str,
        stream_fields: dict[str, str],
        object_ref: str,
        chunk_index: int,
    ) -> bool:
        """Atomically persist and enqueue one decoded chunk exactly once."""

        required_fields = {"event_type", "schema_version", "payload"}
        if not required_fields.issubset(stream_fields):
            raise ValueError("Stream fields are missing the serialized chunk payload")

        script = """
        if redis.call('HGET', KEYS[2], 'ingest_published') == '1' then
            return 0
        end
        redis.call('HSET', KEYS[2],
            'ingest_published', '1',
            'ingest_payload', ARGV[1],
            'object_ref', ARGV[2],
            'chunk_index', ARGV[3])
        redis.call('EXPIRE', KEYS[2], ARGV[4])
        redis.call('HINCRBY', KEYS[1], 'ingest_published_chunks', 1)
        redis.call('EXPIRE', KEYS[1], ARGV[4])
        redis.call('XADD', KEYS[3],
            'MAXLEN', '~', ARGV[5], '*',
            'event_type', ARGV[6],
            'schema_version', ARGV[7],
            'payload', ARGV[1])
        return 1
        """
        result = await self._redis.eval(
            script,
            3,
            self._task_key(task_id),
            self._chunk_key(task_id, chunk_id),
            stream,
            stream_fields["payload"],
            object_ref,
            chunk_index,
            self._settings.task_ttl_seconds,
            self._settings.stream_maxlen,
            stream_fields["event_type"],
            stream_fields["schema_version"],
        )
        return int(result) == 1

    async def mark_ingest_complete(
        self,
        task_id: UUID | str,
        total_chunks: int,
    ) -> None:
        """Mark decoding complete only after every chunk was enqueued."""

        script = """
        local published = tonumber(
            redis.call('HGET', KEYS[1], 'ingest_published_chunks') or '0'
        )
        if published < tonumber(ARGV[1]) then
            return 0
        end
        redis.call('HSET', KEYS[1], 'ingest_complete', '1')
        redis.call('EXPIRE', KEYS[1], ARGV[2])
        return 1
        """
        result = await self._redis.eval(
            script,
            1,
            self._task_key(task_id),
            total_chunks,
            self._settings.task_ttl_seconds,
        )
        if int(result) != 1:
            published = await self.get_ingest_published_chunks(task_id)
            raise RuntimeError(
                f"Ingest finished with {published}/{total_chunks} chunks published"
            )

    async def record_cut_and_dispatch(
        self,
        task_id: UUID | str,
        chunk_id: str,
        chunk_index: int,
        *,
        cut_payload: str,
        object_ref: str,
        tracking_stream: str,
        tracking_stream_fields: dict[str, str],
    ) -> tuple[bool, int, int, int]:
        """Store cut output and dispatch every contiguous tracking job once.

        One Lua call replaces the former distributed lock plus repeated
        HGET/XADD/advance round-trips. Out-of-order cut results remain buffered
        in their chunk hashes until the missing predecessor arrives.
        """

        required_fields = {"event_type", "schema_version", "payload"}
        if not required_fields.issubset(tracking_stream_fields):
            raise ValueError("Stream fields are missing the tracking payload")

        script = """
        local is_new = redis.call('HEXISTS', KEYS[2], 'cut_done') == 0
        redis.call('HSET', KEYS[2],
            'cut_done', '1',
            'cut_payload', ARGV[1],
            'object_ref', ARGV[2],
            'chunk_index', ARGV[3],
            'tracking_job_payload', ARGV[4])
        redis.call('EXPIRE', KEYS[2], ARGV[5])

        local cut_count = tonumber(
            redis.call('HGET', KEYS[1], 'cut_completed_chunks') or '0'
        )
        if is_new then
            cut_count = redis.call(
                'HINCRBY', KEYS[1], 'cut_completed_chunks', 1
            )
        end

        local next_index = tonumber(
            redis.call('HGET', KEYS[1], 'next_tracking_chunk') or '0'
        )
        local dispatched_before = next_index
        while true do
            local next_chunk_key = ARGV[10] .. string.format('%08d', next_index)
            local payload = redis.call(
                'HGET', next_chunk_key, 'tracking_job_payload'
            )
            if not payload then
                break
            end
            if redis.call(
                'HGET', next_chunk_key, 'tracking_job_dispatched'
            ) ~= '1' then
                redis.call('XADD', KEYS[3],
                    'MAXLEN', '~', ARGV[6], '*',
                    'event_type', ARGV[7],
                    'schema_version', ARGV[8],
                    'payload', payload)
                redis.call(
                    'HSET', next_chunk_key, 'tracking_job_dispatched', '1'
                )
            end
            redis.call('HDEL', next_chunk_key, 'tracking_job_payload')
            next_index = next_index + 1
        end
        redis.call('HSET', KEYS[1], 'next_tracking_chunk', next_index)
        redis.call('EXPIRE', KEYS[1], ARGV[5])

        local total = tonumber(redis.call('HGET', KEYS[1], 'total_chunks') or '0')
        local stage = redis.call('HGET', KEYS[1], 'stage') or 'cut_detection'
        local stage_changed = 0
        if next_index > 0 and stage == 'cut_detection' then
            stage = 'tracking'
            stage_changed = 1
        end
        local update_every = tonumber(ARGV[11])
        local should_update = stage_changed == 1
            or (is_new and total > 0 and cut_count >= total)
            or (is_new and cut_count % update_every == 0)
        if should_update and stage ~= 'completed' and stage ~= 'failed' then
            local progress = 30
            if total > 0 then
                progress = 30 + 30 * math.min(cut_count / total, 1)
            end
            local current_progress = tonumber(
                redis.call('HGET', KEYS[1], 'progress') or '0'
            )
            redis.call('HSET', KEYS[1],
                'stage', stage,
                'progress', math.max(current_progress, progress),
                'message', 'Cut detection verified ' .. cut_count .. '/' .. total
                    .. ' chunks; tracking dispatched through chunk '
                    .. (next_index - 1),
                'updated_at', ARGV[9])
        end
        return {is_new and 1 or 0, cut_count, next_index, total,
            next_index - dispatched_before}
        """
        result = await self._redis.eval(
            script,
            3,
            self._task_key(task_id),
            self._chunk_key(task_id, chunk_id),
            tracking_stream,
            cut_payload,
            object_ref,
            chunk_index,
            tracking_stream_fields["payload"],
            self._settings.task_ttl_seconds,
            self._settings.stream_maxlen,
            tracking_stream_fields["event_type"],
            tracking_stream_fields["schema_version"],
            datetime.now(UTC).isoformat(),
            self._chunk_key_prefix(task_id),
            self._settings.progress_update_every_chunks,
        )
        return (
            int(result[0]) == 1,
            int(result[1]),
            int(result[2]),
            int(result[3]),
        )

    async def mark_chunk_tracking_done(
        self,
        task_id: UUID | str,
        chunk_id: str,
        tracking_payload: str,
    ) -> tuple[bool, bool, int, int]:
        """Persist tracking output and update progress in one Redis call.

        Returns ``(cut_done, is_new, completed_count, total_chunks)``.
        The ingest owner observes this durable counter and releases each Ray
        object in contiguous order from the same Ray Client connection that
        created it.
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
        local total = tonumber(
            redis.call('HGET', KEYS[2], 'total_chunks') or '0'
        )
        local cut_done = redis.call('HGET', KEYS[1], 'cut_done') or '0'
        local stage = redis.call('HGET', KEYS[2], 'stage') or 'tracking'
        local update_every = tonumber(ARGV[4])
        local should_update = is_new and (
            (total > 0 and completed >= total)
            or completed % update_every == 0
        )
        if should_update and stage ~= 'completed' and stage ~= 'failed' then
            local progress = 60
            if total > 0 then
                progress = 60 + 30 * math.min(completed / total, 1)
            end
            local target_stage = 'tracking'
            if total > 0 and completed >= total then
                target_stage = 'aggregating'
            end
            local current_progress = tonumber(
                redis.call('HGET', KEYS[2], 'progress') or '0'
            )
            redis.call('HSET', KEYS[2],
                'stage', target_stage,
                'progress', math.max(current_progress, progress),
                'message', 'Tracking completed for ' .. completed .. '/'
                    .. total .. ' chunks',
                'updated_at', ARGV[3])
            redis.call('EXPIRE', KEYS[2], ARGV[2])
        end
        return {cut_done, is_new and 1 or 0, completed, total}
        """
        result = await self._redis.eval(
            script,
            2,
            chunk_key,
            task_key,
            tracking_payload,
            self._settings.task_ttl_seconds,
            datetime.now(UTC).isoformat(),
            self._settings.progress_update_every_chunks,
        )
        return (
            str(result[0]) == "1",
            int(result[1]) == 1,
            int(result[2]),
            int(result[3]),
        )

    async def publish_tracking_result_in_order(
        self,
        task_id: UUID | str,
        chunk_id: str,
        chunk_index: int,
        *,
        stream: str,
        stream_fields: dict[str, str],
    ) -> bool:
        """Atomically publish one tracking result in durable chunk order.

        Returns ``True`` when a new stream event was appended and ``False`` for
        a delivery that had already been published. A future chunk raises
        :class:`TrackingOrderError` instead of advancing the durable sequence.
        """

        required_fields = {"event_type", "schema_version", "payload"}
        if not required_fields.issubset(stream_fields):
            raise ValueError("Stream fields are missing the tracking payload")

        script = """
        local current = tonumber(
            redis.call('HGET', KEYS[1], 'next_tracking_result_chunk') or '0'
        )
        local requested = tonumber(ARGV[1])

        if requested < current then
            return {0, current}
        end
        if requested > current then
            return {-1, current}
        end

        redis.call('HSET', KEYS[2], 'tracking_result_published', '1')
        redis.call('EXPIRE', KEYS[2], ARGV[2])
        redis.call('HSET', KEYS[1],
            'next_tracking_result_chunk', current + 1)
        redis.call('EXPIRE', KEYS[1], ARGV[2])
        redis.call('XADD', KEYS[3],
            'MAXLEN', '~', ARGV[3], '*',
            'event_type', ARGV[4],
            'schema_version', ARGV[5],
            'payload', ARGV[6])
        return {1, current + 1}
        """
        result = await self._redis.eval(
            script,
            3,
            self._task_key(task_id),
            self._chunk_key(task_id, chunk_id),
            stream,
            chunk_index,
            self._settings.task_ttl_seconds,
            self._settings.stream_maxlen,
            stream_fields["event_type"],
            stream_fields["schema_version"],
            stream_fields["payload"],
        )
        published = int(result[0])
        expected = int(result[1])
        if published < 0:
            raise TrackingOrderError(
                "Out-of-order tracking result: "
                f"expected chunk {expected}, received {chunk_index}"
            )
        return published == 1

    async def get_all_chunk_payloads(
        self,
        task_id: UUID | str,
        total_chunks: int | None = None,
    ) -> list[dict[str, str]]:
        """Fetch deterministic chunk hashes in one Redis pipeline.

        The old SCAN followed by one HGETALL per key caused an avoidable
        round-trip for every chunk during final aggregation.
        """

        if total_chunks is None:
            raw_total = await self._redis.hget(
                self._task_key(task_id),
                "total_chunks",
            )
            total_chunks = int(raw_total or 0)
        if total_chunks <= 0:
            return []

        pipe = self._redis.pipeline(transaction=False)
        for chunk_index in range(total_chunks):
            chunk_id = f"{task_id}:{chunk_index:08d}"
            pipe.hgetall(self._chunk_key(task_id, chunk_id))
        rows = await pipe.execute()
        return [dict(row) for row in rows if row]

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
            processing_started_at=(
                data.get("processing_started_at") or None
            ),
            error_code=data.get("error_code") or None,
            error_detail=data.get("error_detail") or None,
        )
