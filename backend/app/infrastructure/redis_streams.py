from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from app.core.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StreamMessage:
    stream: str
    event_id: str
    fields: dict[str, str]


class RedisStreams:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: Any | None = None

    async def connect(self) -> None:
        if self._client is not None:
            return
        try:
            from redis.asyncio import Redis
            from redis.backoff import ExponentialBackoff
            from redis.exceptions import BusyLoadingError
            from redis.retry import Retry
        except ImportError as exc:
            raise RuntimeError("redis package is required for production mode") from exc
        self._client = Redis.from_url(
            self._settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=(
                self._settings.redis_socket_connect_timeout_seconds
            ),
            socket_timeout=self._settings.redis_socket_timeout_seconds,
            retry=Retry(
                ExponentialBackoff(cap=1.0, base=0.1),
                self._settings.redis_retry_attempts,
            ),
            retry_on_error=[BusyLoadingError],
            socket_keepalive=True,
            health_check_interval=(self._settings.redis_health_check_interval_seconds),
        )
        await self._client.ping()

    @staticmethod
    def is_transient_error(exc: BaseException) -> bool:
        """Return whether a Redis/network failure should leave work pending.

        Redis may briefly stop responding while Docker Desktop is under heavy
        CPU, disk, or memory pressure. Those failures must not permanently fail
        a video or consume the message delivery budget.
        """

        try:
            from redis.exceptions import (
                BusyLoadingError,
            )
            from redis.exceptions import (
                ConnectionError as RedisConnectionError,
            )
            from redis.exceptions import (
                TimeoutError as RedisTimeoutError,
            )
        except ImportError:
            return False

        current: BaseException | None = exc
        visited: set[int] = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            if isinstance(
                current,
                (RedisTimeoutError, RedisConnectionError, BusyLoadingError),
            ):
                return True
            current = current.__cause__ or current.__context__
        return False

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> Any:
        if self._client is None:
            raise RuntimeError("RedisStreams.connect() must be called first")
        return self._client

    async def ensure_group(self, stream: str, group: str) -> None:
        try:
            await self.client.xgroup_create(stream, group, id="0", mkstream=True)
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def publish(self, stream: str, fields: dict[str, str]) -> str:
        event_id = await self.client.xadd(
            stream,
            fields,
            maxlen=self._settings.stream_maxlen,
            approximate=True,
        )
        logger.info(
            "Published stream event",
            extra={"stream": stream, "event_id": event_id},
        )
        return str(event_id)

    async def read_group(
        self,
        *,
        stream: str,
        group: str,
        consumer: str,
        count: int | None = None,
        block_ms: int | None = None,
    ) -> list[StreamMessage]:
        await self.ensure_group(stream, group)
        response = await self.client.xreadgroup(
            groupname=group,
            consumername=consumer,
            streams={stream: ">"},
            count=count or self._settings.stream_batch_size,
            block=block_ms or self._settings.stream_block_ms,
        )
        messages: list[StreamMessage] = []
        for stream_name, items in response:
            for event_id, fields in items:
                messages.append(
                    StreamMessage(
                        stream=str(stream_name),
                        event_id=str(event_id),
                        fields={str(key): str(value) for key, value in fields.items()},
                    )
                )
        return messages

    async def claim_stale(
        self,
        *,
        stream: str,
        group: str,
        consumer: str,
    ) -> list[StreamMessage]:
        await self.ensure_group(stream, group)
        response = await self.client.xautoclaim(
            stream,
            group,
            consumer,
            min_idle_time=self._settings.stream_claim_idle_ms,
            start_id="0-0",
            count=self._settings.stream_batch_size,
        )
        if not response or len(response) < 2:
            return []
        items = response[1]
        return [
            StreamMessage(
                stream=stream,
                event_id=str(event_id),
                fields={str(key): str(value) for key, value in fields.items()},
            )
            for event_id, fields in items
        ]

    async def touch_pending(
        self,
        message: StreamMessage,
        *,
        group: str,
        consumer: str,
    ) -> None:
        """Reset the pending-entry idle timer while a long task is active.

        Video ingestion may legitimately run for many minutes or hours. Without
        this heartbeat, another consumer could auto-claim the still-running
        message after ``stream_claim_idle_ms`` and start duplicate decoding.
        """

        await self.client.xclaim(
            message.stream,
            group,
            consumer,
            min_idle_time=0,
            message_ids=[message.event_id],
            idle=0,
            justid=True,
        )

    async def register_failure(self, message: StreamMessage) -> int:
        key = f"cv:delivery_attempts:{message.stream}:{message.event_id}"
        attempts = await self.client.incr(key)
        await self.client.expire(key, self._settings.task_ttl_seconds)
        return int(attempts)

    async def clear_failure_counter(self, message: StreamMessage) -> None:
        key = f"cv:delivery_attempts:{message.stream}:{message.event_id}"
        await self.client.delete(key)

    async def ack(self, message: StreamMessage, group: str) -> None:
        await self.client.xack(message.stream, group, message.event_id)

    async def delete(self, message: StreamMessage) -> None:
        await self.client.xdel(message.stream, message.event_id)

    async def publish_dead_letter(
        self,
        *,
        source_message: StreamMessage,
        worker: str,
        error: str,
    ) -> None:
        await self.publish(
            self._settings.stream_dead_letters,
            {
                "source_stream": source_message.stream,
                "source_event_id": source_message.event_id,
                "worker": worker,
                "error": error[:2_000],
                "payload": source_message.fields.get("payload", ""),
            },
        )

    async def wait_until_ready(self, attempts: int = 30) -> None:
        for attempt in range(attempts):
            try:
                await self.connect()
                return
            except Exception:
                if attempt == attempts - 1:
                    raise
                await asyncio.sleep(1)
