from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable

from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.core.metrics import CHUNK_PROCESS_SECONDS, WORKER_ERRORS, WORKER_UP
from app.infrastructure.redis_streams import RedisStreams, StreamMessage
from app.infrastructure.task_repository import (
    RedisTaskRepository,
    TaskNotFoundError,
)

logger = logging.getLogger(__name__)


class TerminalWorkerError(RuntimeError):
    """The task was already marked failed and must not be retried."""


class StreamWorker:
    def __init__(
        self,
        *,
        worker_name: str,
        stream: str,
        group: str,
        handler: Callable[[StreamMessage], Awaitable[None]],
        settings: Settings,
        streams: RedisStreams,
        max_concurrency: int = 1,
        partition_by_task: bool = False,
    ) -> None:
        self.worker_name = worker_name
        self.stream = stream
        self.group = group
        self.handler = handler
        self.settings = settings
        self.streams = streams
        self.repository = RedisTaskRepository(streams.client, settings)
        self.consumer = f"{socket.gethostname()}-{os.getpid()}"
        self.max_concurrency = max(1, max_concurrency)
        self.partition_by_task = partition_by_task

    async def run_forever(self) -> None:
        await self.streams.wait_until_ready()
        await self.streams.ensure_group(self.stream, self.group)
        logger.info(
            "Worker started",
            extra={
                "stream": self.stream,
                "stage": self.worker_name,
                "concurrency": self.max_concurrency,
                "partition_by_task": self.partition_by_task,
            },
        )
        while True:
            try:
                messages = await self.streams.claim_stale(
                    stream=self.stream,
                    group=self.group,
                    consumer=self.consumer,
                )
                if not messages:
                    messages = await self.streams.read_group(
                        stream=self.stream,
                        group=self.group,
                        consumer=self.consumer,
                        count=max(
                            self.settings.stream_batch_size,
                            self.max_concurrency,
                        ),
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "Redis stream polling failed; retrying without exiting",
                    extra={
                        "stream": self.stream,
                        "stage": self.worker_name,
                    },
                    exc_info=True,
                )
                await asyncio.sleep(max(self.settings.stream_retry_delay_seconds, 0.1))
                continue

            if not messages:
                continue
            await self._process_batch(messages)

    async def _process_batch(self, messages: list[StreamMessage]) -> None:
        if self.max_concurrency == 1 or len(messages) == 1:
            for message in messages:
                await self._process_message(message)
            return

        partitions = self._partition_messages(messages)
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def process_partition(items: list[StreamMessage]) -> None:
            async with semaphore:
                for item in items:
                    # Messages for one task remain strictly ordered. Different
                    # task IDs may progress concurrently in the same process.
                    await self._process_message(item)

        await asyncio.gather(
            *(process_partition(items) for items in partitions.values())
        )

    def _partition_messages(
        self,
        messages: list[StreamMessage],
    ) -> OrderedDict[str, list[StreamMessage]]:
        partitions: OrderedDict[str, list[StreamMessage]] = OrderedDict()
        for message in messages:
            if self.partition_by_task:
                key = _extract_task_id(message) or f"event:{message.event_id}"
            else:
                key = f"event:{message.event_id}"
            partitions.setdefault(key, []).append(message)
        return partitions

    async def _process_message(self, message: StreamMessage) -> None:
        """Process one stream entry with retries and a pending-entry heartbeat."""

        heartbeat = asyncio.create_task(self._heartbeat_pending(message))
        try:
            while True:
                started = time.perf_counter()
                retry = False
                try:
                    await self.handler(message)
                    await self.streams.ack(message, self.group)
                    await self.streams.clear_failure_counter(message)
                    return
                except asyncio.CancelledError:
                    raise
                except TerminalWorkerError as exc:
                    WORKER_ERRORS.labels(self.worker_name).inc()
                    logger.exception(
                        "Terminal worker message failure",
                        extra={
                            "stream": message.stream,
                            "event_id": message.event_id,
                            "stage": self.worker_name,
                        },
                    )
                    await self.streams.publish_dead_letter(
                        source_message=message,
                        worker=self.worker_name,
                        error=str(exc),
                    )
                    await self.streams.ack(message, self.group)
                    await self.streams.clear_failure_counter(message)
                    return
                except Exception as exc:
                    WORKER_ERRORS.labels(self.worker_name).inc()
                    if RedisStreams.is_transient_error(exc):
                        logger.warning(
                            "Transient Redis failure while processing a message; "
                            "leaving it pending and retrying",
                            extra={
                                "stream": message.stream,
                                "event_id": message.event_id,
                                "stage": self.worker_name,
                            },
                            exc_info=True,
                        )
                        retry = True
                    else:
                        logger.exception(
                            "Worker message failed",
                            extra={
                                "stream": message.stream,
                                "event_id": message.event_id,
                                "stage": self.worker_name,
                            },
                        )
                        attempts = await self.streams.register_failure(message)
                        if attempts >= self.settings.stream_max_delivery_attempts:
                            await self.streams.publish_dead_letter(
                                source_message=message,
                                worker=self.worker_name,
                                error=str(exc),
                            )
                            await self._mark_task_failed(message, exc)
                            await self.streams.ack(message, self.group)
                            await self.streams.clear_failure_counter(message)
                            return

                        logger.warning(
                            "Retrying the same message before later stream entries",
                            extra={
                                "stream": message.stream,
                                "event_id": message.event_id,
                                "stage": self.worker_name,
                                "attempt": attempts,
                                "max_attempts": (
                                    self.settings.stream_max_delivery_attempts
                                ),
                            },
                        )
                        retry = True
                finally:
                    CHUNK_PROCESS_SECONDS.labels(self.worker_name).observe(
                        time.perf_counter() - started
                    )

                if retry:
                    await asyncio.sleep(self.settings.stream_retry_delay_seconds)
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)

    async def _heartbeat_pending(self, message: StreamMessage) -> None:
        interval = getattr(
            self.settings,
            "stream_processing_heartbeat_seconds",
            15.0,
        )
        while True:
            await asyncio.sleep(interval)
            try:
                await self.streams.touch_pending(
                    message,
                    group=self.group,
                    consumer=self.consumer,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "Could not refresh the Redis pending-entry heartbeat",
                    extra={
                        "stream": message.stream,
                        "event_id": message.event_id,
                        "stage": self.worker_name,
                    },
                    exc_info=True,
                )

    async def _mark_task_failed(
        self,
        message: StreamMessage,
        exc: Exception,
    ) -> None:
        task_id = _extract_task_id(message)
        if task_id is None:
            return
        try:
            await self.repository.fail(
                task_id,
                code=f"{self.worker_name.replace('-', '_')}_failed",
                detail=str(exc),
            )
        except TaskNotFoundError:
            logger.warning(
                "Cannot mark a dead-letter message task as failed",
                extra={
                    "task_id": task_id,
                    "stream": message.stream,
                    "event_id": message.event_id,
                },
            )
        except Exception:
            logger.exception(
                "Failed to persist terminal task state",
                extra={
                    "task_id": task_id,
                    "stream": message.stream,
                    "event_id": message.event_id,
                },
            )


def _extract_task_id(message: StreamMessage) -> str | None:
    direct = message.fields.get("task_id")
    if direct:
        return direct

    payload = message.fields.get("payload")
    if not payload:
        return None
    try:
        parsed = json.loads(payload)
    except (TypeError, json.JSONDecodeError):
        return None
    task_id = parsed.get("task_id") if isinstance(parsed, dict) else None
    return str(task_id) if task_id else None


async def run_worker(factory: Callable[[Settings, RedisStreams], StreamWorker]) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    from prometheus_client import start_http_server

    start_http_server(settings.worker_metrics_port)
    streams = RedisStreams(settings)
    worker: StreamWorker | None = None
    try:
        await streams.wait_until_ready()
        worker = factory(settings, streams)
        WORKER_UP.labels(worker.worker_name).set(1)
        await worker.run_forever()
    finally:
        if worker is not None:
            WORKER_UP.labels(worker.worker_name).set(0)
        await streams.close()
