from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
from collections.abc import Awaitable, Callable

from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.core.metrics import CHUNK_PROCESS_SECONDS, WORKER_ERRORS, WORKER_UP
from app.infrastructure.redis_streams import RedisStreams, StreamMessage

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
    ) -> None:
        self.worker_name = worker_name
        self.stream = stream
        self.group = group
        self.handler = handler
        self.settings = settings
        self.streams = streams
        self.consumer = f"{socket.gethostname()}-{os.getpid()}"

    async def run_forever(self) -> None:
        await self.streams.wait_until_ready()
        await self.streams.ensure_group(self.stream, self.group)
        logger.info(
            "Worker started",
            extra={"stream": self.stream, "stage": self.worker_name},
        )
        while True:
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
                )
            if not messages:
                continue
            for message in messages:
                started = time.perf_counter()
                try:
                    await self.handler(message)
                    await self.streams.ack(message, self.group)
                    await self.streams.clear_failure_counter(message)
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
                except Exception as exc:
                    WORKER_ERRORS.labels(self.worker_name).inc()
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
                        await self.streams.ack(message, self.group)
                        await self.streams.clear_failure_counter(message)
                    else:
                        logger.warning(
                            "Message left pending for retry",
                            extra={
                                "stream": message.stream,
                                "event_id": message.event_id,
                                "stage": self.worker_name,
                            },
                        )
                finally:
                    CHUNK_PROCESS_SECONDS.labels(self.worker_name).observe(
                        time.perf_counter() - started
                    )


async def run_worker(factory: Callable[[Settings, RedisStreams], StreamWorker]) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    from prometheus_client import start_http_server

    start_http_server(settings.worker_metrics_port)
    streams = RedisStreams(settings)
    worker: StreamWorker | None = None
    try:
        # Handlers access streams.client during construction, so Redis must be
        # connected before the worker factory is called.
        await streams.wait_until_ready()
        worker = factory(settings, streams)
        WORKER_UP.labels(worker.worker_name).set(1)
        await worker.run_forever()
    finally:
        if worker is not None:
            WORKER_UP.labels(worker.worker_name).set(0)
        await streams.close()
