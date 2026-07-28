from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.api.health import router as health_router
from app.api.routes import router as api_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.metrics import API_REQUESTS
from app.infrastructure.ray_store import RayObjectStore
from app.infrastructure.redis_streams import RedisStreams
from app.infrastructure.task_repository import RedisTaskRepository
from app.services.upload_service import UploadService

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    streams = RedisStreams(settings)
    await streams.wait_until_ready()
    app.state.settings = settings
    app.state.streams = streams
    app.state.repository = RedisTaskRepository(streams.client, settings)
    app.state.upload_service = UploadService(settings)
    app.state.object_store = RayObjectStore(settings)
    logger.info("API dependencies initialized")
    try:
        yield
    finally:
        await streams.close()
        logger.info("API dependencies closed")


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.middleware("http")
async def request_metrics(request: Request, call_next):
    response: Response
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        API_REQUESTS.labels(request.method, request.url.path, "500").inc()
        raise
    API_REQUESTS.labels(
        request.method, request.url.path, str(response.status_code)
    ).inc()
    response.headers["X-Process-Time"] = f"{time.perf_counter() - started:.6f}"
    return response


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


app.include_router(health_router)
app.include_router(api_router, prefix=settings.api_prefix)
