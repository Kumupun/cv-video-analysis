from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from fastapi import APIRouter, Request

from app.domain.schemas import HealthResponse
from app.infrastructure.ray_store import RayObjectStore

router = APIRouter()


@router.get("/health/live", response_model=HealthResponse)
async def live() -> HealthResponse:
    return HealthResponse(status="ok", dependencies={"process": "ok"})


@router.get("/health/ready", response_model=HealthResponse)
async def ready(request: Request) -> HealthResponse:
    dependencies: dict[str, str] = {}
    try:
        await request.app.state.streams.client.ping()
        dependencies["redis"] = "ok"
    except Exception:
        dependencies["redis"] = "unavailable"

    store: RayObjectStore = request.app.state.object_store
    ray_ready = await asyncio.to_thread(store.ping)
    dependencies["ray"] = "ok" if ray_ready else "unavailable"
    all_ready = all(value == "ok" for value in dependencies.values())
    overall = "ok" if all_ready else "degraded"
    return HealthResponse(
        status=overall,
        dependencies=dependencies,
        timestamp=datetime.now(UTC),
    )
