from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from fastapi import FastAPI, Response
from fastapi.testclient import TestClient

from app.api.dependencies import get_repository
from app.api.routes import router
from app.core.config import Settings
from app.core.http_cache import apply_cache_policy
from app.domain.enums import TaskStage
from app.domain.schemas import TaskStatus
from app.infrastructure.task_repository import (
    RedisTaskRepository,
    TaskStatusUnavailableError,
)


class _FlakyRedisStatusStub:
    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping
        self.calls = 0

    async def hgetall(self, key: str) -> dict[str, str]:
        del key
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary Redis read failure")
        return self.mapping


@pytest.mark.asyncio
async def test_status_read_retries_a_transient_failure() -> None:
    task_id = uuid4()
    now = datetime.now(UTC)
    status = TaskStatus(
        task_id=task_id,
        stage=TaskStage.TRACKING,
        progress=89.5,
        message="Tracking",
        created_at=now,
        updated_at=now,
    )
    redis = _FlakyRedisStatusStub(RedisTaskRepository._status_mapping(status))
    repository = RedisTaskRepository(
        redis,
        Settings(
            status_read_max_attempts=2,
            status_read_retry_delay_seconds=0,
        ),
    )

    loaded = await repository.get_status(task_id)

    assert loaded == status
    assert redis.calls == 2


class _UnavailableRepositoryStub:
    async def get_status(self, task_id: object) -> TaskStatus:
        raise TaskStatusUnavailableError(f"unavailable: {task_id}")


def test_status_endpoint_returns_retryable_503_instead_of_500() -> None:
    settings = Settings()
    app = FastAPI()
    app.include_router(router, prefix=settings.api_prefix)
    app.dependency_overrides[get_repository] = lambda: _UnavailableRepositoryStub()
    task_id = uuid4()

    with TestClient(app) as client:
        response = client.get(f"{settings.api_prefix}/status/{task_id}")

    assert response.status_code == 503
    assert response.headers["retry-after"] == "1"
    assert "temporarily unavailable" in response.json()["detail"]


def test_results_endpoint_returns_retryable_503_instead_of_500() -> None:
    settings = Settings()
    app = FastAPI()
    app.include_router(router, prefix=settings.api_prefix)
    app.dependency_overrides[get_repository] = lambda: _UnavailableRepositoryStub()
    task_id = uuid4()

    with TestClient(app) as client:
        response = client.get(f"{settings.api_prefix}/results/{task_id}")

    assert response.status_code == 503
    assert response.headers["retry-after"] == "1"


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/status/00000000-0000-0000-0000-000000000000",
        "/api/v1/results/00000000-0000-0000-0000-000000000000",
    ],
)
def test_mutable_task_responses_are_never_cached(path: str) -> None:
    response = Response()

    apply_cache_policy(path, response, api_prefix="/api/v1")

    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("path", ["/", "/index.html", "/app.js", "/styles.css"])
def test_frontend_assets_are_revalidated_after_deploy(path: str) -> None:
    response = Response()

    apply_cache_policy(path, response, api_prefix="/api/v1")

    assert response.headers["cache-control"] == "no-cache"
