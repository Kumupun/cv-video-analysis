from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import get_repository, get_upload_service
from app.api.routes import router
from app.core.config import Settings, get_settings
from app.domain.schemas import SUPPORTED_TRACKING_CLASSES


class _UploadServiceStub:
    def __init__(self, saved_path: Path) -> None:
        self.saved_path = saved_path

    async def save(self, task_id: object, file: object) -> Path:
        del task_id, file
        return self.saved_path


class _RepositoryStub:
    def __init__(self) -> None:
        self.source: Any = None
        self.stream_fields: dict[str, str] | None = None

    async def create_and_enqueue(
        self,
        task_id: object,
        source: object,
        stream: str,
        stream_fields: dict[str, str],
    ) -> None:
        del task_id, stream
        self.source = source
        self.stream_fields = stream_fields


def _client(tmp_path: Path) -> tuple[TestClient, _RepositoryStub]:
    settings = Settings(
        _env_file=None,
        upload_dir=tmp_path / "uploads",
        result_dir=tmp_path / "results",
    )
    repository = _RepositoryStub()
    test_app = FastAPI()
    test_app.include_router(router, prefix=settings.api_prefix)
    test_app.dependency_overrides[get_repository] = lambda: repository
    test_app.dependency_overrides[get_upload_service] = lambda: _UploadServiceStub(
        tmp_path / "video.mp4"
    )
    test_app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(test_app), repository


def test_process_accepts_frontend_class_json_and_persists_it(tmp_path: Path) -> None:
    client, repository = _client(tmp_path)

    with client:
        response = client.post(
            "/api/v1/process",
            files={"file": ("video.mp4", b"video", "video/mp4")},
            data={"classes": '["person", "car"]'},
        )

    assert response.status_code == 202
    assert repository.source.tracking_classes == ("person", "car")
    assert response.json()["status_url"].startswith("/api/v1/status/")
    assert response.json()["results_url"].startswith("/api/v1/results/")


def test_process_rejects_unknown_frontend_class(tmp_path: Path) -> None:
    client, repository = _client(tmp_path)

    with client:
        response = client.post(
            "/api/v1/process",
            files={"file": ("video.mp4", b"video", "video/mp4")},
            data={"classes": '["not-a-real-class"]'},
        )

    assert response.status_code == 422
    assert response.json()["detail"]["unknown"] == ["not-a-real-class"]
    assert repository.source is None


def test_process_accepts_empty_class_selection_for_cuts_only(tmp_path: Path) -> None:
    client, repository = _client(tmp_path)

    with client:
        response = client.post(
            "/api/v1/process",
            files={"file": ("video.mp4", b"video", "video/mp4")},
            data={"classes": "[]"},
        )

    assert response.status_code == 202
    assert repository.source.tracking_classes == ()


def test_frontend_exposes_archive_upload_and_all_tracking_classes() -> None:
    project_root = Path(__file__).resolve().parents[2]
    script = (project_root / "frontend" / "app.js").read_text(encoding="utf-8")
    markup = (project_root / "frontend" / "index.html").read_text(encoding="utf-8")
    class_block = script.split("const CLASS_GROUPS = [", 1)[1].split(
        "const CLASS_OPTIONS", 1
    )[0]
    groups = re.findall(r"classes:\s*\[(.*?)\]\s*}", class_block, re.DOTALL)
    frontend_classes = tuple(
        class_name for group in groups for class_name in re.findall(r"'([^']+)'", group)
    )

    assert frontend_classes == SUPPORTED_TRACKING_CLASSES
    assert "`${API_BASE}/process/archive`" in script
    assert 'accept="video/*,.mkv,.zip,application/zip"' in markup
    assert "cv-video-analysis:active-session:v1" in script
    assert "cv-video-analysis:selected-classes:v1" in script
    assert "restoreActiveSession();" in script
    assert "pollStatus(session.taskId)" in script
    assert "pollArchiveStatuses(session.tasks, token)" in script
    assert 'id="video-restore-note"' in markup
    assert 'id="reattach-video-btn"' in markup
    assert "attachVideoForPlayback(f)" in script
    assert "cache: 'no-store'" in script
    assert "Number(progress)" in script
    assert 'id="status-detail"' in markup
    assert "app.js?v=20260802-1" in markup
    assert "gradual_transition" in markup
    assert "t.start_timestamp" in script
    assert "t.end_timestamp" in script
    assert "Math.max(1, x2-x1)" in script
    assert 'id="archive-back-btn"' in markup
    assert 'id="video-restore-text"' in markup
    assert "openArchiveResult(outcome)" in script
    assert "Відкрити результат" in script
    assert "Повторна обробка не запускається" in script
