from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.datastructures import Headers, UploadFile

from app.api.dependencies import get_archive_upload_service, get_repository
from app.api.routes import router
from app.core.config import Settings, get_settings
from app.services.archive_upload_service import ArchiveUploadService
from app.services.video_validation import (
    UploadSizeLimitError,
    UploadStorageLimitError,
    VideoValidationError,
)


def _zip_bytes(entries: list[tuple[str, bytes]]) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries:
            archive.writestr(name, content)
    return buffer.getvalue()


def _upload(data: bytes, filename: str = "videos.zip") -> UploadFile:
    return UploadFile(
        file=BytesIO(data),
        filename=filename,
        headers=Headers({"content-type": "application/zip"}),
    )


@pytest.mark.asyncio
async def test_archive_extracts_supported_videos_and_reports_skipped_files(
    tmp_path: Path,
) -> None:
    settings = Settings(upload_dir=tmp_path)
    service = ArchiveUploadService(settings)
    result = await service.extract(
        _upload(
            _zip_bytes(
                [
                    ("camera/front.mp4", b"first-video"),
                    ("camera/rear.MKV", b"second-video"),
                    ("notes/readme.txt", b"not a video"),
                ]
            )
        )
    )

    assert [video.original_filename for video in result.videos] == [
        "camera/front.mp4",
        "camera/rear.MKV",
    ]
    assert [video.path.read_bytes() for video in result.videos] == [
        b"first-video",
        b"second-video",
    ]
    assert result.skipped[0].filename == "notes/readme.txt"
    assert "Unsupported video extension" in result.skipped[0].reason

    await service.cleanup(result)
    assert all(not video.path.parent.exists() for video in result.videos)


@pytest.mark.asyncio
async def test_archive_rejects_path_traversal(tmp_path: Path) -> None:
    service = ArchiveUploadService(Settings(upload_dir=tmp_path))

    with pytest.raises(VideoValidationError, match="Unsafe path"):
        await service.extract(_upload(_zip_bytes([("../escape.mp4", b"video")])))

    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_archive_rejects_suspicious_compression_ratio(tmp_path: Path) -> None:
    settings = Settings(
        upload_dir=tmp_path,
        max_archive_compression_ratio=2.0,
    )
    service = ArchiveUploadService(settings)

    with pytest.raises(UploadSizeLimitError, match="compression ratio"):
        await service.extract(_upload(_zip_bytes([("compressed.mp4", b"0" * 50_000)])))

    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_archive_accepts_many_videos_when_combined_size_fits(
    tmp_path: Path,
) -> None:
    settings = Settings(
        upload_dir=tmp_path,
        max_archive_video_bytes=6,
        archive_disk_reserve_bytes=0,
    )
    service = ArchiveUploadService(settings)

    result = await service.extract(
        _upload(
            _zip_bytes(
                [
                    ("one.mp4", b"aa"),
                    ("two.mp4", b"bb"),
                    ("three.mp4", b"cc"),
                ]
            )
        )
    )

    assert len(result.videos) == 3
    assert result.accepted_size_bytes == 6
    await service.cleanup(result)


@pytest.mark.asyncio
async def test_archive_rejects_combined_video_size_over_budget(tmp_path: Path) -> None:
    settings = Settings(
        upload_dir=tmp_path,
        max_archive_video_bytes=5,
        archive_disk_reserve_bytes=0,
    )
    service = ArchiveUploadService(settings)

    with pytest.raises(UploadSizeLimitError, match="Combined size"):
        await service.extract(
            _upload(
                _zip_bytes(
                    [
                        ("one.mp4", b"aaa"),
                        ("two.mp4", b"bbb"),
                    ]
                )
            )
        )

    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_archive_rejects_when_safe_extraction_space_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        upload_dir=tmp_path,
        archive_disk_reserve_bytes=2,
    )
    service = ArchiveUploadService(settings)
    monkeypatch.setattr(
        "app.services.archive_upload_service.shutil.disk_usage",
        lambda _: SimpleNamespace(free=4),
    )

    with pytest.raises(UploadStorageLimitError, match="free disk space"):
        await service.extract(_upload(_zip_bytes([("one.mp4", b"abc")])))

    assert list(tmp_path.iterdir()) == []


class _RepositoryStub:
    def __init__(self) -> None:
        self.queued: list[tuple[Any, Any, Any]] = []
        self.stream: str | None = None

    async def create_many_and_enqueue(
        self,
        tasks: list[tuple[Any, Any, Any]],
        stream: str,
    ) -> list[Any]:
        self.queued = tasks
        self.stream = stream
        return []


def test_archive_endpoint_creates_one_task_per_video(tmp_path: Path) -> None:
    settings = Settings(upload_dir=tmp_path)
    service = ArchiveUploadService(settings)
    repository = _RepositoryStub()
    app = FastAPI()
    app.include_router(router, prefix=settings.api_prefix)
    app.dependency_overrides[get_archive_upload_service] = lambda: service
    app.dependency_overrides[get_repository] = lambda: repository
    app.dependency_overrides[get_settings] = lambda: settings

    with TestClient(app) as client:
        response = client.post(
            f"{settings.api_prefix}/process/archive",
            files={
                "file": (
                    "batch.zip",
                    _zip_bytes(
                        [
                            ("one.mp4", b"one-video"),
                            ("nested/two.mov", b"two-video"),
                            ("metadata.json", b"{}"),
                        ]
                    ),
                    "application/zip",
                )
            },
            data={"classes": '["person", "car"]'},
        )

    assert response.status_code == 202
    payload = response.json()
    assert payload["archive_filename"] == "batch.zip"
    assert payload["archive_size_bytes"] > 0
    assert payload["accepted_size_bytes"] == len(b"one-video") + len(b"two-video")
    assert payload["accepted_count"] == 2
    assert payload["skipped_count"] == 1
    assert {task["filename"] for task in payload["tasks"]} == {
        "one.mp4",
        "nested/two.mov",
    }
    assert len(repository.queued) == 2
    assert all(
        source.tracking_classes == ("person", "car")
        for _, source, _ in repository.queued
    )
    assert repository.stream == settings.stream_ingest_jobs


class _FailingRepositoryStub:
    async def create_many_and_enqueue(
        self,
        tasks: list[tuple[Any, Any, Any]],
        stream: str,
    ) -> list[Any]:
        raise RuntimeError("redis unavailable")


def test_archive_endpoint_cleans_files_when_queue_fails(tmp_path: Path) -> None:
    settings = Settings(upload_dir=tmp_path)
    service = ArchiveUploadService(settings)
    app = FastAPI()
    app.include_router(router, prefix=settings.api_prefix)
    app.dependency_overrides[get_archive_upload_service] = lambda: service
    app.dependency_overrides[get_repository] = lambda: _FailingRepositoryStub()
    app.dependency_overrides[get_settings] = lambda: settings

    with TestClient(app) as client:
        response = client.post(
            f"{settings.api_prefix}/process/archive",
            files={
                "file": (
                    "batch.zip",
                    _zip_bytes([("one.mp4", b"one-video")]),
                    "application/zip",
                )
            },
        )

    assert response.status_code == 503
    assert list(tmp_path.iterdir()) == []
