from pathlib import Path
from uuid import uuid4

import pytest

from app.core.config import Settings
from app.workers.ingest import IngestHandler


@pytest.mark.asyncio
async def test_completed_ingest_removes_only_its_task_directory(
    tmp_path: Path,
) -> None:
    task_id = uuid4()
    task_dir = tmp_path / str(task_id)
    task_dir.mkdir()
    video = task_dir / "source.mp4"
    video.write_bytes(b"video")
    sibling = tmp_path / "keep.txt"
    sibling.write_text("keep", encoding="utf-8")

    handler = IngestHandler.__new__(IngestHandler)
    handler.settings = Settings(_env_file=None, upload_dir=tmp_path)

    await handler._cleanup_processed_source(task_id, video)

    assert not task_dir.exists()
    assert sibling.read_text(encoding="utf-8") == "keep"


@pytest.mark.asyncio
async def test_cleanup_refuses_path_outside_upload_root(tmp_path: Path) -> None:
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    video = external_dir / "source.mp4"
    video.write_bytes(b"video")

    handler = IngestHandler.__new__(IngestHandler)
    handler.settings = Settings(_env_file=None, upload_dir=upload_root)

    await handler._cleanup_processed_source(uuid4(), video)

    assert video.exists()
