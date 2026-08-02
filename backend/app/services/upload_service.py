from __future__ import annotations

import os
import shutil
from pathlib import Path
from uuid import UUID

from fastapi import UploadFile

from app.core.config import Settings
from app.services.video_validation import (
    UploadSizeLimitError,
    validate_upload_metadata,
)


class UploadService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def save(self, task_id: UUID, file: UploadFile) -> Path:
        task_dir = self._settings.upload_dir / str(task_id)
        temporary: Path | None = None
        try:
            suffix = validate_upload_metadata(file, self._settings)
            task_dir.mkdir(parents=True, exist_ok=True)
            target = task_dir / f"source{suffix}"
            temporary = target.with_suffix(target.suffix + ".part")
            written = 0
            with temporary.open("wb") as output:
                while chunk := await file.read(1024 * 1024):
                    written += len(chunk)
                    if written > self._settings.max_upload_bytes:
                        raise UploadSizeLimitError(
                            "Uploaded video exceeds the configured size limit"
                        )
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(target)
            return target
        except Exception:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            shutil.rmtree(task_dir, ignore_errors=True)
            raise
        finally:
            await file.close()
