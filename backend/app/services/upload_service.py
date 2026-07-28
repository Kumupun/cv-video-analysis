from __future__ import annotations

import os
from pathlib import Path
from uuid import UUID

from fastapi import UploadFile

from app.core.config import Settings
from app.services.video_validation import VideoValidationError, validate_upload_metadata


class UploadService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def save(self, task_id: UUID, file: UploadFile) -> Path:
        suffix = validate_upload_metadata(file, self._settings)
        task_dir = self._settings.upload_dir / str(task_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        target = task_dir / f"source{suffix}"
        temporary = target.with_suffix(target.suffix + ".part")
        written = 0
        try:
            with temporary.open("wb") as output:
                while chunk := await file.read(1024 * 1024):
                    written += len(chunk)
                    if written > self._settings.max_upload_bytes:
                        raise VideoValidationError(
                            "Uploaded video exceeds the configured size limit"
                        )
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(target)
            return target
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        finally:
            await file.close()
