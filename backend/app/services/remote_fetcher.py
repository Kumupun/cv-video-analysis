from __future__ import annotations

from pathlib import Path
from urllib.parse import urljoin
from uuid import UUID

from app.core.config import Settings
from app.services.video_validation import VideoValidationError, validate_remote_url


class RemoteVideoFetcher:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def fetch(self, task_id: UUID, url: str) -> Path:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("httpx is required for remote URL ingestion") from exc

        task_dir = self._settings.upload_dir / str(task_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        temporary = task_dir / "remote-video.part"
        timeout = httpx.Timeout(self._settings.remote_download_timeout_seconds)
        redirect_codes = {301, 302, 303, 307, 308}
        current_url = url

        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=False,
            ) as client:
                for redirect_index in range(
                    self._settings.remote_download_max_redirects + 1
                ):
                    validate_remote_url(current_url, self._settings)
                    async with client.stream("GET", current_url) as response:
                        if response.status_code in redirect_codes:
                            location = response.headers.get("location")
                            if not location:
                                raise VideoValidationError(
                                    "Remote server returned an invalid redirect"
                                )
                            if (
                                redirect_index
                                >= self._settings.remote_download_max_redirects
                            ):
                                raise VideoValidationError(
                                    "Remote video exceeded the redirect limit"
                                )
                            current_url = urljoin(current_url, location)
                            continue

                        response.raise_for_status()
                        content_type = response.headers.get("content-type", "").lower()
                        if content_type.startswith("audio/"):
                            raise VideoValidationError(
                                "The remote resource is audio-only; "
                                "RGB video is required"
                            )
                        if content_type and not content_type.startswith("video/"):
                            raise VideoValidationError(
                                f"Remote resource is not a video ({content_type})"
                            )

                        content_length = response.headers.get("content-length")
                        if (
                            content_length
                            and int(content_length) > self._settings.max_upload_bytes
                        ):
                            raise VideoValidationError(
                                "Remote video exceeds the configured size limit"
                            )

                        suffix = self._suffix_from_content_type(content_type)
                        target = task_dir / f"source{suffix}"
                        written = 0
                        with temporary.open("wb") as output:
                            async for chunk in response.aiter_bytes(1024 * 1024):
                                written += len(chunk)
                                if written > self._settings.max_upload_bytes:
                                    raise VideoValidationError(
                                        "Remote video exceeds the configured size limit"
                                    )
                                output.write(chunk)
                        temporary.replace(target)
                        return target
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

        raise VideoValidationError("Remote video could not be downloaded")

    @staticmethod
    def _suffix_from_content_type(content_type: str) -> str:
        mapping = {
            "video/mp4": ".mp4",
            "video/quicktime": ".mov",
            "video/x-matroska": ".mkv",
            "video/webm": ".webm",
            "video/x-msvideo": ".avi",
        }
        return mapping.get(content_type.split(";", 1)[0], ".mp4")
