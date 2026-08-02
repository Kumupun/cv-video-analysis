from __future__ import annotations

import ipaddress
import socket
from pathlib import Path
from urllib.parse import urlparse

from fastapi import UploadFile

from app.core.config import Settings


class VideoValidationError(ValueError):
    pass


class UploadSizeLimitError(VideoValidationError):
    pass


class UploadStorageLimitError(VideoValidationError):
    """The upload is valid but the server lacks safe extraction space."""

    pass


def validate_video_filename(
    filename: str,
    settings: Settings,
    content_type: str | None = None,
) -> str:
    suffix = Path(filename).suffix.lower()
    normalized_content_type = (content_type or "").lower()

    if normalized_content_type.startswith("audio/"):
        raise VideoValidationError(
            "Audio-only files are not accepted. "
            "This pipeline analyzes RGB video frames."
        )
    if suffix not in settings.allowed_video_extensions:
        raise VideoValidationError(
            f"Unsupported video extension '{suffix or '<none>'}'"
        )
    if normalized_content_type and not normalized_content_type.startswith(
        settings.allowed_video_mime_prefix
    ):
        if normalized_content_type != "application/octet-stream":
            raise VideoValidationError(
                f"Unsupported content type '{normalized_content_type}'. "
                "Expected a video file."
            )
    return suffix


def validate_upload_metadata(file: UploadFile, settings: Settings) -> str:
    return validate_video_filename(
        file.filename or "video.bin",
        settings=settings,
        content_type=file.content_type,
    )


def validate_archive_upload_metadata(file: UploadFile) -> None:
    filename = file.filename or "archive.bin"
    suffix = Path(filename).suffix.lower()
    content_type = (file.content_type or "").lower()
    if suffix != ".zip":
        raise VideoValidationError("Only ZIP archives are accepted by this endpoint")
    allowed_types = {
        "",
        "application/octet-stream",
        "application/x-zip-compressed",
        "application/zip",
    }
    if content_type not in allowed_types:
        raise VideoValidationError(
            f"Unsupported content type '{content_type}'. Expected a ZIP archive."
        )


def validate_remote_url(url: str, settings: Settings) -> None:
    if not settings.allow_remote_urls:
        raise VideoValidationError(
            "Remote video URLs are disabled. Set ALLOW_REMOTE_URLS=true to enable them."
        )
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise VideoValidationError("Only absolute HTTP(S) video URLs are supported")
    try:
        addresses = socket.getaddrinfo(
            parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
        )
    except socket.gaierror as exc:
        raise VideoValidationError("The remote host could not be resolved") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if any(
            (
                ip.is_private,
                ip.is_loopback,
                ip.is_link_local,
                ip.is_multicast,
                ip.is_reserved,
                ip.is_unspecified,
            )
        ):
            raise VideoValidationError(
                "Remote URLs resolving to private or reserved networks are forbidden"
            )
