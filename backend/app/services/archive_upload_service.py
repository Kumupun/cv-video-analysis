from __future__ import annotations

import mimetypes
import os
import shutil
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from uuid import UUID, uuid4

from fastapi import UploadFile
from starlette.concurrency import run_in_threadpool

from app.core.config import Settings
from app.services.video_validation import (
    UploadSizeLimitError,
    UploadStorageLimitError,
    VideoValidationError,
    validate_archive_upload_metadata,
    validate_video_filename,
)

_COPY_BUFFER_BYTES = 1024 * 1024
_ZIP_ENCRYPTED_FLAG = 0x1


def _format_gib(value: int) -> str:
    return f"{value / (1024**3):g} GiB"


@dataclass(frozen=True, slots=True)
class ExtractedArchiveVideo:
    task_id: UUID
    path: Path
    original_filename: str
    content_type: str | None
    size_bytes: int


@dataclass(frozen=True, slots=True)
class SkippedArchiveEntry:
    filename: str
    reason: str


@dataclass(frozen=True, slots=True)
class ArchiveExtractionResult:
    archive_filename: str
    archive_size_bytes: int
    accepted_size_bytes: int
    videos: tuple[ExtractedArchiveVideo, ...]
    skipped: tuple[SkippedArchiveEntry, ...]


@dataclass(frozen=True, slots=True)
class _ArchiveVideoCandidate:
    member: zipfile.ZipInfo
    normalized_name: str
    suffix: str


class ArchiveUploadService:
    """Persist and safely unpack ZIP uploads into independent video tasks.

    Admission is intentionally size-based: supported videos may be numerous or
    few, but their combined declared and extracted payload must fit within the
    configured archive video-byte budget.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def extract(self, file: UploadFile) -> ArchiveExtractionResult:
        archive_filename = file.filename or "videos.zip"
        try:
            validate_archive_upload_metadata(file)
        except VideoValidationError:
            await file.close()
            raise
        archive_path = await self._save_archive_to_temporary_file(file)
        archive_size_bytes = archive_path.stat().st_size
        created_task_dirs: list[Path] = []
        try:
            result = await run_in_threadpool(
                self._extract_archive,
                archive_path,
                archive_filename=archive_filename,
                archive_size_bytes=archive_size_bytes,
                created_task_dirs=created_task_dirs,
            )
            if not result.videos:
                raise VideoValidationError(
                    "The ZIP archive does not contain any supported video files"
                )
            return result
        except Exception:
            await run_in_threadpool(self._remove_directories, created_task_dirs)
            raise
        finally:
            archive_path.unlink(missing_ok=True)

    async def cleanup(self, result: ArchiveExtractionResult) -> None:
        task_dirs = [video.path.parent for video in result.videos]
        await run_in_threadpool(self._remove_directories, task_dirs)

    async def _save_archive_to_temporary_file(self, file: UploadFile) -> Path:
        self._settings.upload_dir.mkdir(parents=True, exist_ok=True)
        descriptor, raw_path = tempfile.mkstemp(
            prefix="archive-",
            suffix=".zip.part",
            dir=self._settings.upload_dir,
        )
        path = Path(raw_path)
        written = 0
        try:
            with os.fdopen(descriptor, "wb") as output:
                while chunk := await file.read(_COPY_BUFFER_BYTES):
                    written += len(chunk)
                    if written > self._settings.max_archive_upload_bytes:
                        limit = _format_gib(self._settings.max_archive_upload_bytes)
                        raise UploadSizeLimitError(
                            "Uploaded ZIP archive exceeds the configured "
                            f"compressed-size limit ({limit})"
                        )
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            if written == 0:
                raise VideoValidationError("Uploaded ZIP archive is empty")
            return path
        except Exception:
            path.unlink(missing_ok=True)
            raise
        finally:
            await file.close()

    def _extract_archive(
        self,
        archive_path: Path,
        *,
        archive_filename: str,
        archive_size_bytes: int,
        created_task_dirs: list[Path],
    ) -> ArchiveExtractionResult:
        if not zipfile.is_zipfile(archive_path):
            raise VideoValidationError("Uploaded file is not a valid ZIP archive")

        try:
            with zipfile.ZipFile(archive_path) as archive:
                members = archive.infolist()
                self._validate_archive_structure(members)
                candidates, skipped, accepted_size_bytes = self._collect_candidates(
                    members
                )
                self._validate_available_disk_space(
                    accepted_size_bytes,
                    archive_size_bytes,
                )

                videos = [
                    self._extract_video_member(
                        archive,
                        candidate.member,
                        normalized_name=candidate.normalized_name,
                        suffix=candidate.suffix,
                        created_task_dirs=created_task_dirs,
                    )
                    for candidate in candidates
                ]
        except zipfile.BadZipFile as exc:
            raise VideoValidationError("ZIP archive is corrupted") from exc

        extracted_size_bytes = sum(video.size_bytes for video in videos)
        if extracted_size_bytes != accepted_size_bytes:
            raise VideoValidationError(
                "Extracted video size does not match ZIP metadata"
            )

        return ArchiveExtractionResult(
            archive_filename=archive_filename,
            archive_size_bytes=archive_size_bytes,
            accepted_size_bytes=accepted_size_bytes,
            videos=tuple(videos),
            skipped=tuple(skipped),
        )

    def _collect_candidates(
        self,
        members: list[zipfile.ZipInfo],
    ) -> tuple[
        list[_ArchiveVideoCandidate],
        list[SkippedArchiveEntry],
        int,
    ]:
        candidates: list[_ArchiveVideoCandidate] = []
        skipped: list[SkippedArchiveEntry] = []
        seen_names: set[str] = set()
        accepted_size_bytes = 0

        for member in members:
            if member.is_dir():
                continue
            normalized_name = self._validate_member_path(member)
            if self._is_ignored_system_entry(normalized_name):
                skipped.append(
                    SkippedArchiveEntry(
                        filename=normalized_name,
                        reason="system metadata file",
                    )
                )
                continue
            if normalized_name in seen_names:
                skipped.append(
                    SkippedArchiveEntry(
                        filename=normalized_name,
                        reason="duplicate archive entry",
                    )
                )
                continue
            seen_names.add(normalized_name)

            try:
                suffix = validate_video_filename(
                    normalized_name,
                    settings=self._settings,
                )
            except VideoValidationError as exc:
                skipped.append(
                    SkippedArchiveEntry(
                        filename=normalized_name,
                        reason=str(exc),
                    )
                )
                continue

            self._validate_video_member_limits(member)
            accepted_size_bytes += member.file_size
            if accepted_size_bytes > self._settings.max_archive_video_bytes:
                limit = _format_gib(self._settings.max_archive_video_bytes)
                raise UploadSizeLimitError(
                    "Combined size of supported videos in the ZIP exceeds the "
                    f"configured batch limit ({limit})"
                )
            candidates.append(
                _ArchiveVideoCandidate(
                    member=member,
                    normalized_name=normalized_name,
                    suffix=suffix,
                )
            )

        return candidates, skipped, accepted_size_bytes

    def _validate_archive_structure(self, members: list[zipfile.ZipInfo]) -> None:
        if len(members) > self._settings.max_archive_members:
            raise UploadSizeLimitError(
                "ZIP archive contains more entries than the safety limit"
            )

        total_uncompressed = 0
        for member in members:
            self._validate_member_path(member)
            if member.is_dir():
                continue
            if member.flag_bits & _ZIP_ENCRYPTED_FLAG:
                raise VideoValidationError(
                    f"Encrypted ZIP entry is not supported: {member.filename}"
                )
            if self._is_symlink(member):
                raise VideoValidationError(
                    f"Symbolic links are not allowed in ZIP archives: {member.filename}"
                )
            total_uncompressed += member.file_size
            if total_uncompressed > self._settings.max_archive_uncompressed_bytes:
                raise UploadSizeLimitError(
                    "ZIP archive exceeds the configured total uncompressed-size limit"
                )

    def _validate_video_member_limits(self, member: zipfile.ZipInfo) -> None:
        if member.file_size <= 0:
            raise VideoValidationError(f"Video entry is empty: {member.filename}")
        if member.file_size > self._settings.max_upload_bytes:
            raise UploadSizeLimitError(
                f"Video entry exceeds the per-video size limit: {member.filename}"
            )
        ratio = member.file_size / max(member.compress_size, 1)
        if ratio > self._settings.max_archive_compression_ratio:
            raise UploadSizeLimitError(
                f"Suspicious ZIP compression ratio for entry: {member.filename}"
            )

    def _validate_available_disk_space(
        self,
        accepted_size_bytes: int,
        archive_size_bytes: int,
    ) -> None:
        available = shutil.disk_usage(self._settings.upload_dir).free
        # The compressed temporary ZIP remains on disk while accepted members
        # are extracted, so both copies must fit at the same time.
        required = (
            accepted_size_bytes
            + archive_size_bytes
            + self._settings.archive_disk_reserve_bytes
        )
        if available < required:
            raise UploadStorageLimitError(
                "Not enough free disk space to extract the accepted videos safely"
            )

    def _extract_video_member(
        self,
        archive: zipfile.ZipFile,
        member: zipfile.ZipInfo,
        *,
        normalized_name: str,
        suffix: str,
        created_task_dirs: list[Path],
    ) -> ExtractedArchiveVideo:
        task_id = uuid4()
        task_dir = self._settings.upload_dir / str(task_id)
        task_dir.mkdir(parents=True, exist_ok=False)
        created_task_dirs.append(task_dir)
        target = task_dir / f"source{suffix}"
        temporary = target.with_suffix(target.suffix + ".part")
        written = 0

        try:
            with archive.open(member, "r") as source, temporary.open("wb") as output:
                while chunk := source.read(_COPY_BUFFER_BYTES):
                    written += len(chunk)
                    if written > self._settings.max_upload_bytes:
                        detail = (
                            "Video entry exceeds the per-video size limit: "
                            f"{normalized_name}"
                        )
                        raise UploadSizeLimitError(detail)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            if written != member.file_size:
                raise VideoValidationError(
                    f"ZIP entry size does not match its metadata: {normalized_name}"
                )
            temporary.replace(target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

        content_type, _ = mimetypes.guess_type(normalized_name)
        return ExtractedArchiveVideo(
            task_id=task_id,
            path=target,
            original_filename=normalized_name,
            content_type=content_type,
            size_bytes=written,
        )

    @staticmethod
    def _validate_member_path(member: zipfile.ZipInfo) -> str:
        raw_name = member.filename.replace("\\", "/")
        path = PurePosixPath(raw_name)
        if (
            not raw_name
            or len(raw_name) > 1_024
            or raw_name.startswith("/")
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
            or (path.parts and ":" in path.parts[0])
        ):
            raise VideoValidationError(
                f"Unsafe path in ZIP archive: {member.filename or '<empty>'}"
            )
        return path.as_posix()

    @staticmethod
    def _is_ignored_system_entry(filename: str) -> bool:
        path = PurePosixPath(filename)
        basename = path.name.lower()
        return (
            path.parts[0].lower() == "__macosx"
            or basename in {".ds_store", "desktop.ini", "thumbs.db"}
            or basename.startswith("._")
        )

    @staticmethod
    def _is_symlink(member: zipfile.ZipInfo) -> bool:
        unix_mode = member.external_attr >> 16
        return stat.S_ISLNK(unix_mode)

    @staticmethod
    def _remove_directories(paths: list[Path]) -> None:
        for path in paths:
            shutil.rmtree(path, ignore_errors=True)
