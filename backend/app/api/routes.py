from __future__ import annotations

import json
import shutil
from typing import Annotated, Any
from uuid import UUID, uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse

from app.api.dependencies import (
    get_archive_upload_service,
    get_repository,
    get_upload_service,
)
from app.core.config import Settings, get_settings
from app.core.metrics import TASKS_CREATED
from app.domain.enums import SourceKind, TaskStage
from app.domain.schemas import (
    SUPPORTED_TRACKING_CLASSES,
    ArchiveTaskItem,
    CreateArchiveTasksResponse,
    CreateTaskResponse,
    FinalResult,
    SkippedArchiveFile,
    TaskStatus,
    VideoSource,
)
from app.infrastructure.serialization import model_to_stream_fields
from app.infrastructure.task_repository import (
    RedisTaskRepository,
    TaskNotFoundError,
    TaskStatusUnavailableError,
)
from app.services.archive_upload_service import ArchiveUploadService
from app.services.upload_service import UploadService
from app.services.video_validation import (
    UploadSizeLimitError,
    UploadStorageLimitError,
    VideoValidationError,
    validate_remote_url,
)

router = APIRouter()


def _parse_tracking_classes(raw: Any) -> tuple[str, ...] | None:
    """Validate the optional multipart/JSON class-selection contract."""

    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="classes must be a JSON array of class names",
            ) from exc
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="classes must be a JSON array of class names",
        )

    normalized = tuple(item.strip() for item in raw)
    if any(not item for item in normalized):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="classes cannot contain empty names",
        )
    if len(set(normalized)) != len(normalized):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="classes cannot contain duplicates",
        )

    supported = set(SUPPORTED_TRACKING_CLASSES)
    unknown = sorted(set(normalized) - supported)
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "message": "Unknown tracking classes",
                "unknown": unknown,
                "supported": list(SUPPORTED_TRACKING_CLASSES),
            },
        )
    return normalized


def _upload_error(exc: VideoValidationError) -> HTTPException:
    if isinstance(exc, UploadStorageLimitError):
        return HTTPException(
            status_code=507,
            detail=str(exc),
        )
    if isinstance(exc, UploadSizeLimitError):
        return HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(exc),
        )
    return HTTPException(
        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        detail=str(exc),
    )


@router.post(
    "/process",
    response_model=CreateTaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_process_task(
    request: Request,
    repository: Annotated[RedisTaskRepository, Depends(get_repository)],
    uploads: Annotated[UploadService, Depends(get_upload_service)],
    settings: Annotated[Settings, Depends(get_settings)],
    file: Annotated[UploadFile | None, File()] = None,
    video_url: Annotated[str | None, Form()] = None,
    classes: Annotated[str | None, Form()] = None,
) -> CreateTaskResponse:
    content_type = request.headers.get("content-type", "")
    tracking_classes: tuple[str, ...] | None
    if (
        file is None
        and video_url is None
        and content_type.startswith("application/json")
    ):
        body = await request.json()
        video_url = body.get("video_url")
        tracking_classes = _parse_tracking_classes(body.get("classes"))
    else:
        tracking_classes = _parse_tracking_classes(classes)

    if (file is None) == (video_url is None):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Provide exactly one source: file or video_url",
        )

    task_id = uuid4()
    try:
        if file is not None:
            saved_path = await uploads.save(task_id, file)
            source = VideoSource(
                kind=SourceKind.UPLOAD,
                upload_path=str(saved_path),
                original_filename=file.filename,
                content_type=file.content_type,
                tracking_classes=tracking_classes,
            )
        else:
            assert video_url is not None
            validate_remote_url(video_url, settings)
            source = VideoSource(
                kind=SourceKind.URL,
                url=video_url,
                tracking_classes=tracking_classes,
            )
    except VideoValidationError as exc:
        raise _upload_error(exc) from exc

    try:
        await repository.create_and_enqueue(
            task_id,
            source,
            settings.stream_ingest_jobs,
            model_to_stream_fields(source) | {"task_id": str(task_id)},
        )
    except Exception as exc:
        shutil.rmtree(settings.upload_dir / str(task_id), ignore_errors=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Task queue is temporarily unavailable",
        ) from exc
    TASKS_CREATED.inc()
    base = settings.api_prefix
    return CreateTaskResponse(
        task_id=task_id,
        status_url=f"{base}/status/{task_id}",
        results_url=f"{base}/results/{task_id}",
    )


@router.post(
    "/process/archive",
    response_model=CreateArchiveTasksResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_archive_tasks(
    repository: Annotated[RedisTaskRepository, Depends(get_repository)],
    archive_uploads: Annotated[
        ArchiveUploadService, Depends(get_archive_upload_service)
    ],
    settings: Annotated[Settings, Depends(get_settings)],
    file: Annotated[UploadFile, File(description="ZIP archive with video files")],
    classes: Annotated[str | None, Form()] = None,
) -> CreateArchiveTasksResponse:
    tracking_classes = _parse_tracking_classes(classes)
    try:
        extraction = await archive_uploads.extract(file)
    except VideoValidationError as exc:
        raise _upload_error(exc) from exc

    queued_tasks: list[tuple[UUID, VideoSource, dict[str, str]]] = []
    for video in extraction.videos:
        source = VideoSource(
            kind=SourceKind.UPLOAD,
            upload_path=str(video.path),
            original_filename=video.original_filename,
            content_type=video.content_type,
            tracking_classes=tracking_classes,
        )
        stream_fields = model_to_stream_fields(source) | {"task_id": str(video.task_id)}
        queued_tasks.append((video.task_id, source, stream_fields))

    try:
        await repository.create_many_and_enqueue(
            queued_tasks,
            settings.stream_ingest_jobs,
        )
    except Exception as exc:
        await archive_uploads.cleanup(extraction)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Task queue is temporarily unavailable",
        ) from exc

    TASKS_CREATED.inc(len(extraction.videos))
    base = settings.api_prefix
    tasks = [
        ArchiveTaskItem(
            task_id=video.task_id,
            filename=video.original_filename,
            size_bytes=video.size_bytes,
            status_url=f"{base}/status/{video.task_id}",
            results_url=f"{base}/results/{video.task_id}",
        )
        for video in extraction.videos
    ]
    skipped_files = [
        SkippedArchiveFile(filename=item.filename, reason=item.reason)
        for item in extraction.skipped
    ]
    return CreateArchiveTasksResponse(
        archive_filename=extraction.archive_filename,
        archive_size_bytes=extraction.archive_size_bytes,
        accepted_size_bytes=extraction.accepted_size_bytes,
        accepted_count=len(tasks),
        skipped_count=len(skipped_files),
        tasks=tasks,
        skipped_files=skipped_files,
    )


@router.get("/status/{task_id}", response_model=TaskStatus)
async def get_task_status(
    task_id: UUID,
    repository: Annotated[RedisTaskRepository, Depends(get_repository)],
) -> TaskStatus:
    try:
        return await repository.get_status(task_id)
    except TaskNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Task not found") from exc
    except TaskStatusUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Task status is temporarily unavailable; retry shortly",
            headers={"Retry-After": "1"},
        ) from exc


@router.get("/results/{task_id}", response_model=FinalResult)
async def get_task_results(
    task_id: UUID,
    repository: Annotated[RedisTaskRepository, Depends(get_repository)],
) -> FinalResult | JSONResponse:
    try:
        current = await repository.get_status(task_id)
    except TaskNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Task not found") from exc
    except TaskStatusUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Task status is temporarily unavailable; retry shortly",
            headers={"Retry-After": "1"},
        ) from exc
    if current.stage == TaskStage.FAILED:
        raise HTTPException(
            status_code=422,
            detail={
                "code": current.error_code,
                "message": current.error_detail,
            },
        )
    result = await repository.get_result(task_id)
    if result is None:
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "task_id": str(task_id),
                "stage": current.stage.value,
                "progress": current.progress,
                "message": "Result is not ready yet",
            },
        )
    return result
