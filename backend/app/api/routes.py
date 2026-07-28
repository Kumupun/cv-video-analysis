from __future__ import annotations

import shutil
from typing import Annotated
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

from app.api.dependencies import get_repository, get_upload_service
from app.core.config import Settings, get_settings
from app.core.metrics import TASKS_CREATED
from app.domain.enums import SourceKind, TaskStage
from app.domain.schemas import CreateTaskResponse, FinalResult, TaskStatus, VideoSource
from app.infrastructure.serialization import model_to_stream_fields
from app.infrastructure.task_repository import RedisTaskRepository, TaskNotFoundError
from app.services.upload_service import UploadService
from app.services.video_validation import VideoValidationError, validate_remote_url

router = APIRouter()


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
) -> CreateTaskResponse:
    content_type = request.headers.get("content-type", "")
    if (
        file is None
        and video_url is None
        and content_type.startswith("application/json")
    ):
        body = await request.json()
        video_url = body.get("video_url")

    if (file is None) == (video_url is None):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
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
            )
        else:
            assert video_url is not None
            validate_remote_url(video_url, settings)
            source = VideoSource(kind=SourceKind.URL, url=video_url)
    except VideoValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        ) from exc

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


@router.get("/status/{task_id}", response_model=TaskStatus)
async def get_task_status(
    task_id: UUID,
    repository: Annotated[RedisTaskRepository, Depends(get_repository)],
) -> TaskStatus:
    try:
        return await repository.get_status(task_id)
    except TaskNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Task not found") from exc


@router.get("/results/{task_id}", response_model=FinalResult)
async def get_task_results(
    task_id: UUID,
    repository: Annotated[RedisTaskRepository, Depends(get_repository)],
) -> FinalResult | JSONResponse:
    try:
        current = await repository.get_status(task_id)
    except TaskNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Task not found") from exc
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
