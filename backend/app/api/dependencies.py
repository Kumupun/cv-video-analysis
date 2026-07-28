from __future__ import annotations

from fastapi import Request

from app.infrastructure.redis_streams import RedisStreams
from app.infrastructure.task_repository import RedisTaskRepository
from app.services.upload_service import UploadService


def get_streams(request: Request) -> RedisStreams:
    return request.app.state.streams


def get_repository(request: Request) -> RedisTaskRepository:
    return request.app.state.repository


def get_upload_service(request: Request) -> UploadService:
    return request.app.state.upload_service
