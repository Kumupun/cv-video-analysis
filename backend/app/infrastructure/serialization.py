from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def model_to_stream_fields(model: BaseModel) -> dict[str, str]:
    return {
        "event_type": model.__class__.__name__,
        "schema_version": "1",
        "payload": model.model_dump_json(),
    }


def stream_fields_to_model(fields: dict[str, Any], model_type: type[T]) -> T:
    payload = fields.get("payload")
    if payload is None:
        raise ValueError("Redis stream message does not contain payload")
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    if not isinstance(payload, str):
        payload = json.dumps(payload)
    return model_type.model_validate_json(payload)
