from __future__ import annotations

from uuid import uuid4

import pytest

from app.core.config import Settings
from app.workers.ingest import IngestHandler


class _Repository:
    async def get_chunk_object_ref(self, task_id, chunk_index: int) -> str:
        return f"ray-object:{chunk_index}"


class _ObjectStore:
    def __init__(self) -> None:
        self.released: list[tuple[str, bool]] = []

    def release(self, token: str, *, suppress_errors: bool = True) -> bool:
        self.released.append((token, suppress_errors))
        return True


@pytest.mark.asyncio
async def test_ingest_owner_releases_contiguous_completed_chunks() -> None:
    handler = IngestHandler.__new__(IngestHandler)
    handler.settings = Settings(
        _env_file=None,
        ray_actor_call_timeout_seconds=1,
    )
    handler.repository = _Repository()
    handler.object_store = _ObjectStore()

    released_count = await handler._release_completed_chunks(
        uuid4(),
        released_count=1,
        tracking_completed=4,
    )

    assert released_count == 4
    assert handler.object_store.released == [
        ("ray-object:1", False),
        ("ray-object:2", False),
        ("ray-object:3", False),
    ]
