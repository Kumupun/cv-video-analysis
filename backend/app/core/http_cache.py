from __future__ import annotations

from fastapi import Response


def apply_cache_policy(path: str, response: Response, *, api_prefix: str) -> None:
    """Prevent stale task state and frontend bundles after a deployment."""

    if path.startswith(f"{api_prefix}/status/") or path.startswith(
        f"{api_prefix}/results/"
    ):
        response.headers["Cache-Control"] = "no-store"
        return
    if path in {"/", "/index.html", "/app.js", "/styles.css"}:
        response.headers["Cache-Control"] = "no-cache"
