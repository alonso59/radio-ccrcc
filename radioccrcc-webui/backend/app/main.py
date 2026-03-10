from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from app.api.datasets import router as datasets_router
from app.api.mesh import router as mesh_router
from app.api.review import router as review_router
from app.api.settings import router as settings_router
from app.api.slices import router as slices_router
from app.config import get_settings
from app.middleware.auth import AuthMiddleware

app = FastAPI(title="Radiology WebUI API")
app.add_middleware(AuthMiddleware)
app.include_router(datasets_router)
app.include_router(mesh_router)
app.include_router(review_router)
app.include_router(settings_router)
app.include_router(slices_router)

_STATIC_DIR = Path(os.environ.get("STATIC_ROOT", "/app/static")).resolve()


def _safe_static_file(relative_path: str) -> Path | None:
    candidate = (_STATIC_DIR / relative_path).resolve()
    if not candidate.is_relative_to(_STATIC_DIR):
        return None
    if not candidate.is_file():
        return None
    return candidate


@app.get("/api/health")
def health() -> dict[str, str]:
    _ = get_settings()
    return {"status": "ok"}


@app.get("/{full_path:path}", include_in_schema=False)
def frontend(full_path: str):
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not Found")

    if not _STATIC_DIR.is_dir():
        raise HTTPException(status_code=404, detail="Frontend static build not found at /app/static")

    normalized = full_path.strip("/")
    if normalized:
        static_file = _safe_static_file(normalized)
        if static_file is not None:
            return FileResponse(static_file)

    index_path = _STATIC_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Frontend index.html not found at /app/static/index.html",
        )
    return FileResponse(index_path)
