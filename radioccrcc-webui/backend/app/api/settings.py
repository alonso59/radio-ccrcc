from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.settings import DatasetViewerSettings
from app.services.settings_store import settings_store


router = APIRouter(tags=["settings"])


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, RuntimeError):
        return HTTPException(status_code=500, detail=str(exc))
    return HTTPException(status_code=500, detail="Unexpected settings error")


@router.get("/api/settings", response_model=dict[str, DatasetViewerSettings])
def get_settings_payload() -> dict[str, DatasetViewerSettings]:
    try:
        return settings_store.load()
    except Exception as exc:
        raise _http_error(exc) from exc


@router.put("/api/settings", response_model=dict[str, DatasetViewerSettings])
def put_settings_payload(
    payload: dict[str, DatasetViewerSettings],
) -> dict[str, DatasetViewerSettings]:
    try:
        return settings_store.save(payload)
    except Exception as exc:
        raise _http_error(exc) from exc
