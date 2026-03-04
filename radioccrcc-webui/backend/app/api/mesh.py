from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Response

from app.services.mesh_generator import generate_mesh
from app.services.volume_cache import volume_cache


router = APIRouter(tags=["mesh"])


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, RuntimeError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail="Unexpected mesh error")


@router.get("/api/mesh/{label}")
def mesh_glb(label: int, smooth: bool = Query(default=True)):
    try:
        _volume, mask, spacing = volume_cache.get_current()
        glb_bytes = generate_mesh(mask=mask, label=label, spacing=spacing, smooth=smooth)
    except Exception as exc:
        raise _http_error(exc) from exc

    if glb_bytes is None:
        raise HTTPException(
            status_code=404,
            detail=f"No mesh is available for label {label}",
        )

    return Response(content=glb_bytes, media_type="model/gltf-binary")
