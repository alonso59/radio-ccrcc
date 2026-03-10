from __future__ import annotations

from copy import deepcopy

from fastapi import APIRouter, HTTPException, Query, Request, Response

from app.models.dataset import VolumeInfo
from app.services.slice_renderer import DEFAULT_LAYER_CONFIG, render_slice
from app.services.volume_cache import volume_cache


router = APIRouter(tags=["slices"])


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, RuntimeError):
        message = str(exc)
        if "load handle" in message.lower():
            return HTTPException(status_code=410, detail=message)
        return HTTPException(status_code=409, detail=message)
    if isinstance(exc, (IndexError, ValueError)):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail="Unexpected slice error")


def _parse_layers(layers: str | None) -> list[int]:
    if layers is None:
        return [1, 2]
    cleaned = layers.strip()
    if not cleaned:
        return []
    parsed: list[int] = []
    for part in cleaned.split(","):
        token = part.strip()
        if not token:
            continue
        parsed.append(int(token))
    return sorted(set(parsed))


@router.post(
    "/api/datasets/{dataset_id}/patients/{patient_id}/series/{series_id}/load",
    response_model=VolumeInfo,
)
def load_series(dataset_id: str, patient_id: str, series_id: str):
    try:
        return volume_cache.load_series(dataset_id, patient_id, series_id)
    except Exception as exc:
        raise _http_error(exc) from exc


@router.get("/api/slice/{axis}/{index}")
def slice_png(
    axis: str,
    index: int,
    request: Request,
    load_handle: str = Query(...),
    ww: float = Query(default=400.0),
    wl: float = Query(default=50.0),
    layers: str | None = Query(default="1,2"),
):
    try:
        volume, mask, _spacing = volume_cache.get_by_handle(load_handle)
        visible_layers = _parse_layers(layers)
        layer_config = deepcopy(DEFAULT_LAYER_CONFIG)
        for label, config in layer_config.items():
            opacity_key = f"opacity_{label}"
            if opacity_key in request.query_params:
                config["alpha"] = float(request.query_params[opacity_key])

        png_bytes = render_slice(
            volume=volume,
            mask=mask,
            axis=axis,
            index=index,
            ww=ww,
            wl=wl,
            layers=visible_layers,
            layer_config=layer_config,
        )
    except Exception as exc:
        raise _http_error(exc) from exc

    return Response(content=png_bytes, media_type="image/png")
