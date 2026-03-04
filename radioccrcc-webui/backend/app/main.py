from __future__ import annotations

from fastapi import FastAPI

from app.api.datasets import router as datasets_router
from app.api.mesh import router as mesh_router
from app.api.slices import router as slices_router
from app.config import get_settings

app = FastAPI(title="Radiology WebUI API")
app.include_router(datasets_router)
app.include_router(mesh_router)
app.include_router(slices_router)


@app.get("/api/health")
def health() -> dict[str, str]:
    _ = get_settings()
    return {"status": "ok"}
