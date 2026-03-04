from __future__ import annotations

from fastapi import FastAPI

from app.api.datasets import router as datasets_router
from app.config import get_settings

app = FastAPI(title="Radiology WebUI API")
app.include_router(datasets_router)


@app.get("/api/health")
def health() -> dict[str, str]:
    _ = get_settings()
    return {"status": "ok"}
