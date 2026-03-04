from __future__ import annotations

from fastapi import FastAPI

from app.config import get_settings


settings = get_settings()

app = FastAPI(title="Radiology WebUI API")


@app.get("/api/health")
def health() -> dict[str, str]:
    _ = settings
    return {"status": "ok"}
