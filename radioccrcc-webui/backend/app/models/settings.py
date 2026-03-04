from __future__ import annotations

from pydantic import BaseModel, Field


class DatasetViewerSettings(BaseModel):
    last_patient: str | None = None
    last_series: str | None = None
    ww: float | None = None
    wl: float | None = None
    layers_visible: list[int] = Field(default_factory=list)
    layers_opacity: dict[str, float] = Field(default_factory=dict)
