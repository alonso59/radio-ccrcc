from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field


class DatasetSummary(BaseModel):
    dataset_id: str
    patient_count: int = 0
    has_nifti: bool = False
    has_seg: bool = False
    has_voi: bool = False
    has_manifest: bool = False


class PatientSummary(BaseModel):
    patient_id: str
    source_patient_id: str | None = None
    group: str | None = None
    phases: list[str] = Field(default_factory=list)
    series_count: int = 0
    seg_count: int = 0
    voi_count: int = 0


class SeriesInfo(BaseModel):
    series_id: str
    patient_id: str
    type: str
    group: str | None = None
    phase: str | None = None
    laterality: str | None = None
    filename: str
    has_seg: bool = False


@dataclass(frozen=True)
class SeriesSource:
    series_id: str
    patient_id: str
    type: str
    group: str | None
    phase: str | None
    laterality: str | None
    filename: str
    image_path: str
    mask_path: str | None
    has_seg: bool


class VolumeInfo(BaseModel):
    series_id: str
    load_handle: str
    shape: list[int] = Field(default_factory=list)
    spacing: list[float] = Field(default_factory=list)
    has_mask: bool = False
    labels: list[int] = Field(default_factory=list)
