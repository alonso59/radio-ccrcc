from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.models.dataset import DatasetSummary, PatientSummary, SeriesInfo


CASE_ID_PATTERN = re.compile(r"(case_\d{5})")
LATERALITY_PATTERN = re.compile(r"_(?:side)?([LR])$", re.IGNORECASE)
NIFTI_SUFFIXES = (".nii.gz", ".nii")
PHASE_ORDER = {
    "NC": 0,
    "ART": 1,
    "VEN": 2,
    "DELAY": 3,
    "UNDEFINED": 4,
    "DELETED": 5,
}
PHASE_NORMALIZATION = {
    "nc": "NC",
    "art": "ART",
    "ven": "VEN",
    "delay": "DELAY",
    "undefined": "UNDEFINED",
    "deleted": "DELETED",
}


def resolve_dataset_path(data_root: Path | str, dataset_id: str) -> Path:
    root = Path(data_root).expanduser().resolve()
    dataset_path = (root / dataset_id).resolve()

    if not dataset_path.is_relative_to(root):
        raise ValueError(f"Dataset '{dataset_id}' resolves outside DATA_ROOT")
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset '{dataset_id}' not found")

    return dataset_path


def list_datasets(data_root: Path | str) -> list[DatasetSummary]:
    root = Path(data_root).expanduser().resolve()
    if not root.exists():
        return []

    datasets: list[DatasetSummary] = []
    for dataset_path in sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.startswith("Dataset")),
        key=lambda path: path.name,
    ):
        has_nifti = (dataset_path / "nifti").is_dir()
        has_seg = (dataset_path / "seg").is_dir()
        has_voi = (dataset_path / "voi").is_dir()
        has_manifest = (dataset_path / "manifest.csv").is_file()

        try:
            patient_count = len(discover_patients(dataset_path))
        except Exception:
            patient_count = 0

        datasets.append(
            DatasetSummary(
                dataset_id=dataset_path.name,
                path=str(dataset_path),
                patient_count=patient_count,
                has_nifti=has_nifti,
                has_seg=has_seg,
                has_voi=has_voi,
                has_manifest=has_manifest,
            )
        )

    return datasets


def discover_patients(dataset_path: Path | str) -> list[PatientSummary]:
    dataset_path = Path(dataset_path).expanduser().resolve()
    nifti_entries = _collect_nifti_entries(dataset_path)
    voi_entries = _collect_voi_entries(dataset_path)

    patients: dict[str, dict[str, Any]] = {}
    for entry in nifti_entries:
        patient = patients.setdefault(
            entry["patient_id"],
            {
                "patient_id": entry["patient_id"],
                "source_patient_id": None,
                "group": None,
                "phases": set(),
                "series_count": 0,
                "seg_count": 0,
                "voi_count": 0,
            },
        )
        patient["series_count"] += 1
        patient["seg_count"] += int(entry["has_seg"])
        _merge_patient_metadata(patient, entry)

    for entry in voi_entries:
        patient = patients.setdefault(
            entry["patient_id"],
            {
                "patient_id": entry["patient_id"],
                "source_patient_id": None,
                "group": None,
                "phases": set(),
                "series_count": 0,
                "seg_count": 0,
                "voi_count": 0,
            },
        )
        patient["voi_count"] += 1
        _merge_patient_metadata(patient, entry)

    return [
        PatientSummary(
            patient_id=patient["patient_id"],
            source_patient_id=patient["source_patient_id"],
            group=patient["group"],
            phases=sorted(patient["phases"], key=_phase_sort_key),
            series_count=patient["series_count"],
            seg_count=patient["seg_count"],
            voi_count=patient["voi_count"],
        )
        for patient in sorted(patients.values(), key=lambda item: item["patient_id"])
    ]


def discover_series(dataset_path: Path | str, patient_id: str) -> list[SeriesInfo]:
    dataset_path = Path(dataset_path).expanduser().resolve()
    nifti_entries = [
        entry for entry in _collect_nifti_entries(dataset_path) if entry["patient_id"] == patient_id
    ]
    voi_entries = [
        entry for entry in _collect_voi_entries(dataset_path) if entry["patient_id"] == patient_id
    ]

    series = [
        SeriesInfo(
            series_id=entry["series_id"],
            patient_id=entry["patient_id"],
            type=entry["type"],
            group=entry["group"],
            phase=entry["phase"],
            laterality=entry.get("laterality"),
            filename=entry["filename"],
            image_path=entry["image_path"],
            mask_path=entry["mask_path"],
            has_seg=entry["has_seg"],
        )
        for entry in sorted(nifti_entries + voi_entries, key=_series_sort_key)
    ]
    return series


def _merge_patient_metadata(patient: dict[str, Any], entry: dict[str, Any]) -> None:
    if entry.get("group"):
        patient["group"] = patient["group"] or entry["group"]
    if entry.get("source_patient_id"):
        patient["source_patient_id"] = patient["source_patient_id"] or entry["source_patient_id"]
    if entry.get("phase"):
        patient["phases"].add(entry["phase"])


def _series_sort_key(entry: dict[str, Any]) -> tuple[int, int, str]:
    series_type_order = 0 if entry["type"] == "nifti" else 1
    return (series_type_order, _phase_sort_key(entry.get("phase")), entry["filename"])


def _phase_sort_key(phase: str | None) -> tuple[int, str]:
    normalized = _normalize_phase(phase)
    return (PHASE_ORDER.get(normalized, 99), normalized)


def _normalize_phase(value: str | None) -> str:
    cleaned = (value or "").strip()
    if not cleaned:
        return "UNDEFINED"
    return PHASE_NORMALIZATION.get(cleaned.lower(), cleaned.upper())


def _extract_patient_id(case_name: str) -> str:
    match = CASE_ID_PATTERN.search(case_name)
    return match.group(1) if match else case_name


def _extract_laterality(name: str) -> str | None:
    match = LATERALITY_PATTERN.search(name)
    if not match:
        return None
    return match.group(1).upper()


def _strip_nifti_suffix(filename: str) -> str:
    for suffix in NIFTI_SUFFIXES:
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    return Path(filename).stem


def _nifti_files(nifti_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in nifti_dir.iterdir():
        if path.is_file() and any(path.name.endswith(suffix) for suffix in NIFTI_SUFFIXES):
            files.append(path)
    return sorted(files, key=lambda path: path.name)


def _load_manifest_index(dataset_path: Path) -> dict[str, dict[str, str]]:
    manifest_path = dataset_path / "manifest.csv"
    if not manifest_path.is_file():
        return {}

    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        index: dict[str, dict[str, str]] = {}
        for row in reader:
            filename = (row.get("filename") or "").strip()
            if filename:
                index[filename] = row
        return index


def _find_seg_path(dataset_path: Path, image_stem: str) -> Path | None:
    seg_dir = dataset_path / "seg"
    if not seg_dir.is_dir():
        return None

    seg_stem = re.sub(r"_0000$", "", image_stem)
    for suffix in NIFTI_SUFFIXES:
        candidate = seg_dir / f"{seg_stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _collect_nifti_entries(dataset_path: Path) -> list[dict[str, Any]]:
    nifti_dir = dataset_path / "nifti"
    if not nifti_dir.is_dir():
        return []

    manifest_index = _load_manifest_index(dataset_path)
    entries: list[dict[str, Any]] = []
    for image_path in _nifti_files(nifti_dir):
        filename = image_path.name
        image_stem = _strip_nifti_suffix(filename)
        row = manifest_index.get(filename, {})
        patient_id = (row.get("case_id") or "").strip() or _extract_patient_id(image_stem)
        source_patient_id = (row.get("patient_id") or "").strip() or None
        group = (row.get("group") or "").strip() or None
        phase = _normalize_phase(row.get("phase"))
        seg_path = _find_seg_path(dataset_path, image_stem)

        entries.append(
            {
                "series_id": f"nifti:{image_stem}",
                "patient_id": patient_id,
                "source_patient_id": source_patient_id,
                "type": "nifti",
                "group": group,
                "phase": phase,
                "laterality": None,
                "filename": filename,
                "image_path": str(image_path),
                "mask_path": str(seg_path) if seg_path else None,
                "has_seg": seg_path is not None,
            }
        )

    return entries


def _candidate_voi_mask_roots(dataset_path: Path) -> list[Path]:
    voi_dir = dataset_path / "voi"
    return [path for path in (voi_dir / "mask", voi_dir / "segmentation") if path.is_dir()]


def _collect_voi_entries(dataset_path: Path) -> list[dict[str, Any]]:
    image_root = dataset_path / "voi" / "images"
    if not image_root.is_dir():
        return []

    mask_roots = _candidate_voi_mask_roots(dataset_path)
    entries: list[dict[str, Any]] = []

    for image_path in sorted(image_root.rglob("*.npy")):
        relative = image_path.relative_to(image_root)
        if len(relative.parts) < 3:
            continue

        if len(relative.parts) == 3:
            group, patient_id, filename = relative.parts
            phase = "UNDEFINED"
        else:
            group, patient_id, phase = relative.parts[:3]
            filename = relative.name
        stem = image_path.stem
        mask_path = next((root / relative for root in mask_roots if (root / relative).exists()), None)

        entries.append(
            {
                "series_id": f"voi:{group}:{phase}:{stem}",
                "patient_id": patient_id,
                "source_patient_id": None,
                "type": "voi",
                "group": group,
                "phase": _normalize_phase(phase),
                "laterality": _extract_laterality(stem),
                "filename": filename,
                "image_path": str(image_path),
                "mask_path": str(mask_path) if mask_path else None,
                "has_seg": mask_path is not None,
            }
        )

    return entries
