"""File discovery: match flat nifti/ images with seg/ masks using manifest.csv.

Input layout (from converter):
    DatasetID/nifti/NN_case_YYYYY_0000.nii.gz   (flat)
    DatasetID/seg/NN_case_YYYYY.nii.gz           (flat)
    DatasetID/manifest.csv                        (columns: group, phase, …)

"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import SimpleITK as sitk

from converter import dicom_utils as converter_utils


# ---------------------------------------------------------------------------
# Patient ID extraction
# ---------------------------------------------------------------------------

def extract_patient_id(case_name: str, pattern: Optional[str] = None) -> str:
    """Extract patient ID from a case name.

    Falls back to case_XXXXX regex when *pattern* does not match.
    """
    if pattern:
        m = re.search(pattern, case_name)
        if m:
            return next((g for g in m.groups() if g is not None), m.group(0))
    m = re.search(r"(case_\d{5})", case_name)
    return m.group(1) if m else case_name


# ---------------------------------------------------------------------------
# Manifest lookup
# ---------------------------------------------------------------------------

def _find_manifest(image_folder: Path, manifest_output_dir: Optional[Path] = None) -> Optional[Path]:
    """Auto-discover manifest.csv relative to *image_folder*.

    Search order:
      1. image_folder/manifest.csv        (inside nifti/)
      2. image_folder/../manifest.csv      (DatasetID/ level)
    """
    candidates = []
    if manifest_output_dir is not None:
        candidates.append(manifest_output_dir / "manifest.csv")
    candidates.extend([
        image_folder / "manifest.csv",
        image_folder.parent / "manifest.csv",
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


MANIFEST_COLUMNS: List[str] = [
    "filename",
    "patient_id",
    "case_id",
    "group",
    "phase",
    "phase_source",
    "scan_idx",
    "modality",
    "scan_plane",
    "image_orientation",
    "spacing_quality",
    "nonuniformity_mm",
    "sex",
    "DOB",
    "age",
    "age_at_scan",
    "DOS",
    "study_time",
    "series_description",
    "series_number",
    "manufacturer",
    "manufacturer_model",
    "institution",
    "kvp",
    "contrast_agent",
    "spacing_x",
    "spacing_y",
    "spacing_z",
    "dim_x",
    "dim_y",
    "dim_z",
    "slice_thickness",
    "num_slices",
    "conversion_date",
    "dataset_type",
    "output_path",
    "validation",
    "protocol_source",
    "Laterality",
    "Tumor",
]


def _generate_manifest_if_missing(
    image_folder: Path,
    image_suffix: str,
    manifest_output_dir: Optional[Path] = None,
    patient_id_pattern: Optional[str] = None,
) -> Optional[Path]:
    target_dir = manifest_output_dir if manifest_output_dir is not None else image_folder.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = target_dir / "manifest.csv"
    if manifest_path.exists():
        return manifest_path

    rows = [
        _build_manifest_row(path, image_suffix, patient_id_pattern)
        for path in sorted(image_folder.glob(f"*{image_suffix}"))
        if path.is_file()
    ]
    if not rows:
        return None

    pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(manifest_path, index=False)
    print(f"       Generated fallback manifest: {manifest_path}  ({len(rows)} rows)")
    return manifest_path


def _build_manifest_row(
    image_path: Path,
    image_suffix: str,
    patient_id_pattern: Optional[str],
) -> Dict[str, object]:
    image = sitk.ReadImage(str(image_path))
    stem = image_path.name[: -len(image_suffix)] if image_path.name.endswith(image_suffix) else image_path.stem
    case_name = re.sub(r"_0000$", "", stem)

    case_id = _safe_meta(image, "CaseID", extract_patient_id(case_name, patient_id_pattern))
    patient_id = _safe_meta(image, "PatientID", case_id)
    group = converter_utils.normalize_class(_safe_meta(image, "AssignedClass", "NG"))
    phase, phase_source = _detect_phase_from_nifti(image, image_path.name)
    spacing = image.GetSpacing()
    dims = image.GetSize()

    return {
        "filename": image_path.name,
        "patient_id": patient_id,
        "case_id": case_id,
        "group": group,
        "phase": phase,
        "phase_source": phase_source,
        "scan_idx": _scan_index_from_filename(image_path.name),
        "modality": _safe_meta(image, "Modality"),
        "scan_plane": _detect_scan_plane_from_image(image),
        "image_orientation": _direction_to_string(image.GetDirection()),
        "spacing_quality": "unknown",
        "nonuniformity_mm": "nan",
        "sex": "nan",
        "DOB": "nan",
        "age": "nan",
        "age_at_scan": "nan",
        "DOS": converter_utils.format_date(_safe_meta(image, "StudyDate")),
        "study_time": converter_utils.format_time(_safe_meta(image, "StudyTime")),
        "series_description": _safe_meta(image, "SeriesDescription"),
        "series_number": _safe_meta(image, "SeriesNumber"),
        "manufacturer": _safe_meta(image, "Manufacturer"),
        "manufacturer_model": _safe_meta(image, "ManufacturerModelName"),
        "institution": _safe_meta(image, "InstitutionName"),
        "kvp": _safe_meta(image, "KVP"),
        "contrast_agent": _safe_meta(image, "ContrastBolusAgent"),
        "spacing_x": float(spacing[0]),
        "spacing_y": float(spacing[1]),
        "spacing_z": float(spacing[2]),
        "dim_x": int(dims[0]),
        "dim_y": int(dims[1]),
        "dim_z": int(dims[2]),
        "slice_thickness": _safe_meta(image, "SliceThickness"),
        "num_slices": int(dims[2]) if len(dims) >= 3 else "nan",
        "conversion_date": _safe_meta(image, "ConversionDate"),
        "dataset_type": _safe_meta(image, "DatasetType", "generic"),
        "output_path": image_path.name,
        "validation": "generated",
        "protocol_source": phase_source,
        "Laterality": "",
        "Tumor": "",
    }


def _safe_meta(image: sitk.Image, key: str, default: str = "nan") -> str:
    if image.HasMetaDataKey(key):
        value = image.GetMetaData(key).strip()
        if value:
            return value
    return default


def _scan_index_from_filename(filename: str) -> int:
    match = re.match(r"(\d+)_", filename)
    return int(match.group(1)) if match else 0


def _detect_phase_from_nifti(image: sitk.Image, filename: str) -> tuple[str, str]:
    embedded = _safe_meta(image, "DetectedProtocol", "").lower()
    if embedded in {"nc", "art", "ven", "delay", "undefined"}:
        source = "metadata" if embedded != "undefined" else "undefined"
        return embedded or "undefined", source

    metadata_text = " ".join(
        value
        for value in [
            _safe_meta(image, "SeriesDescription", ""),
            _safe_meta(image, "ProtocolName", ""),
        ]
        if value
    )
    label = converter_utils._classify_text(metadata_text)
    if label in {"nc", "art", "ven", "delay"}:
        return label, "metadata"

    path_label = converter_utils._classify_text(filename)
    if path_label in {"nc", "art", "ven", "delay"}:
        return path_label, "filename"
    return "undefined", "undefined"


def _detect_scan_plane_from_image(image: sitk.Image) -> str:
    direction = image.GetDirection()
    if len(direction) < 9:
        return "AXIAL"
    row = direction[:3]
    col = direction[3:6]
    normal = (
        row[1] * col[2] - row[2] * col[1],
        row[2] * col[0] - row[0] * col[2],
        row[0] * col[1] - row[1] * col[0],
    )
    abs_n = [abs(value) for value in normal]
    idx = int(max(range(3), key=lambda i: abs_n[i]))
    if max(abs_n) < 0.8:
        return "OBLIQUE"
    return ["SAGITTAL", "CORONAL", "AXIAL"][idx]


def _direction_to_string(direction: tuple[float, ...]) -> str:
    if not direction:
        return "nan"
    return "\\".join(f"{value:.6f}" for value in direction[:6])


def _load_manifest(manifest_path: Path) -> Dict[str, Dict]:
    """Read manifest.csv and index rows by filename.

    Returns {filename: {"group": …, "phase": …, "patient_id": …, …}}.
    """
    df = pd.read_csv(manifest_path)
    index: Dict[str, Dict] = {}
    for _, row in df.iterrows():
        fname = str(row.get("filename", ""))
        if fname:
            index[fname] = row.to_dict()
    return index


# ---------------------------------------------------------------------------
# Main discovery (flat nifti/ + seg/ with manifest)
# ---------------------------------------------------------------------------

def discover_files(
    image_folder: Path,
    mask_folder: Path,
    image_suffix: str,
    mask_suffix: str,
    manifest_output_dir: Optional[Path] = None,
    patient_id_pattern: Optional[str] = None,
) -> Dict[str, Dict]:
    """Discover image-mask pairs from flat nifti/ and seg/ folders.

    Reads ``manifest.csv`` (auto-found) for **group** and **phase** per file.

    Filename convention::

        nifti/NN_case_YYYYY_0000.nii.gz   →   seg/NN_case_YYYYY.nii.gz

    The ``_0000`` channel suffix in the image name is stripped to find
    the matching segmentation mask.

    Returns:
        Dict keyed by case_key with values:
            case_name, group, phase, image_path, mask_path, patient_id
    """
    image_folder = Path(image_folder)
    mask_folder = Path(mask_folder)

    # Load manifest (group + phase come from here)
    manifest_path = _find_manifest(image_folder, manifest_output_dir=manifest_output_dir)
    if manifest_path is None:
        manifest_path = _generate_manifest_if_missing(
            image_folder,
            image_suffix,
            manifest_output_dir=manifest_output_dir,
            patient_id_pattern=patient_id_pattern,
        )
    manifest_index: Dict[str, Dict] = {}
    if manifest_path is not None:
        manifest_index = _load_manifest(manifest_path)
        print(f"       Manifest loaded: {manifest_path}  ({len(manifest_index)} rows)")
    else:
        print("       ⚠ No manifest.csv found — group/phase will default to NG/undefined")

    file_list: Dict[str, Dict] = {}

    for img_path in sorted(image_folder.glob(f"*{image_suffix}")):
        fname = img_path.name                             # e.g. 00_case_00001_0000.nii.gz
        cname = fname.replace(image_suffix, "")           # e.g. 00_case_00001

        # Derive mask name: strip _0000 channel suffix
        mask_stem = re.sub(r"_0000$", "", cname)          # e.g. 00_case_00001
        mask_name = f"{mask_stem}{mask_suffix}"            # e.g. 00_case_00001.nii.gz
        mask_path = mask_folder / mask_name
        if not mask_path.exists():
            continue

        # Lookup manifest row by filename
        mrow = manifest_index.get(fname, {})
        group = str(mrow.get("group", "NG")).strip() or "NG"
        phase = str(mrow.get("phase", "undefined")).strip() or "undefined"
        pid = extract_patient_id(cname, patient_id_pattern)

        # Use case_id from manifest if available, else from filename
        case_id = str(mrow.get("case_id", pid)).strip() or pid

        file_list[cname] = {
            "case_name": cname,
            "group": group,
            "phase": phase,
            "image_path": str(img_path),
            "mask_path": str(mask_path),
            "patient_id": pid,
            "case_id": case_id,
        }

    if not file_list:
        raise ValueError(
            f"No matching image-mask pairs found.\n"
            f"  image_folder: {image_folder}  (suffix: {image_suffix})\n"
            f"  mask_folder:  {mask_folder}  (suffix: {mask_suffix})"
        )
    return file_list
