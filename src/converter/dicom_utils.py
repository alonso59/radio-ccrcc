"""DICOM utility functions: I/O, detection, filtering, metadata.

Consolidates all low-level helpers used by the conversion pipeline.
Preserves every feature from converter_v1/{dicom_io, detect, filters, metadata}.
"""

from __future__ import annotations

import csv
import io
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk


# ═══════════════════════════════════════════════════════════════════════════
# 1. DICOM TAG HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def get_tag(reader: sitk.ImageSeriesReader, tag: str, default: str = "") -> str:
    """Read a single DICOM tag from the first slice."""
    try:
        if reader.HasMetaDataKey(0, tag):
            return reader.GetMetaData(0, tag).strip()
    except Exception:
        pass
    return default


# ═══════════════════════════════════════════════════════════════════════════
# 2. SERIES DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════

def find_dicom_dirs(root: Path) -> List[str]:
    """Return leaf directories (no subdirs) under *root*."""
    dirs: List[str] = []
    for dirpath, subdirs, files in os.walk(root):
        if files and not subdirs:
            dirs.append(dirpath)
    return dirs


def discover_series(root: Path) -> List[dict]:
    """Find all DICOM series by SeriesInstanceUID under *root*.

    Returns a list of dicts with keys: series_uid, directory, files.
    """
    results: List[dict] = []
    reader = sitk.ImageSeriesReader()
    for d in find_dicom_dirs(root):
        try:
            uids = reader.GetGDCMSeriesIDs(str(d))
            for uid in uids:
                files = reader.GetGDCMSeriesFileNames(str(d), uid)
                if files:
                    results.append({"series_uid": uid, "directory": d, "files": files})
        except Exception:
            continue
    return results


# ═══════════════════════════════════════════════════════════════════════════
# 3. SERIES READING + QUALITY CHECK
# ═══════════════════════════════════════════════════════════════════════════

def read_series(files: list) -> Optional[Tuple[sitk.ImageSeriesReader, sitk.Image, Dict]]:
    """Read a DICOM series and capture SimpleITK spacing warnings.

    Returns (reader, image, quality_info) or None on failure.
    quality_info keys: nonuniformity_mm, severity ('ok'|'minor'|'moderate'|'critical').
    """
    try:
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(files)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()

        old_stderr = sys.stderr
        sys.stderr = buf = io.StringIO()
        try:
            image = reader.Execute()
        finally:
            sys.stderr = old_stderr

        quality: Dict = {"nonuniformity_mm": None, "severity": "ok"}
        m = re.search(r"maximum nonuniformity[:\s]+(\d+\.?\d*)", buf.getvalue(), re.I)
        if m:
            val = float(m.group(1))
            quality["nonuniformity_mm"] = val
            quality["severity"] = (
                "critical" if val > 50 else "moderate" if val > 10 else "minor"
            )

        return reader, image, quality
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# 4. ORIENTATION
# ═══════════════════════════════════════════════════════════════════════════

def to_ras(image: sitk.Image) -> sitk.Image:
    """Reorient image to RAS."""
    f = sitk.DICOMOrientImageFilter()
    f.SetDesiredCoordinateOrientation("RAS")
    return f.Execute(image)


# ═══════════════════════════════════════════════════════════════════════════
# 5. NIFTI WRITING + VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def write_nifti(image: sitk.Image, path: Path, compress: bool = True) -> None:
    """Write a SimpleITK image to NIfTI."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path), compress)


def validate_nifti(path: Path, original: sitk.Image) -> Dict:
    """Quick validation of written NIfTI against the original image."""
    result: Dict = {
        "valid": True, "file_exists": path.exists(),
        "readable": False, "dims_match": False,
    }
    if not result["file_exists"]:
        result["valid"] = False
        return result
    try:
        converted = sitk.ReadImage(str(path))
        result["readable"] = True
        result["dims_match"] = original.GetSize() == converted.GetSize()
        if not result["dims_match"]:
            result["valid"] = False
    except Exception as exc:
        result["valid"] = False
        result["error"] = str(exc)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 6. DATASET TYPE DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def detect_dataset_type(root: Path) -> str:
    """Detect dataset type from a sample DICOM header.

    Returns 'tcga' | 'ukbonn' | 'kits' | 'generic'.
    """
    for dirpath, subdirs, files in os.walk(root):
        if files and not subdirs:
            try:
                reader = sitk.ImageSeriesReader()
                names = reader.GetGDCMSeriesFileNames(dirpath)
                if not names:
                    continue
                reader.SetFileNames(names)
                reader.MetaDataDictionaryArrayUpdateOn()
                reader.LoadPrivateTagsOn()
                reader.Execute()
                pid = get_tag(reader, "0010|0020")
                if re.match(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", pid):
                    return "tcga"
                if pid.startswith("Anonym_") or re.match(r"^\d+$", pid):
                    return "ukbonn"
                if "KiTS" in pid or pid.startswith("case_"):
                    return "kits"
                return "generic"
            except Exception:
                continue
    return "generic"


# ═══════════════════════════════════════════════════════════════════════════
# 7. PATIENT ID EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

_INVALID_IDS = {"unknown", "anonymous", "none", "test", "patient"}


def _is_valid_pid(pid: str) -> bool:
    if not pid or len(pid) < 2:
        return False
    if pid.lower() in _INVALID_IDS:
        return False
    if re.match(r"^[0-9.]+$", pid) and "." in pid:
        return False
    return True


def _sanitize(pid: str) -> str:
    s = re.sub(r"[^\w\-.]", "_", pid)
    s = re.sub(r"_+", "_", s).strip("_.")
    return s or "UNKNOWN"


def extract_patient_id(
    reader: sitk.ImageSeriesReader, file_path: Optional[str] = None,
) -> Tuple[str, str]:
    """Return (patient_id, source)."""
    for tag, name in [
        ("0010|0020", "PatientID"),
        ("0010|0010", "PatientName"),
        ("0010|1000", "OtherPatientIDs"),
    ]:
        val = get_tag(reader, tag)
        if val and _is_valid_pid(val):
            return _sanitize(val), f"dicom_{name}"

    if file_path:
        parts = Path(file_path).parts
        for part in reversed(parts):
            if re.match(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", part):
                return part, "directory_tcga"
            if part.startswith("Anonym_"):
                return part, "directory_ukbonn"
            if re.match(r"case_\d+", part):
                return part, "directory_kits"
            if re.match(
                r"^[A-Za-z0-9][-_A-Za-z0-9]{2,}$", part
            ) and not part.startswith(("1.", "2.", "CT_", "MR_", "PT_")):
                return _sanitize(part), "directory_folder"

    return "UNKNOWN", "fallback"


# ═══════════════════════════════════════════════════════════════════════════
# 8. SCAN PLANE DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def detect_scan_plane(
    reader: sitk.ImageSeriesReader, image: sitk.Image,
) -> str:
    """Return 'AXIAL' | 'CORONAL' | 'SAGITTAL' | 'OBLIQUE'."""
    iop = get_tag(reader, "0020|0037")
    if iop:
        try:
            vals = [float(x) for x in iop.split("\\")]
            if len(vals) == 6:
                normal = np.cross(vals[:3], vals[3:])
                abs_n = [abs(x) for x in normal]
                idx = int(np.argmax(abs_n))
                if max(abs_n) < 0.8:
                    return "OBLIQUE"
                return ["SAGITTAL", "CORONAL", "AXIAL"][idx]
        except Exception:
            pass

    desc = get_tag(reader, "0008|103e").upper()
    for kw, plane in [
        ("AXIAL", "AXIAL"), (" AX ", "AXIAL"), ("TRANSVERSE", "AXIAL"),
        ("CORONAL", "CORONAL"), (" COR ", "CORONAL"),
        ("SAGITTAL", "SAGITTAL"), (" SAG ", "SAGITTAL"),
    ]:
        if kw in desc:
            return plane
    return "AXIAL"


# ═══════════════════════════════════════════════════════════════════════════
# 9. CONTRAST PHASE / PROTOCOL DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def _norm(text: str) -> str:
    return re.sub(
        r"\s+", " ",
        (text or "").lower().replace("_", " ").replace("/", " ").replace("-", " "),
    ).strip()


def _kw_re(keywords: List[str]) -> re.Pattern:
    parts = [rf"\b{re.escape(_norm(k))}\b" for k in keywords]
    return re.compile("|".join(parts), re.I)


_NC = _kw_re([
    "unenhanced", "non contrast", "non-contrast", "no contrast", "without contrast",
    "pre contrast", "pre-contrast", "precontrast", "native", "baseline",
    "w/o", "w o", "plain", "stone protocol", "renal colic",
])
_ART = _kw_re([
    "arterial", "arterial phase", "corticomedullary",
    "early arterial", "late arterial", "cta", "angio", "art",
])
_VEN = _kw_re([
    "venous", "venous phase", "portal venous", "porto-venous", "portovenous",
    "nephro", "nephrographic", "nephrographic phase",
    "parenchymal", "parenchymal phase", "ven", "pv",
])
_DELAY = _kw_re([
    "delay", "delayed", "delays", "excretory", "excretion",
    "urogram", "urographic", "ivp",
    "3 min", "3-min", "5 min", "8 min", "10 min", "12 min",
])
_EXCLUDE = _kw_re([
    "recon", "mpr", "mip", "vr", "summary", "reformatted", "localizer",
    "smart prep", "timing run", "test dose", "chest", "lungs", "lung",
])
_SEC_RE = re.compile(r"\b(\d{2,3})\s*sec\b", re.I)
_MIN_RE = re.compile(r"\b(\d{1,2})\s*min\b", re.I)


def _classify_text(text: str) -> str:
    """Classify free text into a contrast-phase label.

    Returns one of: 'nc', 'art', 'ven', 'delay', 'excluded', 'undefined'.
    """
    t = _norm(text)
    if not t:
        return "undefined"
    if _EXCLUDE.search(t):
        return "excluded"
    if _NC.search(t):
        return "nc"
    if _ART.search(t):
        return "art"
    if _DELAY.search(t):
        return "delay"
    if _VEN.search(t):
        return "ven"

    m = _SEC_RE.search(t)
    if m:
        sec = int(m.group(1))
        if 15 <= sec <= 40:
            return "art"
        if 55 <= sec <= 120:
            return "ven"
        if sec >= 150:
            return "delay"
    m = _MIN_RE.search(t)
    if m and int(m.group(1)) >= 3:
        return "delay"

    return "undefined"


def detect_protocol(
    reader: sitk.ImageSeriesReader, series_path: str,
) -> Tuple[str, str]:
    """Detect contrast phase from DICOM metadata or path.

    Returns (phase, source).
        phase:  'nc' | 'art' | 'ven' | 'delay' | 'undefined'
        source: 'metadata' | 'filename' | 'undefined'
    """
    meta_text = " ".join(
        get_tag(reader, t) for t in ("0008|103e", "0018|1030", "0018|0024")
    )
    label = _classify_text(meta_text)
    if label in ("nc", "art", "ven", "delay"):
        return label, "metadata"
    if label == "excluded":
        return "undefined", "metadata"

    path_label = _classify_text(series_path)
    if path_label in ("nc", "art", "ven", "delay"):
        return path_label, "filename"
    if path_label == "excluded":
        return "undefined", "filename"

    return "undefined", "undefined"


# ═══════════════════════════════════════════════════════════════════════════
# 10. SERIES FILTERING
# ═══════════════════════════════════════════════════════════════════════════

def filter_series(
    modality: str,
    scan_plane: str,
    num_slices: int,
    quality_severity: str,
    min_slices: int = 10,
) -> Optional[str]:
    """Check whether a series should be skipped.

    Returns a reason string if the series must be skipped, or None if it passes.
    """
    if modality == "SEG":
        return "SEG modality (segmentation)"
    if quality_severity == "critical":
        return "Critical spacing nonuniformity"
    if scan_plane != "AXIAL":
        return f"Non-axial scan ({scan_plane})"
    if num_slices < min_slices:
        return f"Too few slices ({num_slices} < {min_slices})"
    return None


# ═══════════════════════════════════════════════════════════════════════════
# 11. DATE / TIME FORMATTING
# ═══════════════════════════════════════════════════════════════════════════

def format_date(raw: str) -> str:
    """YYYYMMDD → YYYY-MM-DD (or 'nan')."""
    if not raw or raw == "nan" or len(raw) != 8:
        return "nan"
    try:
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    except Exception:
        return "nan"


def format_time(raw: str) -> str:
    """HHMMSS… → HH:MM:SS (or 'nan')."""
    if not raw or raw == "nan" or len(raw) < 6:
        return "nan"
    try:
        return f"{raw[:2]}:{raw[2:4]}:{raw[4:6]}"
    except Exception:
        return "nan"


def age_at_scan(dob: str, dos: str) -> object:
    """Age in years from YYYY-MM-DD strings.  Returns float or 'nan'."""
    try:
        d1 = datetime.strptime(dob, "%Y-%m-%d")
        d2 = datetime.strptime(dos, "%Y-%m-%d")
        return round((d2 - d1).days / 365.25, 1)
    except Exception:
        return "nan"


# ═══════════════════════════════════════════════════════════════════════════
# 12. CLASS LABEL NORMALISATION
# ═══════════════════════════════════════════════════════════════════════════

_VALID_LABELS = {"A", "B", "C", "D", "AB", "AC", "AD", "BC", "BD", "NG"}


def normalize_class(label: str) -> str:
    """Normalize vessel-evaluation class label."""
    c = label.strip().upper().replace("+", "")
    return c if c in _VALID_LABELS else "NG"


# ═══════════════════════════════════════════════════════════════════════════
# 13. METADATA EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

_TAG_MAP = {
    "image_orientation":  "0020|0037",
    "sex":                "0010|0040",
    "DOB":                "0010|0030",
    "age":                "0010|1010",
    "DOS":                "0008|0020",
    "study_time":         "0008|0030",
    "series_description": "0008|103e",
    "series_number":      "0020|0011",
    "modality":           "0008|0060",
    "manufacturer":       "0008|0070",
    "manufacturer_model": "0008|1090",
    "institution":        "0008|0080",
    "kvp":                "0018|0060",
    "contrast_agent":     "0018|0010",
    "slice_thickness":    "0018|0050",
}


def extract_metadata(reader: sitk.ImageSeriesReader, extra: Dict) -> Dict:
    """Build a metadata dict from DICOM tags merged with *extra* context."""
    md = dict(extra)
    for key, tag in _TAG_MAP.items():
        md.setdefault(key, get_tag(reader, tag) or "nan")

    md["DOB"] = format_date(md["DOB"])
    md["DOS"] = format_date(md["DOS"])
    md["study_time"] = format_time(md["study_time"])
    md["age_at_scan"] = (
        age_at_scan(md["DOB"], md["DOS"])
        if md["DOB"] != "nan" and md["DOS"] != "nan"
        else "nan"
    )
    return md


# ═══════════════════════════════════════════════════════════════════════════
# 14. NIFTI HEADER METADATA EMBEDDING
# ═══════════════════════════════════════════════════════════════════════════

_NIFTI_TAGS = {
    "PatientID":            "0010|0020",
    "PatientName":          "0010|0010",
    "StudyDate":            "0008|0020",
    "StudyTime":            "0008|0030",
    "SeriesDescription":    "0008|103e",
    "ProtocolName":         "0018|1030",
    "Modality":             "0008|0060",
    "Manufacturer":         "0008|0070",
    "ManufacturerModelName":"0008|1090",
    "InstitutionName":      "0008|0080",
    "SeriesNumber":         "0020|0011",
    "AcquisitionNumber":    "0020|0012",
    "SliceThickness":       "0018|0050",
    "KVP":                  "0018|0060",
    "ContrastBolusAgent":   "0018|0010",
}


def embed_metadata(
    image: sitk.Image, reader: sitk.ImageSeriesReader, context: Dict,
) -> sitk.Image:
    """Copy selected DICOM tags + conversion context into the image metadata."""
    for key, tag in _NIFTI_TAGS.items():
        val = get_tag(reader, tag)
        if val:
            image.SetMetaData(key, val)

    image.SetMetaData("ConversionDate", datetime.now().isoformat())
    image.SetMetaData("DetectedProtocol", context.get("phase", "undefined"))
    image.SetMetaData("AssignedClass", context.get("group", "NG"))
    image.SetMetaData("CaseID", context.get("case_id", ""))
    image.SetMetaData("DatasetType", context.get("dataset_type", "generic"))
    return image


# ═══════════════════════════════════════════════════════════════════════════
# 15. CLASSIFICATION CSV LOADER
# ═══════════════════════════════════════════════════════════════════════════

def load_classifications(csv_path: Optional[Path]) -> Tuple[dict, dict]:
    """Load patient → group and patient → case_id maps from CSV.

    Returns (class_map, case_map).  Both may be empty dicts.
    """
    class_map: dict[str, str] = {}
    case_map: dict[str, str] = {}

    if csv_path is None or not csv_path.exists():
        return class_map, case_map

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return class_map, case_map

    headers = list(rows[0].keys())

    def _find(candidates: list[str]) -> Optional[str]:
        for c in candidates:
            if c in headers:
                return c
        return None

    pid_col = _find(["TCGA ID", "patient_id", "PatientID", "ID"])
    cls_col = _find(["Vessel evaluation", "class", "Class", "Label", "group", "Group"])
    cid_col = _find(["case_id", "CaseID", "Case"])

    if not pid_col or not cls_col:
        return class_map, case_map

    for row in rows:
        pid = row[pid_col].strip()
        class_map[pid] = normalize_class(row[cls_col])
        if cid_col and row.get(cid_col):
            case_map[pid] = row[cid_col].strip()

    return class_map, case_map
