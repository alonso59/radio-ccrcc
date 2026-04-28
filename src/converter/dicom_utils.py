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
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zipfile import ZipFile

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
                pid_key = normalize_patient_key(pid)
                if re.match(r"^ANONYM-[A-Z0-9]+$", pid_key) or re.match(r"^\d+$", pid_key):
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


def _clean_cell(value: object) -> str:
    """Convert CSV/XLSX cell values to stable string values."""
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none"}:
        return ""
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return text


def normalize_patient_key(raw: object) -> str:
    """Normalize patient identifiers for cross-source matching.

    UKBonn labels use ``Anonym_XXXXXX`` while DICOM headers use
    ``ANONYM-XXXXXX``. This helper is intentionally used for lookups only; the
    converter still preserves the original extracted patient ID in manifests.
    """
    text = _clean_cell(raw)
    if not text:
        return ""

    text = _sanitize(text)
    anonym = re.match(r"(?i)^anonym[-_]?(.+)$", text)
    if anonym:
        token = re.sub(r"^[-_]+", "", anonym.group(1)).upper()
        return f"ANONYM-{token}"
    if re.fullmatch(r"\d+", text):
        return text
    return text.upper()


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
            if re.match(r"(?i)^anonym[-_]", part):
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
# 15. CLASSIFICATION / LABEL LOADERS
# ═══════════════════════════════════════════════════════════════════════════

_UKBONN_LABEL_COLUMNS = {"Anonym", "ID", "Manual Label", "LB", "HB", "SN"}


def _is_xlsx_file(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"PK\x03\x04"
    except OSError:
        return False


def _cell_ref_to_index(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref or "")
    if not match:
        return 0
    idx = 0
    for char in match.group(1):
        idx = idx * 26 + ord(char) - ord("A") + 1
    return idx - 1


def _read_xlsx_rows(path: Path) -> List[dict]:
    """Read the first XLSX worksheet into a list of string dictionaries."""
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as zf:
        shared: List[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", ns):
                shared.append("".join(t.text or "" for t in item.findall(".//a:t", ns)))

        sheet_names = sorted(
            name
            for name in zf.namelist()
            if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
        )
        if not sheet_names:
            return []

        root = ET.fromstring(zf.read(sheet_names[0]))
        table: List[List[str]] = []
        for row in root.findall(".//a:sheetData/a:row", ns):
            values: List[str] = []
            for cell in row.findall("a:c", ns):
                idx = _cell_ref_to_index(cell.attrib.get("r", ""))
                while len(values) <= idx:
                    values.append("")

                cell_type = cell.attrib.get("t")
                value = ""
                if cell_type == "inlineStr":
                    value = "".join(t.text or "" for t in cell.findall(".//a:t", ns))
                else:
                    raw_value = cell.find("a:v", ns)
                    if raw_value is not None and raw_value.text is not None:
                        if cell_type == "s":
                            value = shared[int(raw_value.text)]
                        else:
                            value = raw_value.text
                values[idx] = _clean_cell(value)
            if any(values):
                table.append(values)

    if not table:
        return []

    headers = [_clean_cell(h) for h in table[0]]
    while headers and not headers[-1]:
        headers.pop()
    rows: List[dict] = []
    for values in table[1:]:
        row = {
            header: _clean_cell(values[idx] if idx < len(values) else "")
            for idx, header in enumerate(headers)
            if header
        }
        if any(row.values()):
            rows.append(row)
    return rows


def _read_table_rows(path: Path) -> List[dict]:
    if _is_xlsx_file(path):
        return _read_xlsx_rows(path)
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [
            {str(k): _clean_cell(v) for k, v in row.items() if k is not None}
            for row in csv.DictReader(f)
        ]


def _find_column(candidates: List[str], headers: List[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in headers:
            return candidate
    return None


def _looks_like_ukbonn_labels(rows: List[dict]) -> bool:
    return bool(rows) and _UKBONN_LABEL_COLUMNS.issubset(set(rows[0].keys()))


def load_ukbonn_labels(csv_path: Optional[Path]) -> Tuple[dict, dict]:
    """Load UKBonn label metadata keyed by normalized patient aliases.

    The source file may be a real CSV or an XLSX workbook with a .csv suffix.
    Labels are metadata only; they do not update TCGA-style group values.
    """
    label_metadata: dict[str, dict] = {}
    summary: dict = {
        "label_file": str(csv_path) if csv_path else "",
        "labels_loaded": 0,
        "label_keys_loaded": 0,
        "duplicate_label_keys": [],
        "label_rows": [],
    }

    if csv_path is None or not csv_path.exists():
        return label_metadata, summary

    rows = _read_table_rows(csv_path)
    if not _looks_like_ukbonn_labels(rows):
        return label_metadata, summary

    duplicate_keys: set[str] = set()
    for row_idx, row in enumerate(rows, start=1):
        record = {
            "ukb_anonym": _clean_cell(row.get("Anonym")),
            "ukb_id": _clean_cell(row.get("ID")),
            "manual_label": _clean_cell(row.get("Manual Label")),
            "lb": _clean_cell(row.get("LB")),
            "hb": _clean_cell(row.get("HB")),
            "sn": _clean_cell(row.get("SN")),
            "label_source": Path(csv_path).name,
            "_label_row_id": f"ukbonn:{row_idx}",
        }
        summary["label_rows"].append({
            "row": row_idx,
            "ukb_anonym": record["ukb_anonym"],
            "ukb_id": record["ukb_id"],
            "manual_label": record["manual_label"],
        })

        aliases = [
            normalize_patient_key(record["ukb_anonym"]),
            normalize_patient_key(record["ukb_id"]),
        ]
        for alias in [a for a in aliases if a]:
            existing = label_metadata.get(alias)
            if existing and existing.get("_label_row_id") != record["_label_row_id"]:
                duplicate_keys.add(alias)
                continue
            label_metadata[alias] = record

    summary["labels_loaded"] = len(rows)
    summary["label_keys_loaded"] = len(label_metadata)
    summary["duplicate_label_keys"] = sorted(duplicate_keys)
    return label_metadata, summary


def load_classifications(csv_path: Optional[Path]) -> Tuple[dict, dict]:
    """Load patient → group and patient → case_id maps from CSV/XLSX.

    Returns (class_map, case_map).  Both may be empty dicts.
    """
    class_map: dict[str, str] = {}
    case_map: dict[str, str] = {}

    if csv_path is None or not csv_path.exists():
        return class_map, case_map

    rows = _read_table_rows(csv_path)
    if not rows:
        return class_map, case_map

    headers = list(rows[0].keys())

    pid_col = _find_column(["TCGA ID", "patient_id", "PatientID", "ID"], headers)
    cls_col = _find_column(["Vessel evaluation", "class", "Class", "Label", "group", "Group"], headers)
    cid_col = _find_column(["case_id", "CaseID", "Case"], headers)

    if not pid_col or not cls_col:
        return class_map, case_map

    for row in rows:
        pid = _clean_cell(row.get(pid_col))
        if not pid:
            continue
        class_map[pid] = normalize_class(_clean_cell(row.get(cls_col)))
        if cid_col and _clean_cell(row.get(cid_col)):
            case_map[pid] = _clean_cell(row.get(cid_col))

    return class_map, case_map


def load_conversion_labels(
    csv_path: Optional[Path],
    dataset_type: str = "generic",
) -> Tuple[dict, dict, dict, dict]:
    """Load existing class maps plus optional UKBonn label metadata."""
    class_map, case_map = load_classifications(csv_path)
    label_metadata: dict = {}
    label_summary: dict = {
        "label_file": str(csv_path) if csv_path else "",
        "labels_loaded": 0,
        "label_keys_loaded": 0,
        "duplicate_label_keys": [],
        "label_rows": [],
    }

    if csv_path is None or not csv_path.exists():
        return class_map, case_map, label_metadata, label_summary

    try:
        rows = _read_table_rows(csv_path)
    except Exception:
        return class_map, case_map, label_metadata, label_summary

    if dataset_type == "ukbonn" or _looks_like_ukbonn_labels(rows):
        label_metadata, label_summary = load_ukbonn_labels(csv_path)

    return class_map, case_map, label_metadata, label_summary
