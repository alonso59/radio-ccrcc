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

def _find_manifest(image_folder: Path) -> Optional[Path]:
    """Auto-discover manifest.csv relative to *image_folder*.

    Search order:
      1. image_folder/manifest.csv        (inside nifti/)
      2. image_folder/../manifest.csv      (DatasetID/ level)
    """
    for candidate in [
        image_folder / "manifest.csv",
        image_folder.parent / "manifest.csv",
    ]:
        if candidate.exists():
            return candidate
    return None


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
    manifest_path = _find_manifest(image_folder)
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
