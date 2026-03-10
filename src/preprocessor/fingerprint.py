"""Optional fingerprint computation from saved VOI outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


# ------------------------------------------------------------------
# Intensity collection
# ------------------------------------------------------------------

def _collect_bbox_intensities(
    img_arr: np.ndarray,
    bbox_arr: np.ndarray,
    hu_lo: float,
    hu_hi: float,
) -> np.ndarray:
    """Return filtered intensity values from the bbox region."""
    vals = img_arr[bbox_arr > 0].flatten()
    mask = (vals >= hu_lo) & (vals <= hu_hi)
    return vals[mask].astype(np.float32)


# ------------------------------------------------------------------
# Statistics helpers
# ------------------------------------------------------------------

def _intensity_stats(arr: np.ndarray) -> Dict:
    if arr.size == 0:
        return {}
    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "p0.5": float(np.percentile(arr, 0.5)),
        "p5": float(np.percentile(arr, 5)),
        "p25": q25,
        "p75": q75,
        "p95": float(np.percentile(arr, 95)),
        "p99.5": float(np.percentile(arr, 99.5)),
        "iqr": q75 - q25,
    }


def _shape_stats(shapes: List) -> Dict:
    if not shapes:
        return {}
    a = np.array(shapes)
    return {
        "min": a.min(axis=0).tolist(),
        "max": a.max(axis=0).tolist(),
        "mean": a.mean(axis=0).tolist(),
        "median": np.median(a, axis=0).tolist(),
    }


# ------------------------------------------------------------------
# Scan saved outputs on disk
# ------------------------------------------------------------------

def _scan_vois(output_dir: Path):
    """Yield (mask_path, image_path_or_None) for every .npy in mask/."""
    mask_dir = output_dir / "mask"
    img_dir = output_dir / "images"
    if not mask_dir.exists():
        return
    for mp in mask_dir.rglob("*.npy"):
        rel = mp.relative_to(mask_dir)
        ip = img_dir / rel
        yield mp, (ip if ip.exists() else None)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def compute_fingerprint(
    output_dir: Path,
    bbox_labels: List[int],
    hu_range: tuple = (-200, 300),
    target_spacing: list | None = None,
) -> Dict:
    output_dir = Path(output_dir)
    hu_lo, hu_hi = hu_range

    all_intensities: List[np.ndarray] = []
    shapes: List[list] = []
    total_vox = 0
    filtered_vox = 0

    vois = list(_scan_vois(output_dir))
    if not vois:
        raise FileNotFoundError(f"No mask .npy files found under {output_dir / 'mask'}")

    sp = target_spacing or [1.0, 1.0, 1.0]

    for mask_path, img_path in tqdm(vois, desc="Fingerprint", unit="voi"):
        mask_arr = np.load(mask_path)
        bbox_arr = np.isin(mask_arr, bbox_labels).astype(np.uint8)
        shapes.append(list(mask_arr.shape))

        if img_path is None:
            continue

        img_arr = np.load(img_path)
        bbox_vals = img_arr[bbox_arr > 0].flatten()
        total_vox += len(bbox_vals)
        filt = _collect_bbox_intensities(img_arr, bbox_arr, hu_lo, hu_hi)
        filtered_vox += len(filt)
        all_intensities.append(filt)

    intensities = np.concatenate(all_intensities) if all_intensities else np.array([], dtype=np.float32)
    i_stats = _intensity_stats(intensities)

    fingerprint = {
        "foreground_intensity": {"channel_0": i_stats},
        "normalization": {
            "method": "iqr",
            "p25": i_stats.get("p25"),
            "p75": i_stats.get("p75"),
            "median": i_stats.get("median"),
            "iqr": i_stats.get("iqr"),
        },
        "shape_statistics": _shape_stats(shapes),
        "voxel_info": {
            "hu_range": [hu_lo, hu_hi],
            "total_voxels": total_vox,
            "filtered_voxels": filtered_vox,
        },
        "num_vois": len(vois),
        "target_spacing": sp,
    }

    fp_path = output_dir / "dataset_fingerprint.json"
    with open(fp_path, "w") as f:
        json.dump(fingerprint, f, indent=2)
    print(f"Saved fingerprint → {fp_path}")

    return fingerprint
