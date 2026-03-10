"""Image processing: masks, kidney separation, crop, resample, validate."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk


# ===================================================================
# Dual-mask generation
# ===================================================================

def generate_dual_masks(
    mask: sitk.Image,
    bbox_labels: List[int],
    mask_labels: List[int],
) -> Tuple[sitk.Image, sitk.Image]:
    """Return (bbox_mask, save_mask) as binary uint8 images."""
    arr = sitk.GetArrayFromImage(mask)

    def _bin(labels):
        img = sitk.GetImageFromArray(np.isin(arr, labels).astype(np.uint8))
        img.CopyInformation(mask)
        return img

    return _bin(bbox_labels), _bin(mask_labels)


# ===================================================================
# Kidney separation
# ===================================================================

def separate_kidneys(bbox_mask: sitk.Image) -> Dict[str, Dict]:
    """Separate L/R kidneys via connected components on *bbox_mask*.

    Returns {side: {"bbox": tuple, "center_slice": int, "voxel_count": int}}.
    """
    arr = sitk.GetArrayFromImage(bbox_mask)
    if arr.max() == 0:
        raise ValueError("bbox_mask is empty")

    cc = sitk.ConnectedComponent(bbox_mask)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)

    comps = sorted(
        [(lbl, stats.GetNumberOfPixels(lbl), stats.GetCentroid(lbl)) for lbl in stats.GetLabels()],
        key=lambda x: x[1],
        reverse=True,
    )[:2]

    if not comps:
        raise ValueError("No kidney components found")

    cc_arr = sitk.GetArrayFromImage(cc)

    if len(comps) == 1:
        lbl, _, centroid = comps[0]
        mid_x = (bbox_mask.GetSize()[0] * bbox_mask.GetSpacing()[0]) / 2
        side = "L" if centroid[0] < mid_x else "R"
        sides = [(side, lbl)]
    else:
        left, right = sorted(comps, key=lambda c: c[2][0])
        sides = [("L", left[0]), ("R", right[0])]

    bboxes: Dict[str, Dict] = {}
    for side, lbl in sides:
        km = ((cc_arr == lbl) & (arr == 1)).astype(np.uint8)
        vc = int(km.sum())
        if vc == 0:
            continue
        km_sitk = sitk.GetImageFromArray(km)
        km_sitk.CopyInformation(bbox_mask)
        bs = sitk.LabelShapeStatisticsImageFilter()
        bs.Execute(km_sitk)
        if bs.GetLabels():
            bb = bs.GetBoundingBox(1)
            bboxes[side] = {"bbox": bb, "center_slice": bb[2] + bb[5] // 2, "voxel_count": vc}
    return bboxes


# ===================================================================
# Minimal VOI Size Enforcement
# ===================================================================

def enforce_min_voi_size(
    start: np.ndarray,
    size: np.ndarray,
    image_size: np.ndarray,
    min_voi_size: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Enforce minimum VOI size with balanced 3-directional edge-aware expansion.

    Strategy: For each axis that is below min_voi_size, try symmetric expansion first.
    If hitting image boundary, redistribute expansion to the opposite direction.
    All expansions are clamped to image bounds.

    Args:
        start: Current crop start position [x, y, z]
        size: Current crop size [x, y, z]
        image_size: Image dimensions [x, y, z] in voxels
        min_voi_size: Target minimum VOI size [x, y, z]

    Returns:
        (new_start, new_size, provenance_dict) where provenance tracks all expansions
    """
    start = np.array(start, dtype=np.int32)
    size = np.array(size, dtype=np.int32)
    image_size = np.array(image_size, dtype=np.int32)
    min_voi_size = np.array(min_voi_size, dtype=np.int32)

    new_start = start.copy()
    new_size = size.copy()
    expansions = {}  # track per-axis expansion

    for axis in range(3):
        deficit = min_voi_size[axis] - size[axis]

        if deficit > 0:
            # Need expansion on this axis
            expand_lower = deficit // 2
            expand_upper = deficit - expand_lower

            # Check lower boundary: if start - expand_lower < 0, we can only expand down by start amount
            if new_start[axis] - expand_lower < 0:
                available_lower = new_start[axis]
                expand_upper += expand_lower - available_lower
                expand_lower = available_lower

            # Check upper boundary
            end_pos = new_start[axis] + size[axis]
            if end_pos + expand_upper > image_size[axis]:
                overshoot = end_pos + expand_upper - image_size[axis]
                expand_lower += overshoot
                expand_upper -= overshoot

            # Final clamping to image bounds
            expand_lower = int(np.clip(expand_lower, 0, new_start[axis]))
            expand_upper = int(
                np.clip(expand_upper, 0, image_size[axis] - (new_start[axis] + size[axis]))
            )

            new_start[axis] -= expand_lower
            new_size[axis] += expand_lower + expand_upper

            expansions[axis] = {
                "deficit": int(deficit),
                "expand_lower": int(expand_lower),
                "expand_upper": int(expand_upper),
                "final_size": int(new_size[axis]),
            }

    provenance = {
        "min_voi_size_target": min_voi_size.tolist(),
        "original_size": size.tolist(),
        "final_size": new_size.tolist(),
        "expansions": expansions,
    }

    return new_start, new_size, provenance


# ===================================================================
# Crop
# ===================================================================

def extract_voi_crop(
    image: sitk.Image,
    seg: sitk.Image,
    save_mask: sitk.Image,
    bbox_info: Dict,
    expansion_mm: float,
    spacing: np.ndarray,
    min_voi_size: Optional[np.ndarray] = None,
) -> Tuple[sitk.Image, sitk.Image, sitk.Image, Dict]:
    """Crop VOI with margin and optional minimum size enforcement.

    Args:
        image: Input CT image
        seg: Input segmentation mask
        save_mask: Save mask (target labels)
        bbox_info: Bounding box info dict with "bbox" key
        expansion_mm: Margin expansion in mm
        spacing: Image spacing [x, y, z] in mm
        min_voi_size: Optional minimum VOI size [x, y, z] in voxels. If provided,
                     will enforce this size with balanced edge-aware expansion.

    Returns (voi_image, voi_seg, voi_save_mask, provenance).
    """
    bbox = bbox_info["bbox"]
    exp_vox = np.ceil(expansion_mm / spacing).astype(int)
    start_arr = np.array([bbox[0], bbox[1], bbox[2]])
    end_arr = start_arr + np.array([bbox[3], bbox[4], bbox[5]])
    start = np.maximum(0, start_arr - exp_vox).astype(int)
    size = np.minimum(np.array(image.GetSize()), end_arr + exp_vox) - start
    size = np.maximum(1, size).astype(int)  # Ensure size is at least 1

    provenance = {
        "bbox_original": list(bbox),
        "crop_start_initial": start.tolist(),
        "crop_size_initial": size.tolist(),
        "expansion_mm": float(expansion_mm),
        "expansion_voxels": exp_vox.tolist(),
    }

    # Apply minimum VOI size enforcement if specified
    if min_voi_size is not None:
        min_voi_size = np.array(min_voi_size, dtype=np.int32)
        start, size, min_enforce_prov = enforce_min_voi_size(
            start, size, np.array(image.GetSize()), min_voi_size
        )
        provenance["min_voi_enforcement"] = min_enforce_prov

    final_start = start.astype(int).tolist()
    final_size = size.astype(int).tolist()
    provenance["crop_start"] = final_start
    provenance["crop_size"] = final_size

    return (
        sitk.RegionOfInterest(image, size=final_size, index=final_start),
        sitk.RegionOfInterest(seg, size=final_size, index=final_start),
        sitk.RegionOfInterest(save_mask, size=final_size, index=final_start),
        provenance,
    )


# ===================================================================
# Resample
# ===================================================================

def _resample(image: sitk.Image, target: np.ndarray, interp: int, default_value: float = 0.0) -> sitk.Image:
    sp = np.array(image.GetSpacing())
    new_size = np.ceil(np.array(image.GetSize()) * (sp / target)).astype(np.uint32).tolist()
    r = sitk.ResampleImageFilter()
    r.SetSize(new_size)
    r.SetTransform(sitk.Transform())
    r.SetOutputOrigin(image.GetOrigin())
    r.SetOutputSpacing(target.tolist())
    r.SetOutputDirection(image.GetDirection())
    r.SetInterpolator(interp)
    r.SetDefaultPixelValue(default_value)
    return r.Execute(image)


def resample_voi(
    image: sitk.Image,
    seg: sitk.Image,
    save_mask: sitk.Image,
    target_spacing: list,
    aniso_thresh: float = 3.0,
) -> Tuple[sitk.Image, sitk.Image, sitk.Image, Dict]:
    """Resample with anisotropy-aware strategy.

    Returns (img, seg, mask, provenance).
    """
    tgt = np.array(target_spacing)
    sp = np.array(image.GetSpacing())
    ratio = float(sp.max() / sp.min())
    is_aniso = ratio > aniso_thresh

    if is_aniso:
        ax = int(np.argmax(sp))
        mid = sp.copy()
        for i in range(3):
            if i != ax:
                mid[i] = tgt[i]
        # BSpline for both steps: handles large upsampling factors (e.g. 3.33→1.0 mm)
        # without aliasing or blurring artifacts that sitkLinear introduces.
        image = _resample(image, mid, sitk.sitkBSpline, default_value=-1000.0)
        image = _resample(image, tgt, sitk.sitkBSpline, default_value=-1000.0)
        method = "two_step_bspline"
    else:
        image = _resample(image, tgt, sitk.sitkBSpline, default_value=-1000.0)
        method = "one_shot_bspline"

    seg = _resample(seg, tgt, sitk.sitkNearestNeighbor, default_value=0.0)
    save_mask = _resample(save_mask, tgt, sitk.sitkNearestNeighbor, default_value=0.0)

    provenance = {
        "original_spacing": sp.tolist(),
        "target_spacing": tgt.tolist(),
        "anisotropic": is_aniso,
        "aniso_ratio": ratio,
        "method": method,
    }

    return image, seg, save_mask, provenance


# ===================================================================
# Validation
# ===================================================================

def validate_voi(
    bbox_mask: sitk.Image,
    save_mask: sitk.Image,
    min_bbox_voxels: int = 0,
    min_save_voxels: int = 0,
) -> Tuple[bool, str]:
    """Validate VOI masks. Checks bbox (tissue) and save (target labels) voxel counts.

    Returns (is_valid, reason).
    """
    bv = int(sitk.GetArrayFromImage(bbox_mask).sum())
    if bv == 0:
        return False, "Empty bbox_mask (no tissue)"
    if min_bbox_voxels > 0 and bv < min_bbox_voxels:
        return False, f"Insufficient tissue ({bv} < {min_bbox_voxels} voxels)"

    sv = int(sitk.GetArrayFromImage(save_mask).sum())
    if sv == 0:
        return False, "Empty save_mask (no target labels)"
    if min_save_voxels > 0 and sv < min_save_voxels:
        return False, f"Insufficient target labels ({sv} < {min_save_voxels} voxels)"
    return True, "OK"


def validate_mask_integrity(
    mask: sitk.Image,
    expected_shape: Tuple[int, ...],
) -> Tuple[bool, str]:
    """Check mask dtype, value range, and shape after resampling.

    Returns (is_valid, reason).
    """
    arr = sitk.GetArrayFromImage(mask)
    if arr.dtype != np.uint8:
        return False, f"Mask dtype {arr.dtype} != uint8"
    unique = np.unique(arr)
    if not np.all(np.isin(unique, [0, 1])):
        return False, f"Mask contains invalid values: {unique}"
    if arr.shape != expected_shape:
        return False, f"Shape mismatch: {arr.shape} != {expected_shape}"
    return True, "OK"


# ===================================================================
# Mask application
# ===================================================================

def apply_mask(image: sitk.Image, mask: sitk.Image, bg: float = -1000.0) -> sitk.Image:
    """Mask image: foreground keeps HU, background set to *bg*."""
    ia = sitk.GetArrayFromImage(image).astype(np.float32)
    ma = sitk.GetArrayFromImage(mask)
    out = np.where(ma > 0, ia, bg).astype(np.float32)
    img = sitk.GetImageFromArray(out)
    img.CopyInformation(image)
    return img


# ===================================================================
# RAS orientation
# ===================================================================

def to_ras(image: sitk.Image, mask: sitk.Image) -> Tuple[sitk.Image, sitk.Image]:
    return sitk.DICOMOrient(image, "RAS"), sitk.DICOMOrient(mask, "RAS")


# ===================================================================
# Tumor metrics
# ===================================================================

TUMOR_LABEL = 2

def compute_tumor_metrics(image: sitk.Image, mask: sitk.Image) -> Dict:
    ia = sitk.GetArrayFromImage(image)
    ma = sitk.GetArrayFromImage(mask)
    tm = ma == TUMOR_LABEL
    tv = int(tm.sum())
    if tv == 0:
        return {"has_tumor": False, "tumor_voxels": 0}
    sp = image.GetSpacing()
    vv = float(np.prod(sp))
    hu = ia[tm]
    return {
        "has_tumor": True,
        "tumor_voxels": tv,
        "tumor_volume_mm3": float(tv * vv),
        "tumor_volume_cm3": float(tv * vv / 1000),
        "mean_hu": float(hu.mean()),
        "std_hu": float(hu.std()),
        "min_hu": float(hu.min()),
        "max_hu": float(hu.max()),
    }
