from __future__ import annotations

"""Lesion-centric fingerprint computation from saved side-specific VOI outputs.

This module is intended for datasets where masks and images are already saved as
laterality-specific VOIs, typically with filenames ending in ``_L.npy`` or
``_R.npy`` under:

- ``output_dir/mask/**/*.npy``
- ``output_dir/images/**/*.npy``

Key design points:
- Each file pair is treated as one VOI, not as a full original CT volume.
- Lesion centroids and bounding boxes are reported in VOI-local coordinates.
- The module preserves the original normalization-oriented fingerprint while
  adding lesion-level metadata needed for a future lesion-centric patch sampler.

Outputs:
- ``dataset_fingerprint.json``
- ``lesion_catalog.jsonl``
- ``sampling_profile.json``
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy import ndimage
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class FingerprintConfig:
    """Configuration for offline fingerprint and lesion-catalog generation."""

    tumor_label: int = 2
    kidney_label: int = 1
    bbox_labels: list[int] = field(default_factory=lambda: [1, 2])
    hu_range: tuple[float, float] = (-200.0, 300.0)
    target_spacing: list[float] | None = None
    size_bin_percentiles: list[float] = field(default_factory=lambda: [25.0, 50.0, 75.0])
    balancing_alpha: float = 0.5
    min_lesion_voxels: int = 5
    patch_size_candidates: list[tuple[int, int, int]] | None = None
    contact_dilation_iters: int = 1


# -----------------------------------------------------------------------------
# Helpers: identifiers and scanning
# -----------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class VoiRecord:
    """Metadata for one saved VOI pair on disk."""

    case_id: str
    voi_id: str
    laterality: str | None
    mask_path: Path
    image_path: Path | None
    relative_mask_path: str
    relative_image_path: str | None


@dataclass(slots=True)
class VoiAnalysis:
    """Per-VOI analysis result."""

    record: VoiRecord
    voi_shape: list[int]
    warnings: list[str]
    bbox_voxel_count: int
    filtered_bbox_voxel_count: int
    raw_intensities: np.ndarray
    filtered_intensities: np.ndarray
    lesion_entries: list[dict[str, Any]]
    has_tumor: bool


def _parse_laterality(stem: str) -> tuple[str, str | None]:
    """Parse laterality from filenames like case123_L.npy or case123_R.npy."""
    upper_stem = stem.upper()
    if upper_stem.endswith("_L"):
        return stem[:-2], "L"
    if upper_stem.endswith("_R"):
        return stem[:-2], "R"
    return stem, None


def _scan_vois(output_dir: Path) -> Iterable[VoiRecord]:
    """Yield side-specific VOI records from ``mask/`` and matching ``images/``."""
    mask_dir = output_dir / "mask"
    image_dir = output_dir / "images"
    if not mask_dir.exists():
        return

    for mask_path in sorted(mask_dir.rglob("*.npy")):
        rel = mask_path.relative_to(mask_dir)
        image_path = image_dir / rel
        image_path = image_path if image_path.exists() else None

        stem_without_side, laterality = _parse_laterality(mask_path.stem)
        parent_prefix = "__".join(mask_path.relative_to(mask_dir).parent.parts)
        case_id = stem_without_side if not parent_prefix else f"{parent_prefix}__{stem_without_side}"
        voi_id = str(rel.with_suffix(""))

        yield VoiRecord(
            case_id=case_id,
            voi_id=voi_id,
            laterality=laterality,
            mask_path=mask_path,
            image_path=image_path,
            relative_mask_path=str(rel),
            relative_image_path=str(rel) if image_path is not None else None,
        )


# -----------------------------------------------------------------------------
# Helpers: numerics and JSON safety
# -----------------------------------------------------------------------------


def _to_float_list(values: Sequence[float | int]) -> list[float]:
    return [float(v) for v in values]


def _to_int_list(values: Sequence[float | int]) -> list[int]:
    return [int(v) for v in values]


def _safe_spacing(config: FingerprintConfig, ndim: int = 3) -> list[float] | None:
    if config.target_spacing is None:
        return None
    if len(config.target_spacing) != ndim:
        raise ValueError(f"target_spacing must have length {ndim}, got {config.target_spacing!r}")
    return [float(v) for v in config.target_spacing]


def _shape_stats(shapes: list[list[int]]) -> dict[str, Any]:
    if not shapes:
        return {}
    arr = np.asarray(shapes, dtype=np.int64)
    return {
        "min": arr.min(axis=0).astype(int).tolist(),
        "max": arr.max(axis=0).astype(int).tolist(),
        "mean": arr.mean(axis=0).astype(float).tolist(),
        "median": np.median(arr, axis=0).astype(float).tolist(),
    }


def _percentile_dict(values: Sequence[float], percentiles: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {}
    result: dict[str, float] = {}
    for p in percentiles:
        key = f"p{str(p).replace('.', '_')}"
        result[key] = float(np.percentile(arr, p))
    return result


def _intensity_stats(arr: np.ndarray) -> dict[str, Any]:
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
        "p0_5": float(np.percentile(arr, 0.5)),
        "p5": float(np.percentile(arr, 5)),
        "p25": q25,
        "p75": q75,
        "p95": float(np.percentile(arr, 95)),
        "p99_5": float(np.percentile(arr, 99.5)),
        "iqr": q75 - q25,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize_probs(counter: dict[str, int]) -> dict[str, float]:
    total = float(sum(counter.values()))
    if total <= 0.0:
        return {k: 0.0 for k in counter}
    return {k: float(v) / total for k, v in counter.items()}


# -----------------------------------------------------------------------------
# Helpers: lesion geometry
# -----------------------------------------------------------------------------


def _collect_raw_intensities(
    img_arr: np.ndarray,
    bbox_arr: np.ndarray,
) -> np.ndarray:
    """Return all foreground voxel values without any HU restriction."""
    return img_arr[bbox_arr > 0].reshape(-1).astype(np.float32)


def _collect_windowed_intensities(
    img_arr: np.ndarray,
    bbox_arr: np.ndarray,
    hu_lo: float,
    hu_hi: float,
) -> np.ndarray:
    """Return foreground voxel values clipped to [hu_lo, hu_hi] for normalization stats."""
    vals = img_arr[bbox_arr > 0].reshape(-1)
    return np.clip(vals, hu_lo, hu_hi).astype(np.float32)


def _component_bbox(component_mask: np.ndarray) -> tuple[list[int], list[int], list[int]]:
    coords = np.argwhere(component_mask)
    mins = coords.min(axis=0)
    maxs_inclusive = coords.max(axis=0)
    maxs_exclusive = maxs_inclusive + 1
    size = maxs_exclusive - mins
    return mins.astype(int).tolist(), maxs_exclusive.astype(int).tolist(), size.astype(int).tolist()


def _compute_kidney_contact_ratio(
    lesion_mask: np.ndarray,
    kidney_mask: np.ndarray,
    dilation_iters: int,
) -> tuple[int, float]:
    """Approximate tumor-kidney contact using a 1-voxel shell around the lesion."""
    if not lesion_mask.any() or not kidney_mask.any():
        return 0, 0.0

    structure = np.ones((3, 3, 3), dtype=bool)
    dilated = ndimage.binary_dilation(lesion_mask, structure=structure, iterations=max(1, dilation_iters))
    shell = np.logical_and(dilated, np.logical_not(lesion_mask))
    shell_count = int(shell.sum())
    if shell_count == 0:
        return 0, 0.0

    contact_vox = int(np.logical_and(shell, kidney_mask).sum())
    return contact_vox, float(contact_vox) / float(shell_count)


def _compute_lesion_entries(
    mask_arr: np.ndarray,
    record: VoiRecord,
    config: FingerprintConfig,
) -> tuple[list[dict[str, Any]], list[str], bool]:
    """Compute lesion-level metadata for one VOI."""
    warnings: list[str] = []
    tumor_mask = mask_arr == int(config.tumor_label)
    kidney_mask = mask_arr == int(config.kidney_label)

    if tumor_mask.ndim != 3:
        raise ValueError(f"Expected 3D mask for {record.mask_path}, got shape {mask_arr.shape}")

    if not tumor_mask.any():
        return [], warnings, False

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    cc, num_components = ndimage.label(tumor_mask, structure=structure)
    spacing = _safe_spacing(config)
    voxel_volume_mm3 = float(np.prod(spacing)) if spacing is not None else None

    lesion_entries: list[dict[str, Any]] = []
    kept_component_ids: list[int] = []
    skipped_small = 0

    for component_id in range(1, num_components + 1):
        lesion_mask = cc == component_id
        volume_vox = int(lesion_mask.sum())
        if volume_vox < int(config.min_lesion_voxels):
            skipped_small += 1
            continue

        kept_component_ids.append(component_id)
        bbox_min_vox, bbox_max_vox, bbox_size_vox = _component_bbox(lesion_mask)
        centroid_vox = ndimage.center_of_mass(lesion_mask.astype(np.uint8))
        centroid_vox_list = _to_float_list(centroid_vox)

        volume_mm3 = float(volume_vox) * voxel_volume_mm3 if voxel_volume_mm3 is not None else None
        bbox_size_mm = (
            [float(s) * float(sp) for s, sp in zip(bbox_size_vox, spacing)]
            if spacing is not None
            else None
        )
        centroid_mm = (
            [float(c) * float(sp) for c, sp in zip(centroid_vox_list, spacing)]
            if spacing is not None
            else None
        )

        kidney_overlap_vox = int(np.logical_and(lesion_mask, kidney_mask).sum())
        kidney_contact_vox, kidney_contact_ratio = _compute_kidney_contact_ratio(
            lesion_mask=lesion_mask,
            kidney_mask=kidney_mask,
            dilation_iters=config.contact_dilation_iters,
        )

        lesion_entry = {
            "case_id": record.case_id,
            "voi_id": record.voi_id,
            "laterality": record.laterality if record.laterality is not None else "unknown",
            "mask_path": record.relative_mask_path,
            "image_path": record.relative_image_path,
            "lesion_id": len(lesion_entries),
            "num_lesions_in_voi": 0,
            "voi_shape": _to_int_list(mask_arr.shape),
            "coordinates_reference": "voi_local",
            "full_volume_coordinates_available": False,
            "volume_vox": volume_vox,
            "volume_mm3": volume_mm3,
            "centroid_vox": centroid_vox_list,
            "centroid_mm": centroid_mm,
            "bbox_min_vox": bbox_min_vox,
            "bbox_max_vox": bbox_max_vox,
            "bbox_size_vox": bbox_size_vox,
            "bbox_size_mm": bbox_size_mm,
            "max_extent_vox": int(max(bbox_size_vox)),
            "max_extent_mm": float(max(bbox_size_mm)) if bbox_size_mm is not None else None,
            "kidney_overlap_vox": kidney_overlap_vox,
            "kidney_contact_vox": kidney_contact_vox,
            "kidney_contact_ratio": kidney_contact_ratio,
            "size_bin": None,
            "warnings": [],
        }
        lesion_entries.append(lesion_entry)

    num_kept = len(lesion_entries)
    for lesion_entry in lesion_entries:
        lesion_entry["num_lesions_in_voi"] = num_kept

    if skipped_small > 0:
        warnings.append(
            f"Skipped {skipped_small} connected component(s) smaller than min_lesion_voxels={config.min_lesion_voxels}."
        )
    if not kept_component_ids:
        warnings.append("Tumor voxels were present, but all connected components were below min_lesion_voxels.")

    for lesion_entry in lesion_entries:
        lesion_entry["warnings"] = list(warnings)

    return lesion_entries, warnings, True


# -----------------------------------------------------------------------------
# Main analysis
# -----------------------------------------------------------------------------


def _analyze_single_voi(record: VoiRecord, config: FingerprintConfig) -> VoiAnalysis:
    warnings: list[str] = []
    mask_arr = np.load(record.mask_path)
    if mask_arr.ndim == 4 and mask_arr.shape[0] == 1:
        mask_arr = mask_arr[0]
    if mask_arr.ndim != 3:
        raise ValueError(f"Expected 3D mask array for {record.mask_path}, got shape {mask_arr.shape}")

    bbox_arr = np.isin(mask_arr, config.bbox_labels).astype(np.uint8)
    bbox_voxel_count = int(bbox_arr.sum())

    raw_intensities = np.array([], dtype=np.float32)
    filtered_intensities = np.array([], dtype=np.float32)
    filtered_bbox_voxel_count = 0

    if record.image_path is None:
        warnings.append("Image file missing; skipped intensity extraction.")
    else:
        img_arr = np.load(record.image_path)
        if img_arr.ndim == 4 and img_arr.shape[0] == 1:
            img_arr = img_arr[0]
        if img_arr.shape != mask_arr.shape:
            warnings.append(
                f"Image/mask shape mismatch: image {tuple(img_arr.shape)} vs mask {tuple(mask_arr.shape)}; "
                "skipped intensity extraction."
            )
        else:
            raw_intensities = _collect_raw_intensities(
                img_arr=img_arr,
                bbox_arr=bbox_arr,
            )
            filtered_intensities = _collect_windowed_intensities(
                img_arr=img_arr,
                bbox_arr=bbox_arr,
                hu_lo=float(config.hu_range[0]),
                hu_hi=float(config.hu_range[1]),
            )
            filtered_bbox_voxel_count = int(filtered_intensities.size)

    lesion_entries, lesion_warnings, has_tumor = _compute_lesion_entries(mask_arr, record, config)
    warnings.extend(lesion_warnings)

    if warnings:
        for lesion_entry in lesion_entries:
            existing = lesion_entry.get("warnings", [])
            lesion_entry["warnings"] = sorted(set(existing + warnings))

    return VoiAnalysis(
        record=record,
        voi_shape=_to_int_list(mask_arr.shape),
        warnings=warnings,
        bbox_voxel_count=bbox_voxel_count,
        filtered_bbox_voxel_count=filtered_bbox_voxel_count,
        raw_intensities=raw_intensities,
        filtered_intensities=filtered_intensities,
        lesion_entries=lesion_entries,
        has_tumor=has_tumor,
    )


def compute_lesion_catalog(
    output_dir: Path,
    config: FingerprintConfig,
) -> tuple[list[dict[str, Any]], list[VoiAnalysis]]:
    """Build a lesion catalog from saved side-specific VOIs."""
    output_dir = Path(output_dir)
    voi_records = list(_scan_vois(output_dir))
    if not voi_records:
        raise FileNotFoundError(f"No mask .npy files found under {output_dir / 'mask'}")

    analyses: list[VoiAnalysis] = []
    lesion_catalog: list[dict[str, Any]] = []

    for record in tqdm(voi_records, desc="Analyzing VOIs", unit="voi"):
        analysis = _analyze_single_voi(record, config)
        analyses.append(analysis)
        lesion_catalog.extend(analysis.lesion_entries)

    return lesion_catalog, analyses


def _assign_size_bins(
    lesion_catalog: list[dict[str, Any]],
    config: FingerprintConfig,
) -> tuple[list[dict[str, Any]], dict[str, float], list[dict[str, Any]]]:
    """Assign percentile-based lesion size bins using mm³ when available."""
    if not lesion_catalog:
        return lesion_catalog, {}, []

    values: list[float] = []
    use_mm3 = all(entry.get("volume_mm3") is not None for entry in lesion_catalog)
    key = "volume_mm3" if use_mm3 else "volume_vox"

    for entry in lesion_catalog:
        values.append(float(entry[key]))

    percentiles = sorted(float(p) for p in config.size_bin_percentiles)
    cut_values = [float(np.percentile(values, p)) for p in percentiles]

    # Deduplicate monotonic cuts to avoid empty or invalid bins in small datasets.
    unique_cuts: list[float] = []
    for cut in cut_values:
        if not unique_cuts or cut > unique_cuts[-1]:
            unique_cuts.append(cut)

    default_labels = ["tiny", "small", "medium", "large", "huge"]
    num_bins = len(unique_cuts) + 1
    labels = default_labels[:num_bins] if num_bins <= len(default_labels) else [f"bin_{i}" for i in range(num_bins)]

    def assign(value: float) -> str:
        for idx, cut in enumerate(unique_cuts):
            if value <= cut:
                return labels[idx]
        return labels[-1]

    for entry, value in zip(lesion_catalog, values):
        entry["size_bin"] = assign(value)

    thresholds = {f"p{str(p).replace('.', '_')}": float(v) for p, v in zip(percentiles, cut_values)}
    size_bin_defs: list[dict[str, Any]] = []
    lower = None
    for idx, label in enumerate(labels):
        upper = unique_cuts[idx] if idx < len(unique_cuts) else None
        size_bin_defs.append(
            {
                "name": label,
                "metric": key,
                "min_inclusive": lower,
                "max_inclusive": upper,
            }
        )
        lower = upper

    return lesion_catalog, thresholds, size_bin_defs


def compute_dataset_fingerprint(
    analyses: Sequence[VoiAnalysis],
    lesion_catalog: Sequence[dict[str, Any]],
    config: FingerprintConfig,
) -> dict[str, Any]:
    """Compute normalization-oriented dataset summary plus lesion counts."""
    all_raw = [a.raw_intensities for a in analyses if a.raw_intensities.size > 0]
    all_windowed = [a.filtered_intensities for a in analyses if a.filtered_intensities.size > 0]

    raw_concat = np.concatenate(all_raw) if all_raw else np.array([], dtype=np.float32)
    windowed_concat = np.concatenate(all_windowed) if all_windowed else np.array([], dtype=np.float32)

    raw_stats = _intensity_stats(raw_concat)
    windowed_stats = _intensity_stats(windowed_concat)

    voi_shapes = [analysis.voi_shape for analysis in analyses]
    total_bbox_voxels = int(sum(analysis.bbox_voxel_count for analysis in analyses))
    filtered_bbox_voxels = int(sum(analysis.filtered_bbox_voxel_count for analysis in analyses))
    spacing = _safe_spacing(config)

    return {
        "version": 3,
        "foreground_intensity": {"channel_0": raw_stats},
        "windowed_intensity": {"channel_0": windowed_stats},
        "normalization": {
            "method": "window_rescale",
            "window_min": float(config.hu_range[0]),
            "window_max": float(config.hu_range[1]),
            "output_min": -1.0,
            "output_max": 1.0,
        },
        "voi_shape_statistics": _shape_stats(voi_shapes),
        "voxel_info": {
            "hu_range": [float(config.hu_range[0]), float(config.hu_range[1])],
            "total_bbox_voxels": total_bbox_voxels,
            "filtered_bbox_voxels": filtered_bbox_voxels,
        },
        "num_vois": len(analyses),
        "num_vois_with_tumor": int(sum(1 for analysis in analyses if analysis.has_tumor)),
        "num_lesions": len(lesion_catalog),
        "tumor_label": int(config.tumor_label),
        "kidney_label": int(config.kidney_label),
        "target_spacing": spacing,
        "assumptions": {
            "voi_files_are_side_specific": True,
            "coordinates_reference": "voi_local",
            "full_volume_coordinates_available": False,
            "target_spacing_is_assumed_uniform_if_provided": spacing is not None,
        },
        "notes": [
            "Each .npy pair is treated as a side-specific VOI, not as a full original CT volume.",
            "Lesion centroids and bounding boxes are reported in VOI-local coordinates.",
            "VOI shape statistics describe saved VOI tensor sizes, not lesion extents.",
            "foreground_intensity: stats over full raw voxel range (informational).",
            "windowed_intensity: stats after clipping to hu_range (used by dataloader for normalization).",
        ],
    }


def _suggest_patch_budget(bin_name: str) -> int:
    mapping = {
        "tiny": 2,
        "small": 2,
        "medium": 3,
        "large": 4,
        "huge": 5,
    }
    return mapping.get(bin_name, 3)


def _suggest_role_priors(bin_name: str) -> dict[str, float]:
    priors = {
        "tiny": {"center": 0.85, "border": 0.10, "context": 0.05},
        "small": {"center": 0.65, "border": 0.25, "context": 0.10},
        "medium": {"center": 0.50, "border": 0.30, "context": 0.20},
        "large": {"center": 0.40, "border": 0.35, "context": 0.25},
        "huge": {"center": 0.30, "border": 0.40, "context": 0.30},
    }
    return priors.get(bin_name, {"center": 0.5, "border": 0.3, "context": 0.2})


def _heuristic_patch_recommendations(
    lesion_catalog: Sequence[dict[str, Any]],
    size_bins: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Provide patch-size guidance without assuming fixed patch values."""
    if not lesion_catalog:
        return {}

    result: dict[str, Any] = {}
    for size_bin in size_bins:
        name = size_bin["name"]
        lesions = [entry for entry in lesion_catalog if entry.get("size_bin") == name]
        if not lesions:
            continue

        use_mm = all(entry.get("bbox_size_mm") is not None for entry in lesions)
        bbox_key = "bbox_size_mm" if use_mm else "bbox_size_vox"
        extents = np.asarray([entry[bbox_key] for entry in lesions], dtype=np.float64)
        p75_extent = np.percentile(extents, 75, axis=0)
        p90_extent = np.percentile(extents, 90, axis=0)

        # Simple context-aware heuristic: lesion bbox percentile plus margin.
        # This is guidance only; the downstream sampler remains free to choose.
        margin = 12.0 if use_mm else 12.0
        result[name] = {
            "units": "mm" if use_mm else "vox",
            "bbox_extent_p75": p75_extent.astype(float).tolist(),
            "bbox_extent_p90": p90_extent.astype(float).tolist(),
            "recommended_min_patch_extent": (p75_extent + 2.0 * margin).astype(float).tolist(),
            "recommended_context_patch_extent": (p90_extent + 2.0 * margin).astype(float).tolist(),
        }
    return result


def _evaluate_patch_size_candidates(
    lesion_catalog: Sequence[dict[str, Any]],
    patch_size_candidates: Sequence[tuple[int, int, int]],
) -> dict[str, Any]:
    """Evaluate optional candidate patch sizes against lesion bbox coverage."""
    if not lesion_catalog or not patch_size_candidates:
        return {}

    result: dict[str, Any] = {}
    for candidate in patch_size_candidates:
        candidate_arr = np.asarray(candidate, dtype=np.float64)
        cover_flags = []
        for lesion in lesion_catalog:
            bbox = np.asarray(lesion["bbox_size_vox"], dtype=np.float64)
            cover_flags.append(bool(np.all(candidate_arr >= bbox)))

        result[str(tuple(int(v) for v in candidate))] = {
            "bbox_full_coverage_rate": float(np.mean(cover_flags)) if cover_flags else 0.0,
            "num_lesions": len(cover_flags),
        }
    return result


def compute_sampling_profile(
    lesion_catalog: list[dict[str, Any]],
    config: FingerprintConfig,
) -> dict[str, Any]:
    """Compute a lesion-size-driven sampling profile for a future sampler."""
    lesion_catalog, size_thresholds, size_bin_defs = _assign_size_bins(lesion_catalog, config)

    if not lesion_catalog:
        return {
            "version": 1,
            "num_lesions": 0,
            "size_bin_thresholds": size_thresholds,
            "size_bins": size_bin_defs,
            "target_spacing": _safe_spacing(config),
            "patch_size_candidates": config.patch_size_candidates,
        }

    have_mm3 = all(entry.get("volume_mm3") is not None for entry in lesion_catalog)
    volume_key = "volume_mm3" if have_mm3 else "volume_vox"
    extent_key = "max_extent_mm" if all(entry.get("max_extent_mm") is not None for entry in lesion_catalog) else "max_extent_vox"
    bbox_key = "bbox_size_mm" if all(entry.get("bbox_size_mm") is not None for entry in lesion_catalog) else "bbox_size_vox"

    volumes = [float(entry[volume_key]) for entry in lesion_catalog]
    max_extents = [float(entry[extent_key]) for entry in lesion_catalog]
    bbox_sizes = np.asarray([entry[bbox_key] for entry in lesion_catalog], dtype=np.float64)

    size_bin_counts: dict[str, int] = {}
    for entry in lesion_catalog:
        size_bin = str(entry["size_bin"])
        size_bin_counts[size_bin] = size_bin_counts.get(size_bin, 0) + 1

    empirical_probs = _normalize_probs(size_bin_counts)
    uniform_prob = 1.0 / float(len(size_bin_counts))
    alpha = float(np.clip(config.balancing_alpha, 0.0, 1.0))
    target_probs = {
        size_bin: (1.0 - alpha) * empirical_probs[size_bin] + alpha * uniform_prob
        for size_bin in size_bin_counts
    }

    patch_budget_by_bin = {size_bin: _suggest_patch_budget(size_bin) for size_bin in size_bin_counts}
    role_priors_by_bin = {size_bin: _suggest_role_priors(size_bin) for size_bin in size_bin_counts}

    return {
        "version": 1,
        "num_lesions": len(lesion_catalog),
        "volume_metric": volume_key,
        "extent_metric": extent_key,
        "bbox_metric": bbox_key,
        "target_spacing": _safe_spacing(config),
        "size_bin_thresholds": size_thresholds,
        "size_bins": size_bin_defs,
        "lesion_volume_percentiles": _percentile_dict(volumes, [5, 10, 25, 50, 75, 90, 95]),
        "lesion_max_extent_percentiles": _percentile_dict(max_extents, [5, 10, 25, 50, 75, 90, 95]),
        "lesion_bbox_extent_percentiles": {
            "p25": np.percentile(bbox_sizes, 25, axis=0).astype(float).tolist(),
            "p50": np.percentile(bbox_sizes, 50, axis=0).astype(float).tolist(),
            "p75": np.percentile(bbox_sizes, 75, axis=0).astype(float).tolist(),
            "p90": np.percentile(bbox_sizes, 90, axis=0).astype(float).tolist(),
            "units": "mm" if bbox_key.endswith("_mm") else "vox",
        },
        "size_bin_counts": size_bin_counts,
        "empirical_size_bin_probabilities": empirical_probs,
        "target_size_bin_probabilities": target_probs,
        "balancing_alpha": alpha,
        "suggested_patch_budget_by_size_bin": patch_budget_by_bin,
        "suggested_patch_role_priors_by_size_bin": role_priors_by_bin,
        "heuristic_patch_recommendations": _heuristic_patch_recommendations(lesion_catalog, size_bin_defs),
        "patch_size_candidates": [list(candidate) for candidate in config.patch_size_candidates] if config.patch_size_candidates else None,
        "patch_size_candidate_evaluation": _evaluate_patch_size_candidates(
            lesion_catalog=lesion_catalog,
            patch_size_candidates=config.patch_size_candidates or [],
        ),
        "notes": [
            "This profile is derived from lesion statistics inside saved side-specific VOIs.",
            "The future sampler should sample by lesion and role (center/border/context), not by raw subject tumor burden.",
        ],
    }


def save_outputs(
    output_dir: Path,
    dataset_fingerprint: dict[str, Any],
    lesion_catalog: Sequence[dict[str, Any]],
    sampling_profile: dict[str, Any],
) -> None:
    """Persist the three output artifacts to disk."""
    output_dir = Path(output_dir)
    _write_json(output_dir / "dataset_fingerprint.json", dataset_fingerprint)
    _write_jsonl(output_dir / "lesion_catalog.jsonl", lesion_catalog)
    _write_json(output_dir / "sampling_profile.json", sampling_profile)


def run_fingerprint_pipeline(
    output_dir: Path,
    config: FingerprintConfig | None = None,
) -> dict[str, Any]:
    """Run the full lesion-centric fingerprint pipeline and save outputs."""
    config = config or FingerprintConfig()
    output_dir = Path(output_dir)

    lesion_catalog, analyses = compute_lesion_catalog(output_dir=output_dir, config=config)
    sampling_profile = compute_sampling_profile(lesion_catalog=lesion_catalog, config=config)
    dataset_fingerprint = compute_dataset_fingerprint(
        analyses=analyses,
        lesion_catalog=lesion_catalog,
        config=config,
    )
    save_outputs(
        output_dir=output_dir,
        dataset_fingerprint=dataset_fingerprint,
        lesion_catalog=lesion_catalog,
        sampling_profile=sampling_profile,
    )

    return {
        "dataset_fingerprint": dataset_fingerprint,
        "lesion_catalog": lesion_catalog,
        "sampling_profile": sampling_profile,
    }


# -----------------------------------------------------------------------------
# Backward-compatible entry point
# -----------------------------------------------------------------------------


def compute_fingerprint(
    output_dir: Path,
    bbox_labels: list[int],
    hu_range: tuple[float, float] = (-200.0, 300.0),
    target_spacing: list[float] | None = None,
    tumor_label: int = 2,
    kidney_label: int = 1,
    min_lesion_voxels: int = 5,
    patch_size_candidates: list[tuple[int, int, int]] | None = None,
) -> dict[str, Any]:
    """Backward-compatible wrapper around the new lesion-centric pipeline.

    This keeps the old public function name while upgrading the implementation.
    The function still writes ``dataset_fingerprint.json`` but now also writes:

    - ``lesion_catalog.jsonl``
    - ``sampling_profile.json``
    """
    config = FingerprintConfig(
        tumor_label=tumor_label,
        kidney_label=kidney_label,
        bbox_labels=bbox_labels,
        hu_range=hu_range,
        target_spacing=target_spacing,
        min_lesion_voxels=min_lesion_voxels,
        patch_size_candidates=patch_size_candidates,
    )
    outputs = run_fingerprint_pipeline(output_dir=output_dir, config=config)
    return outputs["dataset_fingerprint"]


if __name__ == "__main__":
    # Example:
    # results = run_fingerprint_pipeline(
    #     output_dir=Path("/path/to/output_dir"),
    #     config=FingerprintConfig(
    #         tumor_label=2,
    #         kidney_label=1,
    #         bbox_labels=[1, 2],
    #         hu_range=(-200, 300),
    #         target_spacing=[1.0, 1.0, 1.0],
    #         min_lesion_voxels=5,
    #         patch_size_candidates=[(96, 96, 64), (128, 128, 96)],
    #     ),
    # )
    # print(results["dataset_fingerprint"]["num_lesions"])
    pass
