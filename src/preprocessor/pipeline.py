"""VOI preprocessing pipeline.

Steps:
  1. Discover image-mask pairs
  2. Scan spacings → compute target spacing
  3. Process each case (load → RAS → masks → separate kidneys → crop → resample → save)
     + inline fingerprint collection
  4. Update manifest.csv with Laterality / Tumor columns
  5. Write patient_preprocess.csv
  6. Write dataset.json
  7. Write dataset_fingerprint.json
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

from .discover import discover_files
from .fingerprint import _collect_bbox_intensities, _intensity_stats, _shape_stats
from .processing import (
    generate_dual_masks,
    separate_kidneys,
    extract_voi_crop,
    resample_voi,
    validate_voi,
    validate_mask_integrity,
    to_ras,
    compute_tumor_metrics,
)

# Uppercase folder names for contrast phases
_PHASE_MAP = {"nc": "NC", "art": "ART", "ven": "VEN", "delay": "DELAY", "undefined": "UNDEFINED"}


# ===================================================================
# Spacing scan / target spacing
# ===================================================================

def scan_spacings(file_list: Dict[str, Dict], max_samples: int = 100) -> np.ndarray:
    """Return array of spacings (N, 3) from a sample of images."""
    spacings = []
    items = list(file_list.values())[:max_samples]
    for item in tqdm(items, desc="Scanning spacings", unit="case"):
        r = sitk.ImageFileReader()
        r.SetFileName(item["image_path"])
        r.ReadImageInformation()
        spacings.append(list(r.GetSpacing()))
    return np.array(spacings)


def determine_target_spacing(
    spacings: np.ndarray,
    cfg_value,
    aniso_thresh: float = 2.0,
    spacing_mode: str = "anisotropic",
    z_coarse_thresh: float = 2.0,
) -> list:
    """Compute target spacing from dataset statistics or a fixed config value.

    Parameters
    ----------
    spacings:
        (N, 3) array of per-image spacings in SimpleITK order [X, Y, Z].
    cfg_value:
        Either ``"auto"`` to derive spacing from the dataset, or a fixed
        3-element list/tuple (e.g. ``[0.78, 0.78, 1.0]``) used verbatim.
    aniso_thresh:
        Retained for backward-compatibility; unused in auto logic.
    spacing_mode:
        - ``"anisotropic"`` (default): Preserve high-res in-plane X/Y from the
          dataset median; sharpen Z to ``max(min(z_median, 1.0), min(x_med, y_med))``
          when Z-median exceeds *z_coarse_thresh*. Avoids the "crushed" kidney
          appearance caused by thick axial slices.
        - ``"isotropic"``: All three axes set to the global minimum median spacing
          (full isotropic resampling).
    z_coarse_thresh:
        Z-axis median threshold in mm above which Z is sharpened
        (only applies when spacing_mode="anisotropic").
    """
    if cfg_value != "auto":
        return list(cfg_value)

    med = np.median(spacings, axis=0)  # [x_med, y_med, z_med]

    if spacing_mode == "isotropic":
        global_min = float(med.min())
        return [global_min, global_min, global_min]

    # anisotropic: keep X/Y in-plane resolution; sharpen Z when coarse
    tgt = med.copy()
    if tgt[2] > z_coarse_thresh:
        in_plane_min = float(min(tgt[0], tgt[1]))
        tgt[2] = max(min(float(tgt[2]), 1.0), in_plane_min)
    return tgt.tolist()


# ===================================================================
# I/O helpers
# ===================================================================

def _output_path(voi_dir: Path, out_type: str, case_data: Dict, side: str) -> Path:
    """Build output path.

    Layout: {voi_dir}/{type}/{group}/{case_id}/{phase}/{case_name}_{side}.npy

    ``group`` and ``phase`` come from manifest.csv via discover_files().
    """
    group = case_data.get("group", "NG") or "NG"
    pid = case_data["patient_id"]
    phase = case_data.get("phase", "undefined") or "undefined"

    d = voi_dir / out_type / group / pid / _PHASE_MAP.get(phase.lower(), phase.upper())
    d.mkdir(parents=True, exist_ok=True)

    return d / f"{case_data['case_name']}_{side}.npy"


# ===================================================================
# Single-case processing
# ===================================================================

def _process_case(
    case_data: Dict,
    cfg: Dict,
    target_spacing: list,
) -> Dict:
    """Process one case through the full pipeline. Returns a results dict."""
    image = sitk.ReadImage(case_data["image_path"])
    mask = sitk.ReadImage(case_data["mask_path"])

    image, mask = to_ras(image, mask)

    bbox_mask, save_mask = generate_dual_masks(
        mask,
        cfg.get("BBOX_LABELS", [1, 2, 3]),
        cfg.get("MASK_LABELS", [2]),
    )

    # B1: explicit empty-bbox guard for clear diagnostics
    arr_bb = sitk.GetArrayFromImage(bbox_mask)
    if arr_bb.max() == 0:
        return {
            "case_name": case_data["case_name"],
            "patient_id": case_data["patient_id"],
            "group": case_data.get("group", "NG"),
            "phase": case_data.get("phase", "undefined"),
            "sides": {},
            "error": "Empty bbox_mask — no tissue labels found in segmentation",
        }

    bboxes = separate_kidneys(bbox_mask)

    result: Dict[str, Any] = {
        "case_name": case_data["case_name"],
        "patient_id": case_data["patient_id"],
        "group": case_data.get("group", "NG"),
        "phase": case_data.get("phase", "undefined"),
        "sides": {},
    }

    # Build resampled bbox_mask for per-side validation
    bbox_labels = cfg.get("BBOX_LABELS", [1, 2, 3])

    # Prepare minimum VOI size in original voxel spacing if specified
    min_voi_size_original = None
    if cfg.get("MIN_VOI_SIZE") is not None:
        min_voi_size_target = np.array(cfg["MIN_VOI_SIZE"], dtype=np.float32)
        orig_spacing = np.array(image.GetSpacing(), dtype=np.float32)
        target_spacing_arr = np.array(target_spacing, dtype=np.float32)
        # Convert from target spacing voxels to original spacing voxels
        min_voi_size_original = (min_voi_size_target * target_spacing_arr / orig_spacing).astype(np.int32)

    for side, bbox_info in bboxes.items():
        voi_img, voi_seg, voi_sm, crop_prov = extract_voi_crop(
            image, mask, save_mask, bbox_info,
            cfg["EXPANSION_MM"], np.array(image.GetSpacing()),
            min_voi_size=min_voi_size_original,
        )
        voi_img, voi_seg, voi_sm, resample_prov = resample_voi(
            voi_img, voi_seg, voi_sm,
            target_spacing,
            cfg.get("ANISO_RATIO_THRESH", 2.0),
        )

        # Rebuild bbox mask on the resampled seg for tissue-level validation
        seg_arr = sitk.GetArrayFromImage(voi_seg)
        voi_bb = sitk.GetImageFromArray(np.isin(seg_arr, bbox_labels).astype(np.uint8))
        voi_bb.CopyInformation(voi_seg)

        ok, reason = validate_voi(
            voi_bb,
            voi_sm,
            cfg.get("MIN_KIDNEY_VOXELS", 0),
            cfg.get("MIN_TUMOR_VOXELS", 0),
        )
        if not ok:
            result["sides"][side] = {
                "status": "skipped", "reason": reason,
                "crop_provenance": crop_prov,
                "resample_provenance": resample_prov,
            }
            continue

        # Mask integrity check (dtype, values, shape match)
        expected_shape = sitk.GetArrayFromImage(voi_img).shape
        mint_ok, mint_reason = validate_mask_integrity(voi_sm, expected_shape)
        if not mint_ok:
            result["sides"][side] = {
                "status": "skipped", "reason": f"Mask integrity: {mint_reason}",
                "crop_provenance": crop_prov,
                "resample_provenance": resample_prov,
            }
            continue

        metrics = compute_tumor_metrics(voi_img, voi_seg)

        result["sides"][side] = {
            "status": "ok",
            "image": voi_img,
            "seg": voi_seg,
            "metrics": metrics,
            "shape": voi_img.GetSize(),
            "spacing": voi_img.GetSpacing(),
            "crop_provenance": crop_prov,
            "resample_provenance": resample_prov,
        }
    return result


# ===================================================================
# Save outputs
# ===================================================================

def _save_case(
    result: Dict, case_data: Dict, voi_dir: Path, axes_order: str = "ZYX"
) -> List[str]:
    """Save mask + images .npy files. Returns list of relative paths (to voi_dir).

    ``axes_order``:
        - ``"ZYX"`` (default): native SimpleITK convention — depth-first
          (Z, Y, X). Compatible with nnU-Net, ITK-Snap, and most medical
          imaging tools.
        - ``"XYZ"``: anatomical-first — applies ``np.transpose(arr, (2, 1, 0))``
          before saving, swapping Z↔X for viewers that expect width-first arrays.
          Both the image and mask receive the same transpose so they stay aligned.
    """
    def _maybe_transpose(arr: np.ndarray) -> np.ndarray:
        if axes_order == "XYZ":
            return np.transpose(arr, (2, 1, 0))
        return arr

    saved: List[str] = []
    for side, sd in result["sides"].items():
        if sd["status"] != "ok":
            continue

        # mask — multilabel segmentation (uint8, original labels preserved)
        mp = _output_path(voi_dir, "mask", case_data, side)
        np.save(mp, _maybe_transpose(sitk.GetArrayFromImage(sd["seg"]).astype(np.uint8)))
        saved.append(str(mp.relative_to(voi_dir)))

        # images — raw CT VOI (bbox crop + resampled, no masking applied, float32)
        ip = _output_path(voi_dir, "images", case_data, side)
        np.save(ip, _maybe_transpose(sitk.GetArrayFromImage(sd["image"]).astype(np.float32)))
        saved.append(str(ip.relative_to(voi_dir)))
    return saved


# ===================================================================
# Manifest update
# ===================================================================

def _update_manifest(dataset_dir: Path, results: Dict[str, Dict]) -> None:
    """Append Laterality + Tumor columns to manifest.csv in the DatasetID folder."""
    manifest_path = dataset_dir / "manifest.csv"
    if not manifest_path.exists():
        return

    df = pd.read_csv(manifest_path)

    patient_info: Dict[str, Dict] = {}
    for res in results.values():
        pid = res["patient_id"]
        if pid not in patient_info:
            patient_info[pid] = {"sides": set(), "has_tumor": False}
        for side, sd in res["sides"].items():
            if sd["status"] == "ok":
                patient_info[pid]["sides"].add(side)
                if sd.get("metrics", {}).get("has_tumor", False):
                    patient_info[pid]["has_tumor"] = True

    def _match_pid(case_id):
        m = re.search(r"(case_\d{5})", str(case_id))
        cid = m.group(1) if m else str(case_id)
        return patient_info.get(cid)

    laterality, tumor = [], []
    for _, row in df.iterrows():
        info = _match_pid(row.get("case_id", row.get("patient_id", "")))
        if info:
            sides = info["sides"]
            if {"L", "R"} <= sides:
                laterality.append("Both")
            elif "L" in sides:
                laterality.append("L")
            elif "R" in sides:
                laterality.append("R")
            else:
                laterality.append("")
            tumor.append("Yes" if info["has_tumor"] else "No")
        else:
            laterality.append("")
            tumor.append("")

    df["Laterality"] = laterality
    df["Tumor"] = tumor
    df.to_csv(manifest_path, index=False)
    print(f"       Updated manifest → {manifest_path}")


# ===================================================================
# patient_preprocess.csv
# ===================================================================

def _build_patient_csv(
    results: Dict[str, Dict],
    failed_keys: List[str],
    file_list: Dict[str, Dict],
    dataset_dir: Path,
    voi_dir: Path,
    image_folder: Optional[Path] = None,
) -> None:
    """Write patient_preprocess.csv with per-patient summary.

    Columns:
      TCGA_ID | CaseID | N_Series | N_ART | N_VEN | N_DELAY | N_UNDEFINED
      RTumor  | LTumor | Manufacturer | ManufacturerModel | Institution
      Status  | Comments
    """
    # Read manifest for TCGA_ID, protocol counts, and equipment info per case_id
    # Look in: 1) IMAGE_FOLDER (source), 2) output dataset_dir
    manifest_path = None
    for candidate in [
        image_folder / "manifest.csv" if image_folder else None,
        dataset_dir / "manifest.csv",
    ]:
        if candidate and candidate.exists():
            manifest_path = candidate
            break
    tcga_map: Dict[str, str] = {}
    proto_counts: Dict[str, Dict[str, int]] = {}
    series_count: Dict[str, int] = {}
    equip_map: Dict[str, Dict[str, str]] = {}  # case_id → {manufacturer, model, institution}

    if manifest_path is not None:
        mdf = pd.read_csv(manifest_path)
        for _, row in mdf.iterrows():
            cid = str(row.get("case_id", ""))
            pid = str(row.get("patient_id", ""))
            proto = str(row.get("phase", row.get("protocol", ""))).lower()
            if cid:
                tcga_map.setdefault(cid, pid)
                if cid not in proto_counts:
                    proto_counts[cid] = defaultdict(int)
                proto_counts[cid][proto] += 1
                series_count[cid] = series_count.get(cid, 0) + 1
                # keep first non-empty equipment info per case
                if cid not in equip_map:
                    equip_map[cid] = {
                        "manufacturer": str(row.get("manufacturer", "")).strip(),
                        "manufacturer_model": str(row.get("manufacturer_model", "")).strip(),
                        "institution": str(row.get("institution", "")).strip(),
                    }

    # Fallback: derive series/phase counts from file_list when manifest is missing
    if not series_count:
        for entry in file_list.values():
            pid = entry["patient_id"]
            phase = (entry.get("phase") or "undefined").lower()
            series_count[pid] = series_count.get(pid, 0) + 1
            if pid not in proto_counts:
                proto_counts[pid] = defaultdict(int)
            proto_counts[pid][phase] += 1

    # Aggregate per patient from processing results
    patient_agg: Dict[str, Dict[str, Any]] = {}

    for _key, res in results.items():
        pid = res["patient_id"]
        if pid not in patient_agg:
            patient_agg[pid] = {
                "r_tumor": False, "l_tumor": False,
                "has_ok": False, "has_skipped": False,
                "error": None, "comments": [],
            }
        pa = patient_agg[pid]

        if res.get("error"):
            pa["error"] = res["error"]
            pa["comments"].append(f"Error: {res['error']}")
            continue

        for side, sd in res["sides"].items():
            if sd["status"] == "ok":
                pa["has_ok"] = True
                if sd.get("metrics", {}).get("has_tumor", False):
                    if side == "R":
                        pa["r_tumor"] = True
                    else:
                        pa["l_tumor"] = True
            elif sd["status"] == "skipped":
                pa["has_skipped"] = True
                pa["comments"].append(f"{side}: {sd.get('reason', 'skipped')}")

    # Mark any remaining failed keys
    for fk in failed_keys:
        pid = file_list[fk]["patient_id"]
        if pid not in patient_agg:
            patient_agg[pid] = {
                "r_tumor": False, "l_tumor": False,
                "has_ok": False, "has_skipped": False,
                "error": "Processing failed", "comments": ["Processing failed"],
            }
        else:
            patient_agg[pid]["error"] = "Processing failed"
            patient_agg[pid]["comments"].append("Processing failed")

    # Build rows
    rows = []
    for pid in sorted(patient_agg):
        pa = patient_agg[pid]
        pc = proto_counts.get(pid, {})

        if pa["error"] and not pa["has_ok"]:
            status = "Fail"
        elif pa["has_skipped"]:
            status = "Warning"
        elif pa["has_ok"]:
            status = "Pass"
        else:
            status = "Fail"

        eq = equip_map.get(pid, {})
        rows.append({
            "TCGA_ID": tcga_map.get(pid, ""),
            "CaseID": pid,
            "N_Series": series_count.get(pid, 0),
            "N_ART": pc.get("art", 0),
            "N_VEN": pc.get("ven", 0),
            "N_DELAY": pc.get("delay", 0),
            "N_UNDEFINED": pc.get("undefined", 0),
            "RTumor": pa["r_tumor"],
            "LTumor": pa["l_tumor"],
            "Manufacturer": eq.get("manufacturer", ""),
            "ManufacturerModel": eq.get("manufacturer_model", ""),
            "Institution": eq.get("institution", ""),
            "Status": status,
            "Comments": "; ".join(pa["comments"]) if pa["comments"] else "",
        })

    df = pd.DataFrame(rows)
    csv_path = voi_dir / "patient_preprocess.csv"
    df.to_csv(csv_path, index=False)
    print(f"       Saved → {csv_path}")


# ===================================================================
# dataset.json
# ===================================================================

def _build_dataset_json(
    results: Dict[str, Dict],
    failed: List[str],
    cfg: Dict,
    target_spacing: list,
    voi_dir: Path,
    all_saved: Dict[str, List[str]],
    file_list: Dict[str, Dict],
) -> Dict:
    patients: Dict[str, Dict] = {}  # per-patient rich record
    patient_protocols: Dict[str, set] = defaultdict(set)
    mask_count = 0
    img_count = 0

    for key, res in results.items():
        pid = res["patient_id"]
        has_ok = any(s["status"] == "ok" for s in res["sides"].values())

        if pid not in patients:
            patients[pid] = {
                "group": res.get("group", "NG") or "NG",
                "cases": [],
                "kidneys": set(),
                "phases": set(),
                "files": {"mask": [], "images": []},
                "status": "FAIL",
            }
        p = patients[pid]
        p["cases"].append(res["case_name"])
        if has_ok:
            p["status"] = "PASS"
        for side, sd in res["sides"].items():
            if sd.get("status") == "ok":
                p["kidneys"].add(side)

        phase = file_list.get(key, {}).get("phase")
        if phase:
            p["phases"].add(phase)
            patient_protocols[pid].add(phase)

        for f in all_saved.get(key, []):
            if f.startswith("mask/"):
                mask_count += 1
                p["files"]["mask"].append(f)
            elif f.startswith("images/"):
                img_count += 1
                p["files"]["images"].append(f)

    # Convert sets → sorted lists for JSON serialization
    for pid in patients:
        patients[pid]["kidneys"] = sorted(patients[pid]["kidneys"])
        patients[pid]["phases"] = sorted(patients[pid]["phases"])

    num_ok = sum(1 for v in patients.values() if v["status"] == "PASS")

    example = None
    for files in all_saved.values():
        for f in files:
            if f.startswith("images/"):
                example = f
                break
        if example:
            break

    # Phase coverage summary
    multi_phase = {pid: sorted(ps) for pid, ps in patient_protocols.items() if len(ps) > 1}
    single_phase = {pid: sorted(ps) for pid, ps in patient_protocols.items() if len(ps) == 1}
    no_phase = [pid for pid in patients if pid not in patient_protocols]

    return {
        "version": "2.0.0",
        "dataset_id": cfg.get("DATASET_ID"),
        "channel_names": {"0": "CT"},
        "labels": cfg.get("LABELS", {1: "kidney", 2: "tumor", 3: "cyst"}),
        "file_ending": ".npy",
        "file_format": "{case_name}_{side}.npy",
        "file_structure": "{output_dir}/{type}/{group}/{patient_id}/{phase}/{case_name}_{side}.npy",
        "example_path": example,
        "output_types": {
            "mask": "Multilabel segmentation (uint8, original label values preserved)",
            "images": "Raw CT VOI bbox crop — no mask applied (float32)",
        },
        "source": {
            "image_folder": str(cfg["IMAGE_FOLDER"]),
            "mask_folder": str(cfg["MASK_FOLDER"]),
            "image_suffix": cfg.get("IMAGE_SUFFIX"),
            "mask_suffix": cfg.get("MASK_SUFFIX"),
        },
        "statistics": {
            "num_cases": num_ok,
            "num_patients": len(patients),
            "num_failed": len(failed),
            "num_mask_files": mask_count,
            "num_image_files": img_count,
        },
        "phase_coverage": {
            "multi_phase_patients": multi_phase,
            "single_phase_patients": single_phase,
            "no_phase_patients": no_phase,
            "statistics": {
                "num_multi": len(multi_phase),
                "num_single": len(single_phase),
                "num_none": len(no_phase),
            },
        },
        "processing_config": {
            "target_spacing": target_spacing,
            "expansion_mm": cfg["EXPANSION_MM"],
            "use_ras_orientation": True,
            "bbox_labels": cfg.get("BBOX_LABELS", [1, 2, 3]),
            "mask_labels": cfg.get("MASK_LABELS", [2]),
            "aniso_ratio_thresh": cfg.get("ANISO_RATIO_THRESH", 2.0),
        },
        "failed_cases": failed,
        "success_cases": [pid for pid, v in patients.items() if v["status"] == "PASS"],
        "patients": patients,
    }


# ===================================================================
# Public: run
# ===================================================================

def run(cfg: Dict) -> Dict:
    """Execute the full preprocessing pipeline.

    Output layout:
        dataset/DatasetID/voi/images/{group}/{case_id}/{phase}/{case_name}_{side}.npy
        dataset/DatasetID/voi/mask/{group}/{case_id}/{phase}/{case_name}_{side}.npy
        dataset/DatasetID/voi/dataset.json
        dataset/DatasetID/voi/dataset_fingerprint.json
        dataset/DatasetID/voi/patient_preprocess.csv

    Group and phase are read from manifest.csv (auto-discovered).

    Returns the dataset.json dict.
    """
    voi_dir = Path(cfg["OUTPUT_DIR"])          # data/dataset/DatasetXXX/voi
    voi_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = voi_dir.parent               # data/dataset/DatasetXXX

    image_folder = Path(cfg["IMAGE_FOLDER"])
    mask_folder = Path(cfg["MASK_FOLDER"])

    # 1 ─ Discover
    print("[1/7] Discovering files…")
    file_list = discover_files(
        image_folder, mask_folder,
        cfg["IMAGE_SUFFIX"], cfg["MASK_SUFFIX"],
        manifest_output_dir=dataset_dir,
        patient_id_pattern=cfg.get("PATIENT_ID_PATTERN"),
    )
    print(f"       → {len(file_list)} image-mask pairs")

    # 2 ─ Target spacing
    print("[2/7] Computing target spacing…")
    spacings = scan_spacings(file_list)
    spacing_mode = cfg.get("SPACING_MODE", "anisotropic")
    target_spacing = determine_target_spacing(
        spacings,
        cfg.get("TARGET_SPACING", "auto"),
        cfg.get("ANISO_RATIO_THRESH", 2.0),
        spacing_mode=spacing_mode,
        z_coarse_thresh=cfg.get("Z_COARSE_THRESH", 2.0),
    )
    print(f"       → {[round(s, 3) for s in target_spacing]} (mode={spacing_mode})")

    # 3 ─ Process + inline fingerprint collection
    print("[3/7] Processing VOIs…")
    results: Dict[str, Dict] = {}
    failed: List[str] = []
    all_saved: Dict[str, List[str]] = {}

    bbox_labels = cfg.get("BBOX_LABELS", [1, 2, 3])
    hu_range = cfg.get("HU_RANGE", [-200, 300])
    fp_intensities: List[np.ndarray] = []
    fp_shapes: List[list] = []
    fp_total_vox = 0
    fp_filtered_vox = 0

    for key in tqdm(file_list, desc="Processing", unit="case"):
        try:
            res = _process_case(file_list[key], cfg, target_spacing)
            results[key] = res
            saved = _save_case(res, file_list[key], voi_dir, cfg.get("SAVE_AXES_ORDER", "ZYX"))
            all_saved[key] = saved

            # Inline fingerprint collection from successful sides
            for _side, sd in res["sides"].items():
                if sd.get("status") != "ok":
                    continue
                img_arr = sitk.GetArrayFromImage(sd["image"])
                seg_arr = sitk.GetArrayFromImage(sd["seg"])
                bb_arr = np.isin(seg_arr, bbox_labels).astype(np.uint8)
                fp_shapes.append(list(img_arr.shape))
                bbox_vals = img_arr[bb_arr > 0].flatten()
                fp_total_vox += len(bbox_vals)
                filt = _collect_bbox_intensities(img_arr, bb_arr, hu_range[0], hu_range[1])
                fp_filtered_vox += len(filt)
                fp_intensities.append(filt)

        except Exception as exc:
            failed.append(key)
            results[key] = {
                "case_name": file_list[key]["case_name"],
                "patient_id": file_list[key]["patient_id"],
                "group": file_list[key].get("group", "NG"),
                "phase": file_list[key].get("phase", "undefined"),
                "sides": {},
                "error": str(exc),
            }

    ok_count = sum(1 for r in results.values() if any(
        s.get("status") == "ok" for s in r.get("sides", {}).values()
    ))
    print(f"       → {ok_count} successful, {len(failed)} failed")

    # 4 ─ Update manifest
    print("[4/7] Updating manifest…")
    _update_manifest(dataset_dir, results)

    # 5 ─ patient_preprocess.csv
    print("[5/7] Writing patient_preprocess.csv…")
    _build_patient_csv(results, failed, file_list, dataset_dir, voi_dir, image_folder)

    # 6 ─ dataset.json
    print("[6/7] Writing dataset.json…")
    ds = _build_dataset_json(results, failed, cfg, target_spacing, voi_dir, all_saved, file_list)
    ds_path = voi_dir / "dataset.json"
    with open(ds_path, "w") as f:
        json.dump(ds, f, indent=2)
    print(f"       → {ds_path}")

    # 7 ─ Fingerprint
    print("[7/7] Writing dataset_fingerprint.json…")
    intensities = np.concatenate(fp_intensities) if fp_intensities else np.array([], dtype=np.float32)
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
        "shape_statistics": _shape_stats(fp_shapes),
        "voxel_info": {
            "hu_range": list(hu_range),
            "total_voxels": fp_total_vox,
            "filtered_voxels": fp_filtered_vox,
        },
        "num_vois": len(fp_shapes),
        "target_spacing": target_spacing,
    }
    fp_path = voi_dir / "dataset_fingerprint.json"
    with open(fp_path, "w") as f:
        json.dump(fingerprint, f, indent=2)
    print(f"       → {fp_path}")

    print(f"\nDone — {ok_count} cases, {len(failed)} failed.\n")
    return ds
