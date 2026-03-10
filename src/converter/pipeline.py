"""Single-stage DICOM → NIfTI conversion pipeline.

Output layout (flat):
    <output_dir>/nifti/NN_case_YYYYY_0000.nii.gz

Where NN is a 2-digit per-case series counter and case_YYYYY is a 5-digit
sequential case ID.  All files land in one ``nifti/`` folder.

A ``manifest.csv`` is written next to ``nifti/`` with full metadata.
Key columns: **group** (class A/B/C/…) and **phase** (nc/art/ven/delay/undefined).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from . import dicom_utils as du


# ── Case-ID assignment ────────────────────────────────────────────────────

def _normalize_case_id(raw: str) -> str:
    m = re.search(r"(\d+)", raw)
    return f"case_{int(m.group(1)):05d}" if m else raw


class _CaseAssigner:
    """Assigns sequential 5-digit case IDs, respecting pre-existing CSV map."""

    def __init__(self, csv_case_map: dict, start: int = 1):
        self._csv = csv_case_map
        self._assigned: dict[str, str] = {}
        self._counter = start - 1

    def get(self, pid: str) -> str:
        if pid in self._csv:
            cid = _normalize_case_id(self._csv[pid])
            self._assigned[pid] = cid
            return cid
        if pid in self._assigned:
            return self._assigned[pid]
        self._counter += 1
        cid = f"case_{self._counter:05d}"
        self._assigned[pid] = cid
        return cid


# ── Discovery + filtering ────────────────────────────────────────────────

def _discover_and_filter(
    input_dir: Path,
    min_slices: int = 10,
) -> tuple[dict, list, dict]:
    """Discover DICOM series, read + filter, group by patient.

    Returns:
        patient_series  – {patient_id: [series_info, …]}
        skipped         – list of dicts describing skipped series
        stats           – counters for scan planes, modalities, quality, etc.
    """
    raw_series = du.discover_series(input_dir)

    patient_series: dict[str, list] = defaultdict(list)
    skipped: list[dict] = []
    stats: dict[str, dict] = {
        "scan_plane": defaultdict(int),
        "modality": defaultdict(int),
        "quality": defaultdict(int),
        "pid_source": defaultdict(int),
    }

    for item in tqdm(raw_series, desc="Reading & filtering", unit="series"):
        result = du.read_series(item["files"])
        if result is None:
            skipped.append({
                "series_uid": item["series_uid"],
                "directory": item["directory"],
                "reason": "Failed to read DICOM series",
            })
            continue

        reader, image, quality = result
        pid, pid_src = du.extract_patient_id(reader, item["directory"])
        stats["pid_source"][pid_src] += 1
        stats["quality"][quality["severity"]] += 1

        modality = du.get_tag(reader, "0008|0060") or "nan"
        if modality != "nan":
            stats["modality"][modality] += 1

        scan_plane = du.detect_scan_plane(reader, image)
        stats["scan_plane"][scan_plane] += 1

        skip_reason = du.filter_series(
            modality=modality,
            scan_plane=scan_plane,
            num_slices=len(item["files"]),
            quality_severity=quality["severity"],
            min_slices=min_slices,
        )
        if skip_reason:
            skipped.append({
                "patient_id": pid,
                "series_uid": item["series_uid"],
                "directory": item["directory"],
                "reason": skip_reason,
            })
            continue

        patient_series[pid].append({
            "path": item["directory"],
            "series_uid": item["series_uid"],
            "image": image,
            "files": item["files"],
            "reader": reader,
            "modality": modality,
            "quality": quality,
        })

    return dict(patient_series), skipped, {k: dict(v) for k, v in stats.items()}


# ── Conversion (flat nifti/ output) ──────────────────────────────────────

def _convert_patients(
    patient_series: dict,
    nifti_dir: Path,
    class_map: dict,
    case_assigner: _CaseAssigner,
    dataset_type: str,
) -> tuple[list, list]:
    """Convert all patients to NIfTI in a single flat *nifti_dir*.

    Filename: NN_case_YYYYY_0000.nii.gz
        NN       – 2-digit per-case series counter
        YYYYY    – 5-digit case number
        _0000    – hardcoded channel suffix

    Returns (records, failures).
    """
    records: list[dict] = []
    failures: list[dict] = []
    series_counters: dict[str, int] = defaultdict(int)

    patients = list(patient_series.items())
    pbar = tqdm(patients, desc="Converting", unit="patient")

    for pid, series_list in pbar:
        pbar.set_postfix_str(pid[:20])
        group = class_map.get(pid, "NG")
        case_id = case_assigner.get(pid)

        for idx, si in enumerate(series_list):
            try:
                ras = du.to_ras(si["image"])
                phase, phase_src = du.detect_protocol(si["reader"], si["path"])
                scan_plane = du.detect_scan_plane(si["reader"], si["image"])

                md = du.extract_metadata(si["reader"], {
                    "patient_id": pid,
                    "group": group,
                    "scan_idx": idx,
                    "phase": phase,
                    "phase_source": phase_src,
                    "scan_plane": scan_plane,
                    "case_id": case_id,
                    "dataset_type": dataset_type,
                    "modality": si["modality"],
                    "spacing_quality": si["quality"]["severity"],
                    "nonuniformity_mm": si["quality"]["nonuniformity_mm"],
                })

                ras = du.embed_metadata(ras, si["reader"], md)

                # ── flat filename: NN_case_YYYYY_0000.nii.gz ──
                snum = series_counters[case_id]
                series_counters[case_id] += 1
                fname = f"{snum:02d}_{case_id}_0000.nii.gz"
                out_path = nifti_dir / fname

                du.write_nifti(ras, out_path)
                val = du.validate_nifti(out_path, si["image"])

                spacing = si["image"].GetSpacing()
                dims = si["image"].GetSize()

                records.append({
                    "filename": fname,
                    "patient_id": pid,
                    "case_id": case_id,
                    "group": group,
                    "phase": phase,
                    "phase_source": phase_src,
                    "scan_idx": idx,
                    "modality": md.get("modality", "nan"),
                    "scan_plane": scan_plane,
                    "image_orientation": md.get("image_orientation", "nan"),
                    "spacing_quality": md.get("spacing_quality", "ok"),
                    "nonuniformity_mm": md.get("nonuniformity_mm"),
                    "sex": md.get("sex", "nan"),
                    "DOB": md.get("DOB", "nan"),
                    "age": md.get("age", "nan"),
                    "age_at_scan": md.get("age_at_scan", "nan"),
                    "DOS": md.get("DOS", "nan"),
                    "study_time": md.get("study_time", "nan"),
                    "series_description": md.get("series_description", "nan"),
                    "series_number": md.get("series_number", "nan"),
                    "manufacturer": md.get("manufacturer", "nan"),
                    "manufacturer_model": md.get("manufacturer_model", "nan"),
                    "institution": md.get("institution", "nan"),
                    "kvp": md.get("kvp", "nan"),
                    "contrast_agent": md.get("contrast_agent", "nan"),
                    "spacing_x": float(spacing[0]),
                    "spacing_y": float(spacing[1]),
                    "spacing_z": float(spacing[2]),
                    "dim_x": int(dims[0]),
                    "dim_y": int(dims[1]),
                    "dim_z": int(dims[2]),
                    "slice_thickness": md.get("slice_thickness", "nan"),
                    "num_slices": len(si["files"]),
                    "conversion_date": datetime.now().isoformat(),
                    "dataset_type": dataset_type,
                    "output_path": fname,
                    "validation": "ok" if val["valid"] else "warning",
                })

            except Exception as exc:
                failures.append({
                    "patient_id": pid,
                    "series_path": si["path"],
                    "error": str(exc),
                })

    return records, failures


# ── Summary builder ───────────────────────────────────────────────────────

def _build_summary(
    records: list,
    skipped: list,
    failures: list,
    stats: dict,
    dataset_type: str,
    input_dir: Path,
    output_dir: Path,
) -> dict:
    """Compile a JSON-serialisable summary dict."""
    unique = {r["patient_id"] for r in records}
    group_dist = defaultdict(int)
    phase_dist = defaultdict(int)
    for r in records:
        group_dist[r["group"]] += 1
        phase_dist[r["phase"]] += 1

    return {
        "conversion_date": datetime.now().isoformat(),
        "dataset_type": dataset_type,
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "total_patients": len(unique),
        "total_series_discovered": len(records) + len(skipped),
        "successful_conversions": sum(1 for r in records if r["validation"] == "ok"),
        "failed_conversions": len(failures),
        "skipped_series": len(skipped),
        "group_distribution": dict(group_dist),
        "phase_distribution": dict(phase_dist),
        **{f"stat_{k}": v for k, v in stats.items()},
        "failed_cases": failures,
        "skipped_details": skipped[:100],
    }


# ── Save outputs ──────────────────────────────────────────────────────────

def _save_outputs(records: list, summary: dict, output_dir: Path) -> None:
    """Write manifest.csv and conversion_summary.json next to nifti/."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "conversion_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if records:
        pd.DataFrame(records).to_csv(output_dir / "manifest.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

def run(
    input_dir: str | Path,
    output_dir: str | Path,
    classification_csv: Optional[str | Path] = None,
    start_case_id: int = 1,
    min_slices: int = 10,
) -> dict:
    """Execute the full single-stage DICOM → NIfTI conversion.

    Output structure::

        <output_dir>/
            nifti/
                00_case_00001_0000.nii.gz
                01_case_00001_0000.nii.gz
                00_case_00002_0000.nii.gz
                …
            manifest.csv
            conversion_summary.json

    Args:
        input_dir:          Root directory containing DICOM files.
        output_dir:         Destination directory (e.g. data/Dataset820).
        classification_csv: Optional CSV with patient→group mapping.
        start_case_id:      Starting case number (default 1).
        min_slices:         Min slices to accept a series (default 10).

    Returns:
        Summary dict (also saved as JSON).
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    nifti_dir = output_dir / "nifti"
    nifti_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(classification_csv) if classification_csv else None

    # 1 ─ Detect dataset type
    print("[1/5] Detecting dataset type…")
    dataset_type = du.detect_dataset_type(input_dir)
    print(f"       → {dataset_type}")

    # 2 ─ Load classifications
    print("[2/5] Loading classifications…")
    class_map, case_map = du.load_classifications(csv_path)
    print(f"       → {len(class_map)} patients loaded")

    # 3 ─ Discover + filter
    print("[3/5] Discovering & filtering DICOM series…")
    patient_series, skipped, stats = _discover_and_filter(input_dir, min_slices)
    total_series = sum(len(v) for v in patient_series.values())
    print(f"       → {len(patient_series)} patients, {total_series} series kept, {len(skipped)} skipped")

    # 4 ─ Convert → flat nifti/
    print("[4/5] Converting DICOM → NIfTI  (flat output in nifti/)…")
    assigner = _CaseAssigner(case_map, start_case_id)
    records, failures = _convert_patients(
        patient_series, nifti_dir, class_map, assigner, dataset_type,
    )
    print(f"       → {len(records)} converted, {len(failures)} failed")

    # 5 ─ Save manifest + summary
    print("[5/5] Saving manifest & summary…")
    summary = _build_summary(
        records, skipped, failures, stats, dataset_type, input_dir, output_dir,
    )
    _save_outputs(records, summary, output_dir)
    print(f"       → {output_dir / 'manifest.csv'}")
    print(f"       → {output_dir / 'conversion_summary.json'}")

    ok = summary["successful_conversions"]
    fail = summary["failed_conversions"]
    print(f"\nDone — {ok} successful, {fail} failed.\n")
    return summary
