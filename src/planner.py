#!/usr/bin/env python3
"""Dataset Planner — preprocess, splits, and fingerprint as separate commands.

Usage:
  python src/planner.py --config config/planner.yaml --preprocess
  python src/planner.py --config config/planner.yaml --splits
  python src/planner.py --config config/planner.yaml --fingerprint
  python src/planner.py --config config/planner.yaml --all   # preprocess + splits
"""

import argparse
import sys
from pathlib import Path

from preprocessor import load_config


def cmd_preprocess(cfg):
    from preprocessor.pipeline import run
    run(cfg)


def cmd_splits(cfg):
    from preprocessor.splits import run_splits
    run_splits(cfg)


def cmd_fingerprint(cfg):
    from preprocessor.fingerprint import run_fingerprint_pipeline, FingerprintConfig

    output_dir = Path(cfg["OUTPUT_DIR"])
    # Build FingerprintConfig from YAML keys, with defaults for backward compatibility
    fcfg = FingerprintConfig(
        tumor_label=cfg.get("TUMOR_LABEL", 2),
        kidney_label=cfg.get("KIDNEY_LABEL", 1),
        bbox_labels=cfg.get("BBOX_LABELS", [1, 2, 3]),
        hu_range=tuple(cfg.get("HU_RANGE", (-200, 300))),
        target_spacing=cfg.get("TARGET_SPACING") if cfg.get("TARGET_SPACING") != "auto" else None,
        size_bin_percentiles=cfg.get("SIZE_BIN_PERCENTILES", [0, 20, 50, 80, 95, 100]),
        balancing_alpha=cfg.get("BALANCING_ALPHA", 0.7),
        min_lesion_voxels=cfg.get("MIN_LESION_VOXELS", 10),
        patch_size_candidates=cfg.get("PATCH_SIZE_CANDIDATES"),
    )
    run_fingerprint_pipeline(output_dir, fcfg)


def main():
    parser = argparse.ArgumentParser(description="Dataset Planner")
    parser.add_argument("--config", default="config/planner.yaml", help="YAML config path")
    parser.add_argument("--preprocess", action="store_true", help="Run VOI preprocessing")
    parser.add_argument("--splits", action="store_true", help="Generate train/val/test splits")
    parser.add_argument("--fingerprint", action="store_true", help="Compute dataset fingerprint")
    parser.add_argument("--all", action="store_true", help="Run preprocess + splits")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if not any([args.preprocess, args.splits, args.fingerprint, args.all]):
        parser.print_help()
        return

    if args.all:
        args.preprocess = True
        args.splits = True

    if args.preprocess:
        cmd_preprocess(cfg)

    if args.splits:
        cmd_splits(cfg)

    if args.fingerprint:
        cmd_fingerprint(cfg)


if __name__ == "__main__":
    main()
