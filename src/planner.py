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
    from preprocessor.fingerprint import compute_fingerprint

    output_dir = Path(cfg["OUTPUT_DIR"])
    compute_fingerprint(
        output_dir,
        bbox_labels=cfg.get("BBOX_LABELS", [1, 2, 3]),
        hu_range=tuple(cfg.get("HU_RANGE", (-200, 300))),
        target_spacing=cfg.get("TARGET_SPACING") if cfg.get("TARGET_SPACING") != "auto" else None,
    )


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
