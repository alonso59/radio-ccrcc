#!/usr/bin/env python3
"""
Dataset Planner

Orchestrates dataset preparation steps:
  - preprocessing
  - fingerprint-only
  - splits-only
  - structure validation

This script keeps all paths under data/dataset/ and uses the config file
as the single source of truth for source paths and output locations.
"""
import argparse
import sys
from pathlib import Path

from preprocessor import VOIPreprocessor, load_config
from preprocessor.utils.metrics import compute_fingerprint_from_outputs


def validate_output_structure(output_dir: Path) -> int:
    """
    Validate minimal output structure for downstream usage.

    Returns:
        0 if ok, 1 if issues found
    """
    errors = []
    warnings = []

    if not output_dir.exists():
        errors.append(f"Output directory not found: {output_dir}")
    else:
        required_files = [
            output_dir / 'dataset.json',
            output_dir / 'dataset_fingerprint_segmentation.json',
        ]
        required_dirs = [
            output_dir / 'mask',
            output_dir / 'segmentation',
        ]
        optional_dirs = [
            output_dir / 'images',
        ]

        for path in required_files:
            if not path.exists():
                errors.append(f"Missing required file: {path}")

        for path in required_dirs:
            if not path.exists():
                errors.append(f"Missing required folder: {path}")

        for path in optional_dirs:
            if not path.exists():
                warnings.append(f"Optional folder missing (recommended): {path}")

    # Enforce data/dataset/ location (soft rule)
    if 'data/dataset' not in str(output_dir):
        warnings.append("OUTPUT_DIR is not under data/dataset/ (recommended).")

    if errors:
        print("\n❌ Validation failed:")
        for err in errors:
            print(f"  - {err}")
    if warnings:
        print("\n⚠ Warnings:")
        for warn in warnings:
            print(f"  - {warn}")

    if not errors:
        print("\n✓ Output structure looks valid.")
        return 0
    return 1


def run_splits(config_path: Path) -> int:
    """Run split generation using config file."""
    import subprocess
    cmd = [sys.executable, 'src/generate_splits.py', '--config', str(config_path)]
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Dataset Planner (preprocess / fingerprint / splits / validate)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python src/planner.py --config config/preprocessor_config.yaml --all

  # Preprocess only
  python src/planner.py --config config/preprocessor_config.yaml --preprocess

  # Fingerprint only
  python src/planner.py --config config/preprocessor_config.yaml --fingerprint

  # Splits only
  python src/planner.py --config config/preprocessor_config.yaml --splits

  # Validate output structure only
  python src/planner.py --config config/preprocessor_config.yaml --validate
        """
    )
    parser.add_argument('--config', default='config/planner.yaml',
                        help='Path to configuration file')
    parser.add_argument('--preprocess', action='store_true',
                        help='Run preprocessing (VOI extraction)')
    parser.add_argument('--fingerprint', action='store_true',
                        help='Recompute fingerprints from saved outputs')
    parser.add_argument('--splits', action='store_true',
                        help='Generate train/val/test splits')
    parser.add_argument('--validate', action='store_true',
                        help='Validate output folder structure')
    parser.add_argument('--all', action='store_true',
                        help='Run preprocess → fingerprint → splits → validate')
    parser.add_argument('--skip-fingerprint', action='store_true',
                        help='Skip fingerprint computation during preprocessing')

    args = parser.parse_args()

    config = load_config(args.config)
    config_path = Path(args.config)
    output_dir = Path(config['OUTPUT_DIR'])

    if not any([args.preprocess, args.fingerprint, args.splits, args.validate, args.all]):
        parser.print_help()
        return

    if args.all:
        args.preprocess = True
        args.fingerprint = True
        args.splits = True
        args.validate = True

    if args.preprocess:
        pipeline = VOIPreprocessor(config)
        pipeline.run_batch(skip_fingerprint=args.skip_fingerprint)

    if args.fingerprint:
        compute_fingerprint_from_outputs(output_dir, config)

    if args.splits:
        code = run_splits(config_path)
        if code != 0:
            sys.exit(code)

    if args.validate:
        code = validate_output_structure(output_dir)
        if code != 0:
            sys.exit(code)


if __name__ == '__main__':
    main()
