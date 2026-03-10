"""CLI entry point for the DICOM → NIfTI converter.

Usage:
    python -m src.converter.convert -i <dicom_dir> -o <output_dir> [-c <csv>]

Output structure:
    <output_dir>/nifti/NN_case_YYYYY_0000.nii.gz   (all files flat)
    <output_dir>/manifest.csv                        (group + phase columns)
    <output_dir>/conversion_summary.json
"""

import argparse
import sys

from .pipeline import run


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DICOM → NIfTI converter  (single-stage, flat output)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Output layout:
  <output>/nifti/NN_case_YYYYY_0000.nii.gz   — all volumes flat
  <output>/manifest.csv                       — full metadata (group, phase, …)
  <output>/conversion_summary.json

Examples:
  python -m converter.convert \\
      -i data/tcga_dicom \\
      -o data/dataset/Dataset820 \\
      -c data/filtered_vessel_evaluation.csv

  python -m converter.convert \\
      -i data/ukb2025_raw \\
      -o data/dataset/Dataset920
""",
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Root directory containing DICOM files",
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Destination directory (nifti/ subfolder created automatically)",
    )
    parser.add_argument(
        "-c", "--csv", default=None,
        help="Patient classification CSV (columns: patient_id + group/class)",
    )
    parser.add_argument(
        "--start-case-id", type=int, default=1,
        help="Starting case number (default: 1)",
    )
    parser.add_argument(
        "--min-slices", type=int, default=10,
        help="Minimum slices to keep a series (default: 10)",
    )
    args = parser.parse_args()

    try:
        run(
            input_dir=args.input,
            output_dir=args.output,
            classification_csv=args.csv,
            start_case_id=args.start_case_id,
            min_slices=args.min_slices,
        )
        return 0
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
