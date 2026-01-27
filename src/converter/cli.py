"""
Command-line interface for Universal DICOM to NIfTI Converter
"""

import argparse
import traceback

from .core.converter import UniversalDICOMConverter


def print_summary_statistics(summary: dict):
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Dataset Type:        {summary['dataset_type'].upper()}")
    print(f"Total Patients:      {summary['total_patients']}")
    print(f"Total Series Found:  {summary['total_series_discovered']}")
    print(f"AXIAL Series:        {summary['axial_series']}")
    print(f"Skipped (Non-AXIAL): {summary['skipped_non_axial']}")
    print(f"Skipped (Few Slices):{summary.get('skipped_few_slices', 0)}")
    print(f"Skipped (Quality):   {summary['skipped_quality']}")
    print(f"Successful:          {summary['successful_conversions']}")
    print(f"Failed:              {summary['failed_conversions']}")
    
    print("\nModality Distribution:")
    for modality, count in sorted(summary.get('modality_distribution', {}).items()):
        print(f"  {modality:>3}: {count:>4}")
    
    print("\nQuality Distribution:")
    quality_dist = summary.get('quality_distribution', {})
    print(f"  OK:       {quality_dist.get('ok', 0):>4} (no spacing issues)")
    print(f"  Minor:    {quality_dist.get('minor', 0):>4} (<10mm nonuniformity)")
    print(f"  Moderate: {quality_dist.get('moderate', 0):>4} (10-50mm nonuniformity)")
    print(f"  Critical: {quality_dist.get('critical', 0):>4} (>50mm - skipped)")
    
    print("\nPatient ID Sources:")
    id_sources = summary.get('patient_id_sources', {})
    for source, count in sorted(id_sources.items()):
        print(f"  {source:>25}: {count:>4}")
    
    print("\nClass Distribution:")
    for class_label, count in sorted(summary['class_distribution'].items()):
        print(f"  {class_label:>3}: {count:>4}")
    
    print("\nProtocol Distribution (AXIAL only):")
    for protocol, count in sorted(summary['protocol_distribution'].items()):
        print(f"  {protocol:>10}: {count:>4}")
    
    print("\nScan Plane Distribution (All):")
    for plane, count in sorted(summary['scan_plane_distribution'].items()):
        print(f"  {plane:>10}: {count:>4}")
    
    if summary['skipped_non_axial'] > 0:
        print("\nSkipped Breakdown:")
        for plane, count in sorted(summary['skipped_breakdown'].items()):
            print(f"  {plane:>10}: {count:>4}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Universal DICOM to NIfTI Converter with automatic dataset detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # TCGA with classification CSV
  python -m src.converter.cli \\
    --input data/tcga_dicom/tcga_kirc \\
    --output data/dataset/tcga_kirc_nii_v3 \\
    --csv data/filtered_vessel_evaluation.csv
  
  # UKBonn without classification (all NG)
  python -m src.converter.cli \\
    --input data/dataset/ukb2025_raw \\
    --output data/dataset/ukb2025_nii
  
  # Custom start case ID
  python -m src.converter.cli \\
    --input data/dicom \\
    --output data/nifti \\
    --start-case-id 100
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Input directory containing DICOM files'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Output directory for NIfTI files (organized by class)'
    )
    
    parser.add_argument(
        '-c', '--csv',
        type=str,
        default=None,
        help='Optional CSV file with patient classifications'
    )
    
    parser.add_argument(
        '--start-case-id',
        type=int,
        default=1,
        help='Starting case number (default: 1)'
    )
    
    args = parser.parse_args()
    
    # Create converter
    converter = UniversalDICOMConverter(
        input_dir=args.input,
        output_dir=args.output,
        classification_csv=args.csv,
        start_case_id=args.start_case_id
    )
    
    # Run conversion
    try:
        summary = converter.run()
        print_summary_statistics(summary)
        return 0
        
    except Exception as e:
        print(f"\nERROR: Conversion failed: {e}")
        import sys
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
