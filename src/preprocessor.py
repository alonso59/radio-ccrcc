#!/usr/bin/env python3
"""
VOI Preprocessor Pipeline

Modes:
  - Full preprocessing (default): extract VOIs, compute fingerprints, run analysis
  - Fingerprint-only: recompute fingerprints from saved outputs
  - Analyze-only: run edge case analysis on existing dataset
  - Splits-only: generate train/val/test splits from existing dataset

Usage:
  python src/preprocessor.py --config config/preprocessor_config.yaml
  python src/preprocessor.py --config config/preprocessor_config.yaml --fingerprint-only
  python src/preprocessor.py --config config/preprocessor_config.yaml --splits-only
"""
import sys
import argparse
from pathlib import Path
from preprocessor import VOIPreprocessor, load_config, EdgeCaseAnalyzer
from preprocessor.utils.metrics import compute_fingerprint_from_outputs


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='VOI Preprocessor Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full preprocessing
  python src/preprocessor.py --config config/preprocessor_config.yaml

  # Recompute fingerprints from saved outputs (no preprocessing)
  python src/preprocessor.py --config config/preprocessor_config.yaml --fingerprint-only

  # Generate splits only
  python src/preprocessor.py --config config/preprocessor_config.yaml --splits-only

  # Edge case analysis only
  python src/preprocessor.py --config config/preprocessor_config.yaml --analyze-only
        """
    )
    parser.add_argument('--config', default='config/preprocessor_config.yaml',
                       help='Path to configuration file (default: config/preprocessor_config.yaml)')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only run edge case analysis on existing dataset')
    parser.add_argument('--fingerprint-only', action='store_true',
                       help='Only recompute fingerprints from saved outputs (requires SAVE_IMAGES=true)')
    parser.add_argument('--splits-only', action='store_true',
                       help='Only generate splits from existing dataset')
    parser.add_argument('--skip-analysis', action='store_true',
                       help='Skip edge case analysis after preprocessing')
    parser.add_argument('--skip-fingerprint', action='store_true',
                       help='Skip fingerprint computation during preprocessing')
    
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = Path(config['OUTPUT_DIR'])
    
    # Mode: Fingerprint only
    if args.fingerprint_only:
        if not output_dir.exists():
            print(f"❌ Output directory not found: {output_dir}")
            return
        compute_fingerprint_from_outputs(output_dir, config)
        return
    
    # Mode: Splits only
    if args.splits_only:
        if not output_dir.exists():
            print(f"❌ Output directory not found: {output_dir}")
            return
        # Import and run generate_splits with config
        import subprocess
        splits_cmd = [sys.executable, 'src/generate_splits.py', '--config', args.config]
        print(f"Running: {' '.join(splits_cmd)}")
        subprocess.run(splits_cmd)
        return
    
    # Mode: Analyze only
    if args.analyze_only:
        dataset_path = output_dir / 'dataset.json'
        if not dataset_path.exists():
            print(f"❌ dataset.json not found at {dataset_path}")
            return
        
        analyzer = EdgeCaseAnalyzer(config, output_dir)
        results = analyzer.analyze_dataset(dataset_path)
        analyzer.save_results(results)
        return
    
    # Mode: Full preprocessing pipeline
    pipeline = VOIPreprocessor(config)
    results = pipeline.run_batch(skip_fingerprint=args.skip_fingerprint)
    
    # Count results
    successful = sum(1 for r in results.values() if r.get('status') == 'success')
    failed = sum(1 for r in results.values() if r.get('status') == 'failed')
    
    print(f"\nPipeline complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    
    # Run edge case analysis unless skipped
    if not args.skip_analysis:
        dataset_path = output_dir / 'dataset.json'
        if dataset_path.exists():
            print("\nRunning edge case analysis...")
            analyzer = EdgeCaseAnalyzer(config, output_dir)
            edge_results = analyzer.analyze_dataset(dataset_path)
            analyzer.save_results(edge_results)


if __name__ == '__main__':
    main()
