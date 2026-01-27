#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
from preprocessor import VOIPreprocessor, load_config, SanityChecker


def main():
    parser = argparse.ArgumentParser(description='VOI Preprocessor Pipeline')
    parser.add_argument('--config', default='config/preprocessor_config.yaml',
                       help='Path to configuration file (default: config/preprocessor_config.yaml)')
    
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = Path(config['OUTPUT_DIR'])
    
    pipeline = VOIPreprocessor(config)
    results = pipeline.run_batch()
    
    successful = sum(1 for r in results.values() if r.get('status') == 'success')
    failed = sum(1 for r in results.values() if r.get('status') == 'failed')
    
    print(f"\nPipeline complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    
    dataset_path = output_dir / 'dataset.json'
    if dataset_path.exists():
        print("\nRunning sanity check...")
        checker = SanityChecker(config, output_dir)
        sanity_results = checker.check_dataset(dataset_path)
        checker.save_results(sanity_results)


if __name__ == '__main__':
    main()
