"""
Edge case analyzer for processed VOI datasets.
Generates edge_cases.json with validation results.
"""
import json
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from tqdm import tqdm

from .validators import run_all_validators


class EdgeCaseAnalyzer:
    """Analyze processed dataset for edge cases."""
    
    def __init__(self, config: Dict, output_dir: Path):
        self.config = config
        self.output_dir = Path(output_dir)
    
    def analyze_dataset(self, dataset_path: Optional[Path] = None) -> Dict:
        """
        Analyze all cases in dataset.json
        
        Returns:
            Dictionary with summary and per-case validation results
        """
        if dataset_path is None:
            dataset_path = self.output_dir / 'dataset.json'
        
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        results = {
            'config': {
                'MIN_VOI_SIZE': self.config.get('MIN_VOI_SIZE', [96, 96, 96]),
                'HU_RANGE': self.config.get('HU_RANGE', [-200, 300]),
                'BOUNDARY_MARGIN': self.config.get('BOUNDARY_MARGIN', 5)
            },
            'summary': {
                'total_cases': 0,
                'clean': 0,
                'flagged': 0,
                'critical': 0,
                'failed_to_load': 0
            },
            'cases': {}
        }
        
        # Collect all cases
        all_cases = self._collect_cases(dataset)
        results['summary']['total_cases'] = len(all_cases)
        
        # Analyze each case
        for case_key, case_info in tqdm(all_cases.items(), desc="Analyzing edge cases"):
            case_result = self._analyze_case(case_key, case_info)
            
            if case_result is None:
                results['summary']['failed_to_load'] += 1
                continue
            
            results['cases'][case_key] = case_result
            results['summary'][case_result['overall_status']] += 1
        
        # Compute validator statistics
        results['summary']['validator_stats'] = self._compute_stats(results['cases'])
        
        return results
    
    def _collect_cases(self, dataset: Dict) -> Dict:
        """Collect all case paths from dataset.json"""
        cases = {}
        
        for patient_id, patient_data in dataset.get('patients', {}).items():
            class_folder = patient_data.get('class', 'NG')
            
            # Get mask files
            mask_files = patient_data.get('files', {}).get('mask', [])
            
            for mask_file in mask_files:
                case_key = f"{patient_id}/{Path(mask_file).stem}"
                cases[case_key] = {
                    'patient_id': patient_id,
                    'class': class_folder,
                    'mask_path': self.output_dir / mask_file,
                    'seg_path': self.output_dir / mask_file.replace('/mask/', '/segmentation/'),
                    'bbox': patient_data.get('bbox', {}),
                    'original_size': patient_data.get('original_size', [])
                }
        
        return cases
    
    def _analyze_case(self, case_key: str, case_info: Dict) -> Optional[Dict]:
        """Analyze a single case"""
        try:
            mask_path = case_info['mask_path']
            seg_path = case_info['seg_path']
            
            if not mask_path.exists() or not seg_path.exists():
                return None
            
            mask_array = np.load(mask_path)
            image_array = np.load(seg_path)
            
            # Get bbox - try to extract from case info
            bbox = case_info.get('bbox', {})
            # Extract side from filename (e.g., "case_L" -> "L")
            side = case_key.split('_')[-1] if '_' in case_key else 'L'
            bbox_tuple = tuple(bbox.get(side, [0, 0, 0, 96, 96, 96]))
            
            original_size = tuple(case_info.get('original_size', image_array.shape[::-1]))
            
            return run_all_validators(
                image_array=image_array,
                mask_array=mask_array,
                bbox=bbox_tuple,
                original_size=original_size,
                config=self.config
            )
            
        except Exception as e:
            print(f"Error analyzing {case_key}: {e}")
            return None
    
    def _compute_stats(self, cases: Dict) -> Dict:
        """Compute per-validator statistics"""
        if not cases:
            return {}
        
        validators = ['boundary_contact', 'tumor_centering', 'intensity', 'voi_size', 'multi_lesion']
        stats = {}
        
        for v in validators:
            passed = sum(1 for c in cases.values() if c['validators'].get(v, {}).get('passed', True))
            total = len(cases)
            stats[v] = {
                'passed': passed,
                'failed': total - passed,
                'pass_rate': round(passed / total, 3) if total > 0 else 0
            }
        
        return stats
    
    def save_results(self, results: Dict, output_path: Optional[Path] = None) -> Path:
        """Save edge cases JSON"""
        if output_path is None:
            output_path = self.output_dir / 'edge_cases.json'
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        s = results['summary']
        print(f"\n{'='*50}")
        print(f"Edge Case Analysis Complete")
        print(f"{'='*50}")
        print(f"  Total:    {s['total_cases']}")
        print(f"  Clean:    {s['clean']} ({s['clean']/max(1,s['total_cases'])*100:.1f}%)")
        print(f"  Flagged:  {s['flagged']} ({s['flagged']/max(1,s['total_cases'])*100:.1f}%)")
        print(f"  Critical: {s['critical']} ({s['critical']/max(1,s['total_cases'])*100:.1f}%)")
        print(f"\nSaved to: {output_path}")
        
        return output_path
