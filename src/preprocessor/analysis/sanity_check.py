import json
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from tqdm import tqdm

from .validators import run_all_validators


class SanityChecker:
    
    def __init__(self, config: Dict, output_dir: Path):
        self.config = config
        self.output_dir = Path(output_dir)
    
    def check_dataset(self, dataset_path: Optional[Path] = None) -> Dict:
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
        
        all_cases = self._collect_cases(dataset)
        results['summary']['total_cases'] = len(all_cases)
        
        for case_key, case_info in tqdm(all_cases.items(), desc="Sanity check"):
            case_result = self._check_case(case_key, case_info)
            
            if case_result is None:
                results['summary']['failed_to_load'] += 1
                continue
            
            results['cases'][case_key] = case_result
            results['summary'][case_result['overall_status']] += 1
        
        results['summary']['validator_stats'] = self._compute_stats(results['cases'])
        
        return results
    
    def _collect_cases(self, dataset: Dict) -> Dict:
        cases = {}
        
        for patient_id, patient_data in dataset.get('patients', {}).items():
            class_folder = patient_data.get('class', 'NG')
            mask_files = patient_data.get('files', {}).get('mask', [])
            
            for mask_file in mask_files:
                case_key = f"{patient_id}/{Path(mask_file).stem}"
                seg_file = mask_file.replace('/mask/', '/segmentation/')
                
                cases[case_key] = {
                    'patient_id': patient_id,
                    'class': class_folder,
                    'mask_path': self.output_dir / mask_file,
                    'seg_path': self.output_dir / seg_file,
                }
        
        return cases
    
    def _check_case(self, case_key: str, case_info: Dict) -> Optional[Dict]:
        try:
            mask_path = case_info['mask_path']
            seg_path = case_info['seg_path']
            
            if not mask_path.exists() or not seg_path.exists():
                return None
            
            mask_array = np.load(mask_path)
            image_array = np.load(seg_path)
            
            parts = mask_path.stem.split('_')
            side = parts[-1] if parts and parts[-1] in ['L', 'R'] else None
            
            bbox_tuple = None
            original_size = tuple(image_array.shape[::-1])
            
            return run_all_validators(
                image_array=image_array,
                mask_array=mask_array,
                bbox=bbox_tuple,
                original_size=original_size,
                config=self.config
            )
            
        except Exception as e:
            print(f"Error checking {case_key}: {e}")
            return None
    
    def _compute_stats(self, cases: Dict) -> Dict:
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
        if output_path is None:
            output_path = self.output_dir / 'sanity_check.json'
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        s = results['summary']
        print(f"\n{'='*50}")
        print(f"Sanity Check Complete")
        print(f"{'='*50}")
        print(f"  Total:    {s['total_cases']}")
        print(f"  Clean:    {s['clean']} ({s['clean']/max(1,s['total_cases'])*100:.1f}%)")
        print(f"  Flagged:  {s['flagged']} ({s['flagged']/max(1,s['total_cases'])*100:.1f}%)")
        print(f"  Critical: {s['critical']} ({s['critical']/max(1,s['total_cases'])*100:.1f}%)")
        print(f"\nSaved to: {output_path}")
        
        return output_path
