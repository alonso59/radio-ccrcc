import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from .config import validate_config
from ..utils.io import (
    detect_input_format,
    discover_files,
    load_case,
    save_voi,
)
from .processing import (
    clip_hu_range,
    convert_to_ras,
    resample_to_spacing,
    separate_kidneys,
    extract_voi,
    apply_mask_to_voi,
    validate_voi,
)
from ..utils.metrics import (
    compute_tumor_metrics,
    collect_fingerprint_data,
    generate_fingerprint,
)


# =============================================================================
# VOI PREPROCESSOR CLASS
# =============================================================================
class VOIPreprocessor:

    def __init__(self, config: Dict):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        validate_config(config)
        
        self.image_folder = Path(config['IMAGE_FOLDER'])
        self.mask_folder = Path(config['MASK_FOLDER'])
        self.output_dir = Path(config['OUTPUT_DIR'])
        
        self.file_list: Dict[str, Dict] = {}
        self.processed_data: Dict[str, Dict] = {}
        
        # Segmentation fingerprint (mask > 0 voxels)
        self.fingerprint_data: Dict[str, Any] = {
            'intensities': [],
            'shapes': [],
            'spacings': [],
            'total_foreground_voxels': 0,
            'filtered_voxels': 0,
        }
        
        # Image fingerprint (all VOI voxels)
        self.image_fingerprint_data: Dict[str, Any] = {
            'intensities': [],
            'shapes': [],
            'spacings': [],
            'total_foreground_voxels': 0,
            'filtered_voxels': 0,
        }
        
        self.patient_files: Dict[str, List[str]] = defaultdict(list)
        
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize file list and detect input format."""
        # Check base folders exist
        for folder, name in [(self.image_folder, "Image"), (self.mask_folder, "Mask")]:
            if not folder.exists():
                raise ValueError(f"{name} folder not found: {folder}")
        
        # Get class filter from config
        class_filter = self.config.get('CLASS_FILTER', None)
        if class_filter == []:
            class_filter = None  # Empty list means all
        
        # Discover class subfolders
        if class_filter is None:
            # Auto-discover all class subfolders
            class_folders = sorted([d.name for d in self.image_folder.iterdir() if d.is_dir()])
            if not class_folders:
                # No subfolders - treat as flat structure
                class_folders = ['']
            print(f"Auto-discovered class folders: {class_folders}")
        else:
            # Use specified classes
            class_folders = class_filter if isinstance(class_filter, list) else [class_filter]
            print(f"Processing class folders: {class_folders}")
        
        # Detect input format (check first available folder)
        input_format = self.config.get('INPUT_FORMAT', 'auto')
        if input_format == 'auto':
            check_folder = self.image_folder / class_folders[0] if class_folders[0] else self.image_folder
            input_format = detect_input_format(check_folder)
        
        self.input_format = input_format
        print(f"Detected input format: {input_format}")
        
        # Discover files across all class folders
        all_files = {}
        for class_name in class_folders:
            img_folder = self.image_folder / class_name if class_name else self.image_folder
            msk_folder = self.mask_folder / class_name if class_name else self.mask_folder
            
            if not img_folder.exists():
                print(f"Warning: Image folder not found: {img_folder}, skipping")
                continue
            if not msk_folder.exists():
                print(f"Warning: Mask folder not found: {msk_folder}, skipping")
                continue
            
            class_files = discover_files(
                img_folder,
                msk_folder,
                self.config['IMAGE_SUFFIX'],
                self.config['MASK_SUFFIX'],
                self.config.get('PATIENT_ID_PATTERN'),
            )
            
            # Update subfolder to class name for proper output organization
            for key, data in class_files.items():
                if class_name:
                    data['subfolder'] = class_name
            
            all_files.update(class_files)
            print(f"  {class_name or '(root)'}: Found {len(class_files)} cases")
        
        self.file_list = all_files
        
        print(f"\nTotal: {len(self.file_list)} matching image-mask pairs")
        
        patients = set(v['patient_id'] for v in self.file_list.values())
        print(f"Found {len(patients)} unique patients")
        
        # Show distribution of scans per patient
        from collections import Counter
        patient_scan_counts = Counter(v['patient_id'] for v in self.file_list.values())
        multiple_scans = sum(1 for count in patient_scan_counts.values() if count > 1)
        if multiple_scans > 0:
            print(f"  → {multiple_scans} patients have multiple scans (longitudinal data)")
    
    def load_from_dataset(self) -> None:
        """Load processed data from existing dataset.json."""
        dataset_path = self.output_dir / 'dataset.json'
        if not dataset_path.exists():
            raise FileNotFoundError(f"dataset.json not found at {dataset_path}")
        
        print(f"Loading dataset from: {dataset_path}")
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        # Reconstruct processed_data and patient_files from dataset.json
        for patient_id, patient_data in dataset.get('patients', {}).items():
            # Get all files for this patient
            files = []
            for file_type in ['mask', 'segmentation']:
                files.extend(patient_data.get('files', {}).get(file_type, []))
            
            self.patient_files[patient_id] = files
            
            # Create a processed_data entry for each case
            for case_name in patient_data.get('cases', []):
                case_key = f"{patient_data.get('class', '')}/{case_name}" if patient_data.get('class') else case_name
                self.processed_data[case_key] = {
                    'patient_id': patient_id,
                    'case_name': case_name,
                    'subfolder': patient_data.get('class', ''),
                    'status': 'success',
                    'kidneys_found': patient_data.get('kidneys', []),
                }
        
        print(f"Loaded {len(self.patient_files)} patients with {len(self.processed_data)} cases")
    
    # =========================================================================
    # SINGLE CASE PROCESSING
    # =========================================================================
    
    def process_case(self, case_key: str, verbose: bool = False) -> Dict:
        """
        Process a single case.
        
        Args:
            case_key: Case identifier
            verbose: Whether to print progress
            
        Returns:
            Dictionary of processing results
        """
        case_data = self.file_list[case_key]
        
        if verbose:
            print(f"Processing: {case_key}")
        
        image, mask = load_case(
            case_data,
            self.input_format,
            self.config.get('DEFAULT_NPY_SPACING'),
        )
        
        hu_min = self.config.get('HU_CLIP_MIN', -200)
        hu_max = self.config.get('HU_CLIP_MAX', 300)
        if hu_min is not None and hu_max is not None:
            image = clip_hu_range(image, hu_min, hu_max)
        
        image, mask = convert_to_ras(
            image, mask,
            self.config.get('USE_RAS_ORIENTATION', True),
        )
        
        image, mask = resample_to_spacing(
            image, mask,
            self.config['TARGET_SPACING'],
        )
        
        bboxes = separate_kidneys(
            mask,
            self.config.get('BBOX_LABELS'),
        )
        
        results = {
            'case_key': case_key,
            'case_name': case_data['case_name'],
            'patient_id': case_data['patient_id'],
            'subfolder': case_data['subfolder'],
            'kidneys_found': list(bboxes.keys()),
        }
        
        for side, bbox_info in bboxes.items():
            voi_img, voi_mask, center = extract_voi(
                image, mask, bbox_info,
                self.config['EXPANSION_MM'],
                self.config['TARGET_SPACING'],
                self.config.get('MIN_VOI_SIZE'),
            )
            
            is_valid, reason = validate_voi(
                voi_img, voi_mask,
                self.config.get('MIN_KIDNEY_VOXELS', 0),
                self.config.get('MIN_TUMOR_VOXELS', 0),
                self.config.get('MIN_HU_IN_RANGE_RATIO', 0),
            )
            
            if not is_valid:
                if verbose:
                    print(f"  {side} kidney skipped: {reason}")
                continue
            
            collect_fingerprint_data(
                voi_img, voi_mask, 
                self.fingerprint_data,
                self.image_fingerprint_data,
            )
            
            voi_seg = apply_mask_to_voi(
                voi_img, voi_mask,
                self.config.get('MASK_LABELS'),
            )
            
            tumor_metrics = compute_tumor_metrics(voi_img, voi_mask)
            
            results[f'{side}_image'] = voi_img
            results[f'{side}_mask'] = voi_mask
            results[f'{side}_segmented'] = voi_seg
            results[f'{side}_metrics'] = tumor_metrics
            results[f'{side}_shape'] = voi_img.GetSize()
            results[f'{side}_spacing'] = voi_img.GetSpacing()
        
        return results
    
    # =========================================================================
    # BATCH PROCESSING
    # =========================================================================
    
    def run_batch(self, verbose: bool = True, skip_fingerprint: bool = False) -> Dict:
        """
        Run preprocessing on all cases.
        
        Args:
            verbose: Whether to print progress
            skip_fingerprint: Whether to skip fingerprint computation
            
        Returns:
            Dictionary of all results
        """
        print(f"\n{'='*70}")
        print("VOI PREPROCESSOR - BATCH PROCESSING")
        print(f"{'='*70}")
        print(f"Total cases: {len(self.file_list)}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*70}\n")
        
        all_results = {}
        failed_cases = []
        
        for case_key in tqdm(self.file_list.keys(), desc="Processing"):
            try:
                results = self.process_case(case_key, verbose=False)
                case_data = self.file_list[case_key]
                
                saved_files = save_voi(
                    results,
                    case_data,
                    self.output_dir,
                    self.config.get('SAVE_MASK', True),
                    self.config.get('SAVE_SEGMENTATION', True),
                    self.config.get('SAVE_IMAGES', False),
                )
                
                patient_id = case_data['patient_id']
                self.patient_files[patient_id].extend(saved_files)
                
                report = self._create_case_report(results, saved_files, case_data)
                all_results[case_key] = report
                self.processed_data[case_key] = report
                
            except Exception as e:
                if verbose:
                    print(f"\nError: {case_key} - {e}")
                failed_cases.append(case_key)
                all_results[case_key] = {
                    'case_key': case_key,
                    'status': 'failed',
                    'error': str(e),
                }
        
        self._generate_dataset_json(all_results, failed_cases)
        
        if not skip_fingerprint:
            generate_fingerprint(
                self.fingerprint_data,
                self.processed_data,
                self.patient_files,
                self.config,
                self.output_dir,
                self.image_fingerprint_data,
            )
        
        self._print_summary(all_results, failed_cases)
        
        return all_results
    
    def _create_case_report(self, results: Dict, saved_files: List[str], case_data: Dict) -> Dict:
        """Create serializable report for a case."""
        report = {
            'case_key': results['case_key'],
            'case_name': results['case_name'],
            'patient_id': results['patient_id'],
            'subfolder': results['subfolder'],
            'status': 'success',
            'kidneys_found': results['kidneys_found'],
            'files': saved_files,
            'source_image_path': case_data.get('image_path'),
            'source_mask_path': case_data.get('mask_path'),
        }
        
        for side in ['L', 'R']:
            if f'{side}_metrics' in results:
                report[f'{side}_kidney'] = {
                    'shape': results[f'{side}_shape'],
                    'spacing': results[f'{side}_spacing'],
                    'metrics': results[f'{side}_metrics'],
                }
        
        return report
    
    # =========================================================================
    # DATASET JSON GENERATION
    # =========================================================================
    
    def _generate_dataset_json(self, results: Dict, failed: List[str]) -> None:
        """
        Generate dataset.json metadata file.
        
        Structure:
        {
            "name": "TCGA_KIRC_VOI",
            "file_format": "{case_name}_{side}.npy",
            "file_structure": "{output_dir}/{type}/{class}/{patient_id}/{case_name}_{side}.npy",
            "example_path": "segmentation/A/TCGA-B0-5399/TCGA-B0-5399_A_00_fallback_case_00113_L.npy",
            ...
        }
        """
        successful = [k for k, v in results.items() if v.get('status') == 'success']
        
        patients = {}
        all_files = {'mask': [], 'segmentation': [], 'images': []}
        
        for case_key, data in results.items():
            if data.get('status') != 'success':
                continue
            
            patient_id = data['patient_id']
            if patient_id not in patients:
                patients[patient_id] = {
                    'class': data['subfolder'],
                    'cases': [],
                    'kidneys': set(),
                    'files': {'mask': [], 'segmentation': [], 'images': []},
                }
            
            patients[patient_id]['cases'].append(data['case_name'])
            patients[patient_id]['kidneys'].update(data.get('kidneys_found', []))
            
            # Categorize files by type
            for f in data.get('files', []):
                if f.startswith('mask/'):
                    patients[patient_id]['files']['mask'].append(f)
                    all_files['mask'].append(f)
                elif f.startswith('segmentation/'):
                    patients[patient_id]['files']['segmentation'].append(f)
                    all_files['segmentation'].append(f)
                elif f.startswith('images/'):
                    patients[patient_id]['files']['images'].append(f)
                    all_files['images'].append(f)
        
        # Convert sets to lists for JSON
        for pid in patients:
            patients[pid]['kidneys'] = sorted(list(patients[pid]['kidneys']))
        
        # Get example file path
        example_path = None
        if all_files['segmentation']:
            example_path = all_files['segmentation'][0]
        elif all_files['images']:
            example_path = all_files['images'][0]
        elif all_files['mask']:
            example_path = all_files['mask'][0]
        
        dataset = {
            'version': '1.1.0',
            'dataset_id': self.config.get('DATASET_ID'),
            'channel_names': {'0': 'CT'},
            'labels': self.config.get('LABELS', {1: 'kidney', 2: 'tumor'}),
            'file_ending': '.npy',
            'file_format': '{case_name}_{side}.npy',
            'file_structure': '{output_dir}/{type}/{class}/{patient_id}/{case_name}_{side}.npy',
            'example_path': example_path,
            'output_types': {
                'mask': 'Segmentation labels (uint8)',
                'segmentation': 'Masked CT image with background=-1000 HU (float32)',
                'images': 'Full CT VOI without masking (float32)',
            },
            'source': {
                'image_folder': str(self.image_folder),
                'mask_folder': str(self.mask_folder),
                'image_suffix': self.config.get('IMAGE_SUFFIX'),
                'mask_suffix': self.config.get('MASK_SUFFIX'),
                'input_format': self.input_format,
            },
            'statistics': {
                'num_cases': len(successful),
                'num_patients': len(patients),
                'num_failed': len(failed),
                'num_mask_files': len(all_files['mask']),
                'num_segmentation_files': len(all_files['segmentation']),
                'num_image_files': len(all_files['images']),
            },
            'processing_config': {
                'target_spacing': self.config['TARGET_SPACING'],
                'expansion_mm': self.config['EXPANSION_MM'],
                'min_voi_size': self.config.get('MIN_VOI_SIZE'),
                'use_ras_orientation': self.config.get('USE_RAS_ORIENTATION', True),
                'bbox_labels': self.config.get('BBOX_LABELS'),
                'mask_labels': self.config.get('MASK_LABELS'),
            },
            'failed_cases': failed,
            'patients': patients
        }
        
        output_path = self.output_dir / 'dataset.json'
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"Saved: {output_path}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    def _print_summary(self, results: Dict, failed: List[str]) -> None:
        """Print processing summary."""
        successful = len([r for r in results.values() if r.get('status') == 'success'])
        
        print(f"\n{'='*70}")
        print("PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Total cases: {len(self.file_list)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(failed)}")
        print(f"Unique patients: {len(self.patient_files)}")
        
        if failed:
            print(f"\nFailed cases: {', '.join(failed[:10])}")
            if len(failed) > 10:
                print(f"  ... and {len(failed) - 10} more")
        
        print(f"\nOutputs saved to: {self.output_dir}")
        print(f"{'='*70}\n")
