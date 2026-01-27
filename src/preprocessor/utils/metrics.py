"""Metrics: tumor statistics, fingerprint generation, intensity analysis."""

import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

import numpy as np
import SimpleITK as sitk

from .common import TUMOR_LABEL, BACKGROUND_HU

# HU bounds for foreground statistics (excludes air, bone, artifacts)
HU_LOWER_BOUND = BACKGROUND_HU  # -200
HU_UPPER_BOUND = 300


# =============================================================================
# VOXEL COLLECTION (NO SAMPLING - EXACT STATISTICS)
# =============================================================================
def collect_voxels_with_hu_filter(
    values: np.ndarray,
    collector: List[float],
    hu_lower: float = HU_LOWER_BOUND,
    hu_upper: float = HU_UPPER_BOUND
) -> int:
    """
    Collect all voxels within HU bounds for exact statistics.
    
    Args:
        values: Array of intensity values
        collector: List to append filtered values to
        hu_lower: Lower HU bound (exclusive)
        hu_upper: Upper HU bound (inclusive)
        
    Returns:
        Count of voxels added
    """
    valid_mask = (values > hu_lower) & (values <= hu_upper)
    valid_values = values[valid_mask]
    collector.extend(valid_values.astype(np.float32).tolist())
    return len(valid_values)


# =============================================================================
# TUMOR METRICS
# =============================================================================
def compute_tumor_metrics(image: sitk.Image, mask: sitk.Image) -> Dict:
    """
    Compute tumor metrics for a VOI.
    
    Args:
        image: VOI CT image
        mask: VOI segmentation mask
        
    Returns:
        Dictionary of tumor metrics
    """
    img_array = sitk.GetArrayFromImage(image)
    mask_array = sitk.GetArrayFromImage(mask)
    spacing = image.GetSpacing()
    
    tumor_mask = (mask_array == TUMOR_LABEL)
    tumor_voxels = tumor_mask.sum()
    
    if tumor_voxels == 0:
        return {'has_tumor': False, 'tumor_voxels': 0}
    
    voxel_vol = np.prod(spacing)
    tumor_hu = img_array[tumor_mask]
    
    return {
        'has_tumor': True,
        'tumor_voxels': int(tumor_voxels),
        'tumor_volume_mm3': float(tumor_voxels * voxel_vol),
        'tumor_volume_cm3': float(tumor_voxels * voxel_vol / 1000),
        'mean_hu': float(tumor_hu.mean()),
        'std_hu': float(tumor_hu.std()),
        'min_hu': float(tumor_hu.min()),
        'max_hu': float(tumor_hu.max()),
    }


def collect_fingerprint_data(
    image: sitk.Image,
    mask: sitk.Image,
    fingerprint_data: Dict[str, Any],
    image_fingerprint_data: Dict[str, Any] = None
) -> None:
    """
    Collect intensity statistics for fingerprint generation.
    
    Collects ALL voxels (no sampling) with HU bounds filtering:
    - Segmentation fingerprint: voxels where mask > 0
    - Image fingerprint: ALL voxels in the VOI
    
    Args:
        image: VOI CT image
        mask: VOI segmentation mask
        fingerprint_data: Accumulator for segmentation stats (mask > 0 region)
        image_fingerprint_data: Accumulator for image stats (all VOI voxels)
    """
    img_array = sitk.GetArrayFromImage(image)
    mask_array = sitk.GetArrayFromImage(mask)
    
    # Collect shape/spacing for primary fingerprint
    fingerprint_data['shapes'].append(list(img_array.shape))
    fingerprint_data['spacings'].append(list(image.GetSpacing()))
    
    # SEGMENTATION fingerprint: foreground voxels only (mask > 0)
    fg_mask = mask_array > 0
    fg_intensities = img_array[fg_mask].flatten()
    
    seg_voxels = collect_voxels_with_hu_filter(
        fg_intensities,
        fingerprint_data['intensities']
    )
    fingerprint_data['total_foreground_voxels'] += len(fg_intensities)
    fingerprint_data['filtered_voxels'] += seg_voxels
    
    # IMAGE fingerprint: all voxels in the VOI
    if image_fingerprint_data is not None:
        image_fingerprint_data['shapes'].append(list(img_array.shape))
        image_fingerprint_data['spacings'].append(list(image.GetSpacing()))
        
        all_intensities = img_array.flatten()
        img_voxels = collect_voxels_with_hu_filter(
            all_intensities,
            image_fingerprint_data['intensities']
        )
        image_fingerprint_data['total_foreground_voxels'] += len(all_intensities)
        image_fingerprint_data['filtered_voxels'] += img_voxels


# =============================================================================
# TUMOR STATISTICS
# =============================================================================
def compute_tumor_statistics(processed_data: Dict[str, Dict]) -> Dict:
    """
    Compute aggregate tumor statistics from processed data.
    
    Args:
        processed_data: Dictionary of processed case data
        
    Returns:
        Dictionary of tumor statistics
    """
    tumor_volumes = []
    tumor_hu_means = []
    cases_with_tumor = 0
    cases_without_tumor = 0
    
    for case_data in processed_data.values():
        if case_data.get('status') != 'success':
            continue
        
        for side in ['L', 'R']:
            metrics = case_data.get(f'{side}_kidney', {}).get('metrics', {})
            if metrics.get('has_tumor', False):
                cases_with_tumor += 1
                if metrics.get('tumor_volume_cm3'):
                    tumor_volumes.append(metrics['tumor_volume_cm3'])
                if metrics.get('mean_hu'):
                    tumor_hu_means.append(metrics['mean_hu'])
            elif f'{side}_kidney' in case_data:
                cases_without_tumor += 1
    
    stats = {
        'cases_with_tumor': cases_with_tumor,
        'cases_without_tumor': cases_without_tumor,
        'tumor_ratio': cases_with_tumor / max(1, cases_with_tumor + cases_without_tumor),
    }
    
    if tumor_volumes:
        stats['tumor_volume_cm3'] = {
            'mean': float(np.mean(tumor_volumes)),
            'std': float(np.std(tumor_volumes)),
            'min': float(np.min(tumor_volumes)),
            'max': float(np.max(tumor_volumes)),
            'median': float(np.median(tumor_volumes)),
        }
    
    if tumor_hu_means:
        stats['tumor_hu'] = {
            'mean': float(np.mean(tumor_hu_means)),
            'std': float(np.std(tumor_hu_means)),
            'min': float(np.min(tumor_hu_means)),
            'max': float(np.max(tumor_hu_means)),
        }
    
    return stats


# =============================================================================
# FINGERPRINT GENERATION
# =============================================================================
def _compute_intensity_stats(intensities: np.ndarray) -> Dict:
    """Compute intensity statistics from collected voxels."""
    if len(intensities) == 0:
        return {}
    
    q25 = float(np.percentile(intensities, 25.0))
    q75 = float(np.percentile(intensities, 75.0))
    iqr = q75 - q25
    
    return {
        'mean': float(np.mean(intensities)),
        'std': float(np.std(intensities)),
        'min': float(np.min(intensities)),
        'max': float(np.max(intensities)),
        'median': float(np.median(intensities)),
        'percentile_00_5': float(np.percentile(intensities, 0.5)),
        'percentile_05_0': float(np.percentile(intensities, 5.0)),
        'percentile_25_0': q25,
        'percentile_75_0': q75,
        'percentile_95_0': float(np.percentile(intensities, 95.0)),
        'percentile_99_5': float(np.percentile(intensities, 99.5)),
        'iqr': iqr,
    }


def _compute_shape_stats(shapes: List) -> Dict:
    """Compute shape statistics from collected shapes."""
    if not shapes:
        return {}
    
    shapes_arr = np.array(shapes)
    return {
        'min_shape': shapes_arr.min(axis=0).tolist(),
        'max_shape': shapes_arr.max(axis=0).tolist(),
        'mean_shape': shapes_arr.mean(axis=0).tolist(),
        'median_shape': np.median(shapes_arr, axis=0).tolist(),
    }


def _build_fingerprint(
    fingerprint_data: Dict[str, Any],
    processed_data: Dict[str, Dict],
    patient_files: Dict[str, List],
    config: Dict,
    fingerprint_type: str
) -> Dict:
    """
    Build a fingerprint dictionary from collected data.
    
    Args:
        fingerprint_data: Accumulated voxel data
        processed_data: Processed case data
        patient_files: Patient to files mapping
        config: Configuration dictionary
        fingerprint_type: 'segmentation' or 'images'
    """
    intensities = np.array(fingerprint_data['intensities'], dtype=np.float32)
    shapes = fingerprint_data['shapes']
    
    total_voxels = fingerprint_data.get('total_foreground_voxels', 0)
    filtered_voxels = fingerprint_data.get('filtered_voxels', len(intensities))
    
    intensity_stats = _compute_intensity_stats(intensities)
    shape_stats = _compute_shape_stats(shapes)
    class_dist = _get_class_distribution(processed_data)
    tumor_stats = compute_tumor_statistics(processed_data)
    
    is_segmentation = fingerprint_type == 'segmentation'
    
    return {
        'fingerprint_type': fingerprint_type,
        'foreground_intensity_properties_per_channel': {
            '0': intensity_stats
        },
        'normalization': {
            'method': 'iqr',
            'percentile_25': intensity_stats.get('percentile_25_0'),
            'percentile_75': intensity_stats.get('percentile_75_0'),
            'median': intensity_stats.get('median'),
            'iqr': intensity_stats.get('iqr'),
        },
        'shape_statistics': shape_stats,
        'dataset_statistics': {
            'num_samples': len(shapes),
            'num_patients': len(patient_files),
            'num_classes': len(class_dist),
        },
        'class_distribution': class_dist,
        'tumor_statistics': tumor_stats,
        'voxel_collection_info': {
            'method': 'all_voxels',
            'region': 'mask > 0' if is_segmentation else 'full VOI',
            'hu_bounds': [HU_LOWER_BOUND, HU_UPPER_BOUND],
            'total_voxels': total_voxels,
            'filtered_voxels': filtered_voxels,
            'filter_ratio': filtered_voxels / max(1, total_voxels),
        },
        'processing_info': {
            'target_spacing': config['TARGET_SPACING'],
            'expansion_mm': config['EXPANSION_MM'],
            'min_voi_size': config.get('MIN_VOI_SIZE'),
            'bbox_labels': config.get('BBOX_LABELS'),
            'mask_labels': config.get('MASK_LABELS'),
        },
    }


def generate_fingerprint(
    fingerprint_data: Dict[str, List],
    processed_data: Dict[str, Dict],
    patient_files: Dict[str, List],
    config: Dict,
    output_dir: Path,
    image_fingerprint_data: Dict[str, List] = None
) -> None:
    """
    Generate dataset fingerprint JSON files with global statistics.
    
    Generates:
    - dataset_fingerprint_segmentation.json: stats from mask > 0 region
    - dataset_fingerprint_images.json: stats from full VOI (if image_fingerprint_data provided)
    
    Args:
        fingerprint_data: Accumulated segmentation data (mask > 0 voxels)
        processed_data: All processed case data
        patient_files: Patient to files mapping
        config: Configuration dictionary
        output_dir: Output directory path
        image_fingerprint_data: Accumulated image data (all VOI voxels)
    """
    # Generate segmentation fingerprint
    print(f"\nComputing SEGMENTATION fingerprint from {len(fingerprint_data['shapes'])} VOIs...")
    seg_intensities = np.array(fingerprint_data['intensities'], dtype=np.float32)
    print(f"  Region: mask > 0, HU bounds: ({HU_LOWER_BOUND}, {HU_UPPER_BOUND}]")
    print(f"  Voxels: {len(seg_intensities):,} filtered / {fingerprint_data.get('total_foreground_voxels', 0):,} total")
    
    seg_fingerprint = _build_fingerprint(
        fingerprint_data, processed_data, patient_files, config, 'segmentation'
    )
    
    seg_path = output_dir / 'dataset_fingerprint_segmentation.json'
    with open(seg_path, 'w') as f:
        json.dump(seg_fingerprint, f, indent=2)
    print(f"  Saved: {seg_path}")
    
    if seg_intensities.size > 0:
        stats = seg_fingerprint['normalization']
        print(f"  IQR normalization: Q25={stats['percentile_25']:.1f}, Q75={stats['percentile_75']:.1f}, IQR={stats['iqr']:.1f}")
    
    # Generate image fingerprint (if data provided)
    if image_fingerprint_data is not None and image_fingerprint_data['shapes']:
        print(f"\nComputing IMAGES fingerprint from {len(image_fingerprint_data['shapes'])} VOIs...")
        img_intensities = np.array(image_fingerprint_data['intensities'], dtype=np.float32)
        print(f"  Region: full VOI, HU bounds: ({HU_LOWER_BOUND}, {HU_UPPER_BOUND}]")
        print(f"  Voxels: {len(img_intensities):,} filtered / {image_fingerprint_data.get('total_foreground_voxels', 0):,} total")
        
        img_fingerprint = _build_fingerprint(
            image_fingerprint_data, processed_data, patient_files, config, 'images'
        )
        
        img_path = output_dir / 'dataset_fingerprint_images.json'
        with open(img_path, 'w') as f:
            json.dump(img_fingerprint, f, indent=2)
        print(f"  Saved: {img_path}")
        
        if img_intensities.size > 0:
            stats = img_fingerprint['normalization']
            print(f"  IQR normalization: Q25={stats['percentile_25']:.1f}, Q75={stats['percentile_75']:.1f}, IQR={stats['iqr']:.1f}")


def _get_class_distribution(processed_data: Dict[str, Dict]) -> Dict[str, int]:
    """Get distribution of classes."""
    distribution = defaultdict(int)
    for data in processed_data.values():
        if data.get('status') == 'success':
            cls = data.get('subfolder', 'unknown')
            distribution[cls] += 1
    return dict(distribution)


# =============================================================================
# STANDALONE FINGERPRINT COMPUTATION (from saved outputs)
# =============================================================================
def _scan_output_filesystem(output_dir: Path) -> Dict:
    """
    Scan output filesystem to reconstruct patient records.
    
    Fallback when dataset.json is empty or outdated.
    
    Returns:
        patients dict compatible with dataset.json structure
    """
    from collections import defaultdict
    
    patients = {}
    mask_dir = output_dir / 'mask'
    images_dir = output_dir / 'images'
    seg_dir = output_dir / 'segmentation'
    
    if not mask_dir.exists():
        return {}
    
    # Scan mask files (required)
    for class_folder in mask_dir.iterdir():
        if not class_folder.is_dir():
            continue
        
        class_label = class_folder.name
        
        for patient_folder in class_folder.iterdir():
            if not patient_folder.is_dir():
                continue
            
            patient_id = patient_folder.name
            
            if patient_id not in patients:
                patients[patient_id] = {
                    'class': class_label,
                    'cases': [],
                    'kidneys': [],
                    'files': {'mask': [], 'segmentation': [], 'images': []},
                }
            
            # Collect mask files
            for mask_file in patient_folder.glob('*.npy'):
                rel_path = f"mask/{class_label}/{patient_id}/{mask_file.name}"
                patients[patient_id]['files']['mask'].append(rel_path)
            
            # Collect images files
            images_patient_dir = images_dir / class_label / patient_id
            if images_patient_dir.exists():
                for img_file in images_patient_dir.glob('*.npy'):
                    rel_path = f"images/{class_label}/{patient_id}/{img_file.name}"
                    patients[patient_id]['files']['images'].append(rel_path)
            
            # Collect segmentation files
            seg_patient_dir = seg_dir / class_label / patient_id
            if seg_patient_dir.exists():
                for seg_file in seg_patient_dir.glob('*.npy'):
                    rel_path = f"segmentation/{class_label}/{patient_id}/{seg_file.name}"
                    patients[patient_id]['files']['segmentation'].append(rel_path)
    
    return patients


def compute_fingerprint_from_outputs(
    output_dir: Path,
    config: Dict,
) -> None:
    """
    Recompute fingerprints from saved VOI outputs (standalone mode).
    
    Reads dataset.json to index files, then loads images/*.npy and mask/*.npy
    to compute intensity statistics without re-running preprocessing.
    
    Requires:
        - SAVE_IMAGES=true during preprocessing (for images fingerprint)
        - mask/ folder (for segmentation fingerprint)
    
    Args:
        output_dir: Path to VOI output directory containing dataset.json
        config: Configuration dictionary (for processing_config metadata)
    """
    from tqdm import tqdm
    
    output_dir = Path(output_dir)
    dataset_path = output_dir / 'dataset.json'
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset.json not found at {dataset_path}")
    
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    print(f"\n{'='*70}")
    print("FINGERPRINT COMPUTATION (from saved outputs)")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    
    # Collect all VOI files
    patients = dataset.get('patients', {})
    
    # Initialize fingerprint accumulators
    seg_fingerprint_data = {
        'intensities': [],
        'shapes': [],
        'spacings': [],
        'total_foreground_voxels': 0,
        'filtered_voxels': 0,
    }
    img_fingerprint_data = {
        'intensities': [],
        'shapes': [],
        'spacings': [],
        'total_foreground_voxels': 0,
        'filtered_voxels': 0,
    }
    
    # Reconstruct processed_data and patient_files for fingerprint generation
    processed_data = {}
    patient_files = {}
    
    # Get spacing from config or dataset
    target_spacing = config.get('TARGET_SPACING') or dataset.get('processing_config', {}).get('target_spacing', [1.0, 1.0, 1.0])
    
    # FALLBACK: If dataset.json has no patients, scan filesystem directly
    if not patients:
        print("⚠ dataset.json has no patient records. Scanning filesystem...")
        patients = _scan_output_filesystem(output_dir)
        if not patients:
            print("❌ No VOI files found on disk either.")
            print("   Run preprocessing first: python src/planner.py --config <config> --preprocess")
            return
        print(f"Found {len(patients)} patients from filesystem scan\n")
    
    # Count VOIs
    total_vois = sum(
        len(pdata.get('files', {}).get('mask', []))
        for pdata in patients.values()
    )
    print(f"Found {total_vois} VOIs to process\n")
    
    has_images = False
    
    for patient_id, pdata in tqdm(patients.items(), desc="Processing patients"):
        patient_files[patient_id] = []
        
        mask_files = pdata.get('files', {}).get('mask', [])
        image_files = pdata.get('files', {}).get('images', [])
        
        # Build lookup for image files by base name
        image_lookup = {}
        for img_rel in image_files:
            img_path = output_dir / img_rel
            base = img_path.stem  # e.g., "case_00001_L"
            image_lookup[base] = img_path
        
        for mask_rel in mask_files:
            mask_path = output_dir / mask_rel
            if not mask_path.exists():
                continue
            
            base = mask_path.stem
            case_key = f"{patient_id}/{base}"
            
            # Load mask
            mask_array = np.load(mask_path)
            
            # Collect shape/spacing
            seg_fingerprint_data['shapes'].append(list(mask_array.shape))
            seg_fingerprint_data['spacings'].append(list(target_spacing))
            
            # Check for corresponding image file
            img_path = image_lookup.get(base)
            if img_path and img_path.exists():
                has_images = True
                img_array = np.load(img_path)
                
                img_fingerprint_data['shapes'].append(list(img_array.shape))
                img_fingerprint_data['spacings'].append(list(target_spacing))
                
                # Segmentation fingerprint: voxels where mask > 0
                fg_mask = mask_array > 0
                fg_intensities = img_array[fg_mask].flatten()
                seg_voxels = collect_voxels_with_hu_filter(
                    fg_intensities,
                    seg_fingerprint_data['intensities']
                )
                seg_fingerprint_data['total_foreground_voxels'] += len(fg_intensities)
                seg_fingerprint_data['filtered_voxels'] += seg_voxels
                
                # Images fingerprint: all voxels
                all_intensities = img_array.flatten()
                img_voxels = collect_voxels_with_hu_filter(
                    all_intensities,
                    img_fingerprint_data['intensities']
                )
                img_fingerprint_data['total_foreground_voxels'] += len(all_intensities)
                img_fingerprint_data['filtered_voxels'] += img_voxels
            
            # Minimal processed_data entry for class distribution
            processed_data[case_key] = {
                'status': 'success',
                'subfolder': pdata.get('class', 'unknown'),
                'patient_id': patient_id,
            }
            patient_files[patient_id].append(mask_rel)
    
    if not has_images:
        print("\n⚠ Warning: No images/*.npy files found.")
        print("  Fingerprint requires SAVE_IMAGES=true during preprocessing.")
        print("  Only shape/spacing statistics will be available.\n")
    
    # Generate fingerprints using existing function
    print(f"\n{'='*70}")
    generate_fingerprint(
        seg_fingerprint_data,
        processed_data,
        patient_files,
        config,
        output_dir,
        img_fingerprint_data if has_images else None,
    )
    
    print(f"{'='*70}")
    print("Fingerprint computation complete!")
    print(f"{'='*70}\n")
