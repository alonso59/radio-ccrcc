"""
Edge case validators for processed VOI data.
Non-destructive - flags issues without modifying data.
"""
from typing import Dict, List, Tuple
import numpy as np
import SimpleITK as sitk


def validate_boundary_contact(
    voi_size: Tuple[int, ...],
    original_size: Tuple[int, ...],
    bbox: Tuple[int, ...],
    margin: int = 5
) -> Dict:
    """Check if VOI touches image boundaries."""
    flags = []
    axes_affected = []
    
    for i, axis in enumerate(['X', 'Y', 'Z']):
        # Check if bbox start is near image start
        if bbox[i] < margin:
            flags.append(f"Touches {axis}-min boundary")
            axes_affected.append(f"{axis}-min")
        # Check if bbox end is near image end
        if bbox[i] + bbox[i+3] > original_size[i] - margin:
            flags.append(f"Touches {axis}-max boundary")
            axes_affected.append(f"{axis}-max")
    
    return {
        'passed': len(flags) == 0,
        'severity': 'high' if len(axes_affected) >= 2 else 'medium' if axes_affected else 'low',
        'flags': flags,
        'metrics': {'axes_affected': axes_affected}
    }


def validate_tumor_centering(
    mask_array: np.ndarray,
    tumor_label: int = 2,
    max_offset_ratio: float = 0.25
) -> Dict:
    """Check if tumor is centered in VOI."""
    tumor_mask = (mask_array == tumor_label)
    
    if tumor_mask.sum() == 0:
        return {
            'passed': False,
            'severity': 'high',
            'flags': ['No tumor found in VOI'],
            'metrics': {}
        }
    
    # Calculate tumor centroid
    coords = np.argwhere(tumor_mask)
    centroid = coords.mean(axis=0)
    voi_center = np.array(mask_array.shape) / 2
    
    # Offset from center
    offset = np.abs(centroid - voi_center)
    offset_ratio = offset / np.array(mask_array.shape)
    max_ratio = offset_ratio.max()
    
    flags = []
    if max_ratio > max_offset_ratio:
        flags.append(f"Tumor off-center: {max_ratio*100:.1f}% offset")
    
    return {
        'passed': len(flags) == 0,
        'severity': 'medium' if flags else 'low',
        'flags': flags,
        'metrics': {
            'offset_voxels': offset.tolist(),
            'offset_ratio': offset_ratio.tolist(),
            'max_offset_ratio': float(max_ratio)
        }
    }


def validate_intensity(
    image_array: np.ndarray,
    mask_array: np.ndarray,
    hu_range: Tuple[float, float] = (-200, 300)
) -> Dict:
    """Detect intensity artifacts (metal, air, out-of-range)."""
    foreground = image_array[mask_array > 0]
    
    if len(foreground) == 0:
        return {
            'passed': False,
            'severity': 'high',
            'flags': ['Empty mask'],
            'metrics': {}
        }
    
    flags = []
    
    # Metal artifacts (very high HU)
    metal_ratio = (foreground > 1000).sum() / len(foreground)
    if metal_ratio > 0.05:
        flags.append(f"Metal artifacts: {metal_ratio*100:.1f}%")
    
    # Air/noise (very low HU)
    air_ratio = (foreground < -500).sum() / len(foreground)
    if air_ratio > 0.1:
        flags.append(f"Air/noise: {air_ratio*100:.1f}%")
    
    # Out-of-range ratio
    oor = ((foreground < hu_range[0]) | (foreground > hu_range[1])).sum()
    oor_ratio = oor / len(foreground)
    if oor_ratio > 0.5:
        flags.append(f"Out-of-range HU: {oor_ratio*100:.1f}%")
    
    return {
        'passed': len(flags) == 0,
        'severity': 'high' if len(flags) >= 2 else 'medium' if flags else 'low',
        'flags': flags,
        'metrics': {
            'metal_ratio': float(metal_ratio),
            'air_ratio': float(air_ratio),
            'out_of_range_ratio': float(oor_ratio),
            'hu_mean': float(foreground.mean()),
            'hu_std': float(foreground.std())
        }
    }


def validate_voi_size(
    actual_size: Tuple[int, ...],
    min_size: Tuple[int, ...] = (96, 96, 96)
) -> Dict:
    """Check if VOI meets minimum size requirements."""
    flags = []
    undersized = []
    
    for i, axis in enumerate(['X', 'Y', 'Z']):
        if actual_size[i] < min_size[i]:
            flags.append(f"{axis}: {actual_size[i]} < {min_size[i]}")
            undersized.append(axis)
    
    return {
        'passed': len(flags) == 0,
        'severity': 'high' if len(undersized) >= 2 else 'medium' if undersized else 'low',
        'flags': flags,
        'metrics': {
            'actual_size': list(actual_size),
            'min_size': list(min_size),
            'undersized_axes': undersized
        }
    }


def validate_multi_lesion(
    mask_array: np.ndarray,
    tumor_label: int = 2
) -> Dict:
    """Detect multiple disconnected tumors."""
    tumor_mask = (mask_array == tumor_label).astype(np.uint8)
    
    if tumor_mask.sum() == 0:
        return {'passed': True, 'severity': 'low', 'flags': [], 'metrics': {'n_lesions': 0}}
    
    # Connected component analysis
    tumor_sitk = sitk.GetImageFromArray(tumor_mask)
    labeled = sitk.ConnectedComponent(tumor_sitk)
    
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(labeled)
    
    n_lesions = len(stats.GetLabels())
    sizes = sorted([stats.GetNumberOfPixels(lbl) for lbl in stats.GetLabels()], reverse=True)
    
    flags = []
    if n_lesions > 1:
        flags.append(f"Multiple lesions: {n_lesions} (sizes: {sizes})")
    
    return {
        'passed': n_lesions <= 1,
        'severity': 'medium' if n_lesions > 1 else 'low',
        'flags': flags,
        'metrics': {'n_lesions': n_lesions, 'lesion_sizes': sizes}
    }


def run_all_validators(
    image_array: np.ndarray,
    mask_array: np.ndarray,
    bbox: Tuple[int, ...],
    original_size: Tuple[int, ...],
    config: Dict
) -> Dict:
    """
    Run all validators on a VOI.
    
    Returns:
        {
            'overall_status': 'clean'|'flagged'|'critical',
            'all_flags': [...],
            'validators': {...}
        }
    """
    voi_size = image_array.shape[::-1]  # ZYX -> XYZ
    min_voi_size = tuple(config.get('MIN_VOI_SIZE', [96, 96, 96]))
    hu_range = tuple(config.get('HU_RANGE', [-200, 300]))
    boundary_margin = config.get('BOUNDARY_MARGIN', 5)
    
    results = {
        'boundary_contact': validate_boundary_contact(voi_size, original_size, bbox, boundary_margin),
        'tumor_centering': validate_tumor_centering(mask_array),
        'intensity': validate_intensity(image_array, mask_array, hu_range),
        'voi_size': validate_voi_size(voi_size, min_voi_size),
        'multi_lesion': validate_multi_lesion(mask_array)
    }
    
    # Aggregate results
    all_flags = []
    max_severity = 'low'
    
    for name, result in results.items():
        if not result['passed']:
            all_flags.extend(result['flags'])
            if result['severity'] == 'high':
                max_severity = 'high'
            elif result['severity'] == 'medium' and max_severity == 'low':
                max_severity = 'medium'
    
    if len(all_flags) == 0:
        status = 'clean'
    elif max_severity == 'high':
        status = 'critical'
    else:
        status = 'flagged'
    
    return {
        'overall_status': status,
        'severity': max_severity,
        'total_flags': len(all_flags),
        'all_flags': all_flags,
        'validators': results
    }
