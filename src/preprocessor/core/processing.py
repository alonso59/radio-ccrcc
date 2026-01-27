"""Image processing: orientation, resampling, kidney separation, VOI extraction."""

from typing import Dict, List, Tuple, Optional
import numpy as np
import SimpleITK as sitk

from ..utils.common import TUMOR_LABEL, BACKGROUND_HU


# =============================================================================
# HU CLIPPING
# =============================================================================
def clip_hu_range(
    image: sitk.Image,
    hu_min: float = -200,
    hu_max: float = 300
) -> sitk.Image:
    """Clip image HU values to specified range."""
    clamp_filter = sitk.ClampImageFilter()
    clamp_filter.SetLowerBound(hu_min)
    clamp_filter.SetUpperBound(hu_max)
    return clamp_filter.Execute(image)


# =============================================================================
# ORIENTATION & RESAMPLING
# =============================================================================
def convert_to_ras(
    image: sitk.Image, 
    mask: sitk.Image,
    enabled: bool = True
) -> Tuple[sitk.Image, sitk.Image]:
    """
    Convert images to RAS+ canonical orientation.
    
    Args:
        image: CT image
        mask: Segmentation mask
        enabled: Whether to perform conversion
        
    Returns:
        Tuple of (image, mask) in RAS orientation
    """
    if not enabled:
        return image, mask
    return sitk.DICOMOrient(image, 'RAS'), sitk.DICOMOrient(mask, 'RAS')


def resample_to_spacing(
    image: sitk.Image,
    mask: sitk.Image,
    target_spacing,
) -> Tuple[sitk.Image, sitk.Image]:
    """
    Resample image and mask to target spacing.
    
    Args:
        image: CT image
        mask: Segmentation mask
        target_spacing: Target voxel spacing [x, y, z]
        
    Returns:
        Tuple of resampled (image, mask)
    """
    target_spacing = np.array(target_spacing)
    original_spacing = np.array(image.GetSpacing())
    
    new_size = np.ceil(np.array(image.GetSize()) * (original_spacing / target_spacing)).astype(np.uint32)
    
    resample_params = {
        'size': new_size.tolist(),
        'transform': sitk.Transform(),
        'outputOrigin': image.GetOrigin(),
        'outputSpacing': target_spacing.tolist(),
        'outputDirection': image.GetDirection(),
    }
    
    resampled_image = sitk.Resample(image, interpolator=sitk.sitkBSpline, defaultPixelValue=0.0, **resample_params)
    resampled_mask = sitk.Resample(mask, interpolator=sitk.sitkNearestNeighbor, defaultPixelValue=0, **resample_params)
    
    return resampled_image, resampled_mask


# =============================================================================
# KIDNEY SEPARATION
# =============================================================================
def separate_kidneys(mask: sitk.Image ,bbox_labels,) -> Dict[str, Dict]:
    """
    Separate left and right kidneys using connected components.
    
    Args:
        mask: Segmentation mask
        bbox_labels: Labels to use for bounding box computation
        
    Returns:
        Dictionary mapping side ('L'/'R') to bbox info
        
    Raises:
        ValueError: If no kidney components found
    """
    if bbox_labels is None:
        bbox_labels = [1, 2]
    
    mask_array = sitk.GetArrayFromImage(mask)
    
    if mask_array.max() == 0:
        raise ValueError("Mask contains no labeled regions")
    
    unified_mask = np.isin(mask_array, bbox_labels).astype(np.uint8)
    
    if unified_mask.sum() == 0:
        raise ValueError(f"Mask does not contain BBOX_LABELS: {bbox_labels}")
    
    unified_sitk = sitk.GetImageFromArray(unified_mask)
    unified_sitk.CopyInformation(mask)
    labeled_components = sitk.ConnectedComponent(unified_sitk)
    
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(labeled_components)
    
    components = sorted(
        [(lbl, stats.GetNumberOfPixels(lbl), stats.GetCentroid(lbl)) for lbl in stats.GetLabels()],
        key=lambda x: x[1], reverse=True
    )[:2]
    
    if not components:
        raise ValueError("No kidney components found")
    
    return _process_kidney_components(components, mask, labeled_components, unified_mask)


def _process_kidney_components(
    components: List[Tuple],
    mask: sitk.Image,
    labeled_components: sitk.Image,
    unified_mask: np.ndarray
) -> Dict[str, Dict]:
    """
    Process kidney components and determine left/right sides.
    
    Args:
        components: List of (label, pixel_count, centroid) tuples
        mask: Original mask
        labeled_components: Connected component labeled image
        unified_mask: Binary unified mask array
        
    Returns:
        Dictionary mapping side to bbox info
    """
    labeled_array = sitk.GetArrayFromImage(labeled_components)
    bboxes = {}
    
    if len(components) == 1:
        label, _, centroid = components[0]
        image_center_x = (mask.GetSize()[0] * mask.GetSpacing()[0]) / 2
        side = 'L' if centroid[0] < image_center_x else 'R'
        bbox_info = _calculate_bbox(label, labeled_array, unified_mask, mask)
        if bbox_info:
            bboxes[side] = bbox_info
    else:
        left_comp, right_comp = sorted(components, key=lambda x: x[2][0])
        for side, (label, _, _) in [('L', left_comp), ('R', right_comp)]:
            bbox_info = _calculate_bbox(label, labeled_array, unified_mask, mask)
            if bbox_info:
                bboxes[side] = bbox_info
    
    return bboxes


def _calculate_bbox(
    label: int,
    labeled_array: np.ndarray,
    unified_mask: np.ndarray,
    mask: sitk.Image
) -> Optional[Dict]:
    """
    Calculate bounding box for a kidney component.
    
    Args:
        label: Component label
        labeled_array: Array of component labels
        unified_mask: Binary unified mask
        mask: Original mask image
        
    Returns:
        Dictionary with 'bbox' and 'center_slice', or None
    """
    kidney_mask = (labeled_array == label) & (unified_mask == 1)
    kidney_sitk = sitk.GetImageFromArray(kidney_mask.astype(np.uint8))
    kidney_sitk.CopyInformation(mask)
    
    bbox_stats = sitk.LabelShapeStatisticsImageFilter()
    bbox_stats.Execute(kidney_sitk)
    
    if bbox_stats.GetLabels():
        bbox = bbox_stats.GetBoundingBox(1)
        return {'bbox': bbox, 'center_slice': bbox[2] + bbox[5] // 2}
    return None


# =============================================================================
# VOI EXTRACTION
# =============================================================================
def extract_voi(
    image: sitk.Image,
    mask: sitk.Image,
    bbox_info: Dict,
    expansion_mm: float,
    target_spacing,
    min_voi_size
) -> Tuple[sitk.Image, sitk.Image, int]:
    """
    Extract Volume of Interest using bounding box with expansion.
    
    Args:
        image: Full CT image
        mask: Full segmentation mask
        bbox_info: Dictionary with 'bbox' and 'center_slice'
        expansion_mm: Expansion margin in mm
        target_spacing: Target voxel spacing
        min_voi_size: Minimum VOI size in voxels
        
    Returns:
        Tuple of (voi_image, voi_mask, center_slice)
    """
    if min_voi_size is None:
        min_voi_size = [64, 64, 64]
    
    bbox = bbox_info['bbox']
    spacing = np.array(image.GetSpacing())
    target_spacing = np.array(target_spacing)
    min_voi_size = np.array(min_voi_size)
    
    min_size_voxels = np.ceil(min_voi_size * (target_spacing / spacing)).astype(int)
    expansion_voxels = (expansion_mm / spacing).astype(int)
    
    start, size = _calculate_voi_bounds(bbox, expansion_voxels, image.GetSize(), min_size_voxels)
    
    voi_image = sitk.RegionOfInterest(image, size=size, index=start)
    voi_mask = sitk.RegionOfInterest(mask, size=size, index=start)
    
    return voi_image, voi_mask, bbox_info['center_slice']


def _calculate_voi_bounds(
    bbox: Tuple[int, ...],
    expansion: np.ndarray,
    image_size: Tuple[int, ...],
    min_size: np.ndarray
) -> Tuple[List[int], List[int]]:
    """
    Calculate VOI bounds with expansion and minimum size constraints.
    
    Args:
        bbox: Original bounding box (x, y, z, width, height, depth)
        expansion: Expansion in voxels for each dimension
        image_size: Size of the full image
        min_size: Minimum VOI size in voxels
        
    Returns:
        Tuple of (start_indices, sizes) as lists
    """
    start = [max(0, bbox[i] - expansion[i]) for i in range(3)]
    end = [min(image_size[i], bbox[i] + bbox[i+3] + expansion[i]) for i in range(3)]
    size = [end[i] - start[i] for i in range(3)]
    
    for i in range(3):
        if size[i] < min_size[i]:
            deficit = int(min_size[i] - size[i])
            expand_start = deficit // 2
            expand_end = deficit - expand_start
            
            new_start = max(0, start[i] - expand_start)
            new_end = min(image_size[i], end[i] + expand_end)
            
            remaining_deficit = int(min_size[i]) - (new_end - new_start)
            if remaining_deficit > 0:
                if new_start == 0:
                    new_end = min(image_size[i], new_end + remaining_deficit)
                elif new_end == image_size[i]:
                    new_start = max(0, new_start - remaining_deficit)
            
            start[i], end[i] = new_start, new_end
            size[i] = end[i] - start[i]
    
    return [int(s) for s in start], [int(s) for s in size]


# =============================================================================
# MASK APPLICATION
# =============================================================================
def apply_mask_to_voi(
    image: sitk.Image,
    mask: sitk.Image,
    mask_labels
) -> sitk.Image:
    """
    Apply mask to VOI, setting background to HU=-1000.
    
    Args:
        image: VOI CT image
        mask: VOI segmentation mask
        mask_labels: Labels to keep (None = all non-zero)
        
    Returns:
        Masked image with background set to -1000 HU
    """
    img_array = sitk.GetArrayFromImage(image)
    mask_array = sitk.GetArrayFromImage(mask)
    segmented_array = np.full_like(img_array, BACKGROUND_HU)
    
    if mask_labels is None:
        mask_keep = mask_array > 0
    else:
        mask_keep = np.isin(mask_array, mask_labels)
    
    segmented_array[mask_keep] = img_array[mask_keep]
    segmented_img = sitk.GetImageFromArray(segmented_array)
    segmented_img.CopyInformation(image)
    return segmented_img


# =============================================================================
# VOI VALIDATION
# =============================================================================
def validate_voi(
    image: sitk.Image,
    mask: sitk.Image,
    min_kidney_voxels: int = 0,
    min_tumor_voxels: int = 0,
    min_hu_in_range_ratio: float = 0.0,
) -> Tuple[bool, str]:
    """
    Validate VOI mask quality.
    
    Args:
        image: VOI CT image
        mask: VOI segmentation mask
        min_kidney_voxels: Minimum kidney voxel count
        min_tumor_voxels: Minimum tumor voxel count
        min_hu_in_range_ratio: Minimum ratio of voxels in valid range
        
    Returns:
        Tuple of (is_valid, reason_string)
    """
    mask_array = sitk.GetArrayFromImage(mask)
    
    kidney_voxels = int((mask_array > 0).sum())
    if kidney_voxels == 0:
        return False, "Empty VOI mask"
    
    if min_kidney_voxels > 0 and kidney_voxels < min_kidney_voxels:
        return False, f"Kidney too small ({kidney_voxels} voxels)"
    
    if min_tumor_voxels > 0:
        tumor_voxels = int((mask_array == TUMOR_LABEL).sum())
        if 0 < tumor_voxels < min_tumor_voxels:
            return False, f"Tumor too small ({tumor_voxels} voxels)"
    
    return True, "OK"
