"""I/O module: file discovery, loading, and saving."""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import SimpleITK as sitk

from .common import extract_patient_id, SUPPORTED_INPUT_FORMATS


# =============================================================================
# FILE DISCOVERY
# =============================================================================
def detect_input_format(folder: Path) -> str:
    """Auto-detect input format ('nifti' or 'npy') based on files in folder."""
    for ext in ['.nii.gz', '.nii', '.npy']:
        if list(folder.rglob(f'*{ext}')):
            return 'nifti' if ext in ['.nii.gz', '.nii'] else 'npy'
    raise ValueError(f"No supported files found in {folder}")


def discover_files(
    image_folder: Path,
    mask_folder: Path,
    image_suffix: str,
    mask_suffix: str,
    patient_id_pattern: Optional[str] = None
) -> Dict[str, Dict]:
    """
    Discover and match image-mask pairs.
    
    Args:
        image_folder: Path to image directory
        mask_folder: Path to mask directory
        image_suffix: Suffix for image files
        mask_suffix: Suffix for mask files
        patient_id_pattern: Regex pattern for patient ID extraction
        
    Returns:
        Dictionary mapping case_key to case data
    """
    # Find all files
    image_paths = list(image_folder.rglob(f'*{image_suffix}'))
    mask_paths = list(mask_folder.rglob(f'*{mask_suffix}'))
    
    if not image_paths:
        raise ValueError(f"No image files found with suffix {image_suffix}")
    if not mask_paths:
        raise ValueError(f"No mask files found with suffix {mask_suffix}")
    
    # Build case mapping for images
    image_cases = {}
    for img_path in image_paths:
        rel_path = img_path.parent.relative_to(image_folder)
        case_name = img_path.name.replace(image_suffix, '')
        subfolder = str(rel_path) if str(rel_path) != '.' else ''
        case_key = f"{subfolder}/{case_name}" if subfolder else case_name
        image_cases[case_key] = {
            'case_name': case_name,
            'subfolder': subfolder,
            'image_path': str(img_path),
        }
    
    # Build case mapping for masks
    mask_cases = {}
    for msk_path in mask_paths:
        rel_path = msk_path.parent.relative_to(mask_folder)
        case_name = msk_path.name.replace(mask_suffix, '')
        subfolder = str(rel_path) if str(rel_path) != '.' else ''
        case_key = f"{subfolder}/{case_name}" if subfolder else case_name
        mask_cases[case_key] = str(msk_path)
    
    # Find matching cases
    matching_keys = set(image_cases.keys()) & set(mask_cases.keys())
    
    if not matching_keys:
        raise ValueError("No matching image-mask pairs found")
    
    # Build file list with patient ID extraction
    file_list = {}
    for key in matching_keys:
        case_data = image_cases[key]
        case_data['mask_path'] = mask_cases[key]
        case_data['patient_id'] = extract_patient_id(case_data['case_name'], patient_id_pattern)
        file_list[key] = case_data
    
    return file_list


# =============================================================================
# IMAGE LOADING
# =============================================================================
def load_npy_as_sitk(path: str, spacing) -> sitk.Image:
    arr = np.load(path)
    img = sitk.GetImageFromArray(arr)
    if spacing:
        img.SetSpacing(spacing)
    return img


def load_case(
    case_data: Dict,
    input_format: str,
    default_spacing
) -> Tuple[sitk.Image, sitk.Image]:
    if input_format == 'npy':
        image = load_npy_as_sitk(case_data['image_path'], default_spacing)
        mask = load_npy_as_sitk(case_data['mask_path'], default_spacing)
    else:
        image = sitk.ReadImage(case_data['image_path'])
        mask = sitk.ReadImage(case_data['mask_path'])
    
    return image, mask


# =============================================================================
# VOI SAVING
# =============================================================================
def get_output_path(
    output_dir: Path,
    output_type: str,
    case_data: Dict,
    side: str
) -> Path:
    subfolder = case_data['subfolder']  # This is the class (A, A+B, etc.)
    patient_id = case_data['patient_id']
    case_name = case_data['case_name']
    
    # Use 'NG' (No Grade) if no class/subfolder is specified
    class_folder = subfolder if subfolder else 'NG'
    
    # Build path: {output_dir}/{type}/{class}/{patient_id}/
    out_dir = output_dir / output_type / class_folder / patient_id
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Filename: {case_name}_{side}.npy
    filename = f"{case_name}_{side}.npy"
    return out_dir / filename


def save_voi(
    results: Dict,
    case_data: Dict,
    output_dir: Path,
    save_mask: bool = True,
    save_segmentation: bool = True,
    save_images: bool = False
) -> List[str]:
    saved_files = []
    
    for side in ['L', 'R']:
        if f'{side}_image' not in results:
            continue
        
        # Save mask (segmentation labels)
        if save_mask:
            msk_path = get_output_path(output_dir, 'mask', case_data, side)
            msk_array = sitk.GetArrayFromImage(results[f'{side}_mask']).astype(np.uint8)
            np.save(msk_path, msk_array)
            # Store as relative path to output_dir
            saved_files.append(str(msk_path.relative_to(output_dir)))
        
        # Save segmentation (masked CT image / VOI)
        if save_segmentation:
            seg_path = get_output_path(output_dir, 'segmentation', case_data, side)
            seg_array = sitk.GetArrayFromImage(results[f'{side}_segmented']).astype(np.float32)
            np.save(seg_path, seg_array)
            # Store as relative path to output_dir
            saved_files.append(str(seg_path.relative_to(output_dir)))
        
        # Save full images (full CT VOI without masking)
        if save_images:
            img_path = get_output_path(output_dir, 'images', case_data, side)
            img_array = sitk.GetArrayFromImage(results[f'{side}_image']).astype(np.float32)
            np.save(img_path, img_array)
            # Store as relative path to output_dir
            saved_files.append(str(img_path.relative_to(output_dir)))
    
    return saved_files
