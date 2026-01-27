"""Utility functions and constants."""

import re
from typing import Dict, List, Optional

# =============================================================================
# CONSTANTS
# =============================================================================
TUMOR_LABEL = 2
BACKGROUND_HU = -200  # After HU clipping [-200, 300], background becomes -200
SUPPORTED_INPUT_FORMATS = ['.nii.gz', '.nii', '.npy']


# =============================================================================
# PATIENT ID EXTRACTION
# =============================================================================
def extract_patient_id(case_name: str, pattern: Optional[str] = None) -> str:
    """
    Extract patient ID from case name using regex pattern.
    
    Args:
        case_name: Full case name (e.g., "00_case_00002" or "TCGA-XX-YYYY")
        pattern: Regex pattern to extract patient ID
        
    Returns:
        Extracted patient ID or fallback
        
    Examples:
        >>> extract_patient_id("00_case_00002")  # UKB25 format
        'case_00002'
        >>> extract_patient_id("TCGA-XX-YYYY")  # TCGA format
        'TCGA-XX-YYYY'
    """
    if pattern:
        match = re.search(pattern, case_name)
        if match:
            # Return first non-None group (supports multiple alternations)
            return next((g for g in match.groups() if g is not None), match.group(0))
    
    # Fallback: Extract case_XXXXX pattern (common format for UKB/KiTS datasets)
    # Removes scan prefix (e.g., "00_case_00002" -> "case_00002")
    case_match = re.search(r'(case_\d{5})', case_name)
    if case_match:
        return case_match.group(1)
    
    # Final fallback: use full name (safer than truncation to avoid data loss)
    return case_name


# =============================================================================
# FILE CATEGORIZATION
# =============================================================================
def categorize_files_by_type(files: List[str]) -> Dict[str, List[str]]:
    """
    Categorize files by output type (mask/segmentation).
    
    Args:
        files: List of file paths (can be relative or absolute)
        
    Returns:
        Dictionary with 'mask' and 'segmentation' file lists
    """
    result = {'mask': [], 'segmentation': []}
    for f in files:
        # Handle both relative and absolute paths
        if '/mask/' in f or f.startswith('mask/'):
            result['mask'].append(f)
        elif '/segmentation/' in f or f.startswith('segmentation/'):
            result['segmentation'].append(f)
    return result


def get_files_by_type_for_patients(
    patient_list: List[str], 
    patient_files: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """
    Get categorized files for a list of patients.
    
    Args:
        patient_list: List of patient IDs
        patient_files: Mapping of patient ID to file list
        
    Returns:
        Dictionary with 'mask' and 'segmentation' file lists
    """
    all_files = []
    for p in patient_list:
        all_files.extend(patient_files.get(p, []))
    return categorize_files_by_type(all_files)
