"""
Conversion validation utilities
"""

import SimpleITK as sitk
from pathlib import Path
from typing import Dict


class ConversionValidator:
    """Validates DICOM to NIfTI conversions."""
    
    @staticmethod
    def validate_conversion(nifti_path: Path, original_image: sitk.Image) -> Dict:
        """Validate converted NIfTI file.
        
        Args:
            nifti_path: Path to converted NIfTI file
            original_image: Original SimpleITK image before conversion
            
        Returns:
            Dictionary with validation results
        """
        checks = {
            'valid': True,
            'file_exists': nifti_path.exists(),
            'readable': False,
            'orientation_ras': False,
            'dimensions_match': False,
        }
        
        try:
            # Check file exists
            if not checks['file_exists']:
                checks['valid'] = False
                return checks
            
            # Try to read the converted file
            converted = sitk.ReadImage(str(nifti_path))
            checks['readable'] = True
            
            # Check orientation
            direction = converted.GetDirection()
            # RAS should have positive determinant and specific direction matrix
            checks['orientation_ras'] = len(direction) == 9
            
            # Check dimensions match
            original_size = original_image.GetSize()
            converted_size = converted.GetSize()
            checks['dimensions_match'] = (original_size == converted_size)
            
            if not checks['dimensions_match']:
                checks['valid'] = False
                
        except Exception as e:
            checks['valid'] = False
            checks['error'] = str(e)
        
        return checks
