"""
Anatomical scan plane detection
"""

import numpy as np
import SimpleITK as sitk

from ..utils.dicom_utils import get_metadata_value


class ScanPlaneDetector:
    """Detects anatomical scan plane (AXIAL, CORONAL, SAGITTAL, OBLIQUE)."""
    
    @staticmethod
    def determine_scan_plane(reader: sitk.ImageSeriesReader, image: sitk.Image) -> str:
        """Determine scan plane from Image Orientation Patient DICOM tag.
        
        Args:
            reader: DICOM reader with loaded series
            image: SimpleITK image
            
        Returns:
            'AXIAL', 'CORONAL', 'SAGITTAL', or 'OBLIQUE'
        """
        try:
            # Get Image Orientation Patient (0020|0037) - 6 values
            iop_str = get_metadata_value(reader, '0020|0037', '')
            
            if not iop_str:
                # Fallback: Check series description
                return ScanPlaneDetector._determine_scan_plane_fallback(reader)
            
            # Parse orientation values
            iop_values = [float(x) for x in iop_str.split('\\')]
            
            if len(iop_values) != 6:
                return ScanPlaneDetector._determine_scan_plane_fallback(reader)
            
            # First 3 values: direction of rows, Last 3: direction of columns
            row_vec = iop_values[0:3]
            col_vec = iop_values[3:6]
            
            # Calculate normal vector (cross product)
            normal = np.cross(row_vec, col_vec)
            
            # Find dominant axis
            abs_normal = [abs(x) for x in normal]
            max_idx = abs_normal.index(max(abs_normal))
            
            # Threshold for oblique detection
            if max(abs_normal) < 0.8:
                return 'OBLIQUE'
            
            # Determine plane based on dominant normal axis
            if max_idx == 2:  # Z-axis dominant
                return 'AXIAL'
            elif max_idx == 1:  # Y-axis dominant
                return 'CORONAL'
            elif max_idx == 0:  # X-axis dominant
                return 'SAGITTAL'
            else:
                return 'OBLIQUE'
                
        except Exception as e:
            # If anything fails, use fallback
            return ScanPlaneDetector._determine_scan_plane_fallback(reader)
    
    @staticmethod
    def _determine_scan_plane_fallback(reader: sitk.ImageSeriesReader) -> str:
        """Fallback method using series description keywords.
        
        Args:
            reader: DICOM reader with loaded series
            
        Returns:
            'AXIAL', 'CORONAL', 'SAGITTAL' (defaults to AXIAL if unclear)
        """
        series_desc = get_metadata_value(reader, '0008|103e', '').upper()
        
        if any(kw in series_desc for kw in ['AXIAL', ' AX ', 'AX.', 'TRANSVERSE', 'TRA']):
            return 'AXIAL'
        elif any(kw in series_desc for kw in ['CORONAL', ' COR ', 'COR.']):
            return 'CORONAL'
        elif any(kw in series_desc for kw in ['SAGITTAL', ' SAG ', 'SAG.']):
            return 'SAGITTAL'
        else:
            # Default to AXIAL if unclear (most CT scans are axial)
            return 'AXIAL'
