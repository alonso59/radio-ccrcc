"""
CT protocol detection from multiple sources
"""

import SimpleITK as sitk
from typing import Tuple

from ..utils.dicom_utils import get_metadata_value


class ProtocolDetector:
    """Detects CT protocol phase (non-contrast, arterial, venous, undefined)."""
    
    # Enhanced keyword lists
    NC_KEYWORDS = [
        'non-contrast', 'non contrast', 'nc', 'plain', 'without',
        'pre-contrast', 'precontrast', 'native', 'baseline',
        'unenhanced', 'phase 1', 'phase1', 'pre contrast'
    ]
    
    ART_KEYWORDS = [
        'arterial', 'art', 'artery', 'early', 'phase 2', 'phase2',
        'angio', 'cta', 'early arterial', 'late arterial', 'arterial phase'
    ]
    
    VEN_KEYWORDS = [
        'venous', 'ven', 'portal', 'vein', 'phase 3', 'phase3',
        'porto-venous', 'portvenous', 'pv', 'delayed', 'late',
        'venous phase', 'portal venous'
    ]
    
    @staticmethod
    def determine_protocol(reader: sitk.ImageSeriesReader, series_path: str) -> Tuple[str, str]:
        """Determine CT protocol from multiple sources.
        
        Priority:
        1. DICOM metadata (series description, protocol name, sequence name)
        2. Directory/filename analysis
        3. Undefined if no match
        
        Args:
            reader: DICOM reader with loaded series
            series_path: Path to DICOM series directory
            
        Returns:
            Tuple of (protocol, source) where:
                protocol: 'nc', 'art', 'ven', 'undefined'
                source: 'metadata', 'filename', 'undefined'
        """
        # Priority 1: DICOM Metadata
        series_desc = get_metadata_value(reader, "0008|103e", "").lower()
        protocol_name = get_metadata_value(reader, "0018|1030", "").lower()
        sequence_name = get_metadata_value(reader, "0018|0024", "").lower()
        metadata_text = f"{series_desc} {protocol_name} {sequence_name}"
        
        if any(kw in metadata_text for kw in ProtocolDetector.NC_KEYWORDS):
            return 'nc', 'metadata'
        elif any(kw in metadata_text for kw in ProtocolDetector.ART_KEYWORDS):
            return 'art', 'metadata'
        elif any(kw in metadata_text for kw in ProtocolDetector.VEN_KEYWORDS):
            return 'ven', 'metadata'
        
        # Priority 2: Directory/Filename Analysis
        path_lower = series_path.lower()
        
        if any(kw in path_lower for kw in ProtocolDetector.NC_KEYWORDS):
            return 'nc', 'filename'
        elif any(kw in path_lower for kw in ProtocolDetector.ART_KEYWORDS):
            return 'art', 'filename'
        elif any(kw in path_lower for kw in ProtocolDetector.VEN_KEYWORDS):
            return 'ven', 'filename'
        
        # If no match found, return undefined
        return 'undefined', 'undefined'
