"""
Universal DICOM to NIfTI Converter Package
Automatic dataset detection with standardized naming and metadata preservation
"""

from .core.converter import UniversalDICOMConverter

__version__ = "3.0.0"
__all__ = ["UniversalDICOMConverter"]
