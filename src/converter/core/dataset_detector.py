"""
Dataset type detection logic
"""

import re
import os
from pathlib import Path
from typing import Optional, Tuple
import SimpleITK as sitk

from ..utils.dicom_utils import get_metadata_value


class DatasetDetector:
    """Detects dataset type from DICOM headers and directory structure."""
    
    @staticmethod
    def detect_dataset_type(input_dir: Path) -> str:
        """Auto-detect dataset from directory structure and DICOM headers.
        
        Args:
            input_dir: Root directory containing DICOM files
            
        Returns:
            Dataset type ('tcga', 'ukbonn', 'kits', 'generic')
        """
        try:
            sample_reader = DatasetDetector._read_sample_series(input_dir)
            if sample_reader is None:
                return 'generic'
            
            try:
                patient_id = sample_reader.GetMetaData(0, '0010|0020') if sample_reader.HasMetaDataKey(0, '0010|0020') else ''
            except:
                patient_id = ''
            
            if re.match(r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}', patient_id):
                return 'tcga'
            elif patient_id.startswith('Anonym_') or re.match(r'^\d+$', patient_id):
                return 'ukbonn'
            elif 'KiTS' in patient_id or patient_id.startswith('case_'):
                return 'kits'
            else:
                return 'generic'
                
        except Exception as e:
            print(f"      Warning: Could not detect dataset type: {e}")
            return 'generic'
    
    @staticmethod
    def _read_sample_series(input_dir: Path) -> Optional[sitk.ImageSeriesReader]:
        """Read a sample DICOM series for dataset detection."""
        for root, dirs, files in os.walk(input_dir):
            if files and not dirs:
                try:
                    reader = sitk.ImageSeriesReader()
                    dicom_names = reader.GetGDCMSeriesFileNames(root)
                    if dicom_names:
                        reader.SetFileNames(dicom_names)
                        reader.MetaDataDictionaryArrayUpdateOn()
                        reader.LoadPrivateTagsOn()
                        reader.Execute()
                        return reader
                except:
                    continue
        return None
    
    @staticmethod
    def extract_patient_id(reader: sitk.ImageSeriesReader, file_path: Optional[str] = None) -> Tuple[str, str]:
        """Extract patient ID with fallback chain and validation.
        
        Priority order:
        1. DICOM metadata tags (0010|0020, 0010|0010, 0010|1000)
        2. Directory structure parsing
        3. Sanitized folder name
        
        Args:
            reader: DICOM reader with loaded series
            file_path: Optional path to DICOM file for directory parsing
            
        Returns:
            Tuple of (patient_id, source) where source indicates extraction method
        """
        patient_id, source = DatasetDetector._extract_from_dicom_tags(reader)
        
        if patient_id and DatasetDetector._validate_patient_id(patient_id):
            return DatasetDetector._sanitize_patient_id(patient_id), source
        
        if file_path:
            patient_id, source = DatasetDetector._extract_from_directory(file_path)
            if patient_id and DatasetDetector._validate_patient_id(patient_id):
                return DatasetDetector._sanitize_patient_id(patient_id), source
        
        return 'UNKNOWN', 'fallback'
    
    @staticmethod
    def _extract_from_dicom_tags(reader: sitk.ImageSeriesReader) -> Tuple[str, str]:
        """Extract patient ID from DICOM metadata tags."""
        tag_candidates = [
            ('0010|0020', 'PatientID'),
            ('0010|0010', 'PatientName'),
            ('0010|1000', 'OtherPatientIDs'),
        ]
        
        for tag, name in tag_candidates:
            try:
                if reader.HasMetaDataKey(0, tag):
                    value = reader.GetMetaData(0, tag).strip()
                    if value and value.lower() not in ['none', 'unknown', 'anonymous', '']:
                        return value, f'dicom_{name}'
            except:
                continue
        
        return '', 'none'
    
    @staticmethod
    def _extract_from_directory(file_path: str) -> Tuple[str, str]:
        """Extract patient ID from directory structure.
        
        Handles common patterns:
        - TCGA: /path/TCGA-XX-XXXX/study/series/files
        - UKBonn: /path/Anonym_12345/study/series/files
        - Generic: /path/PatientName/study/series/files
        """
        path_parts = Path(file_path).parts
        
        for part in reversed(path_parts):
            if re.match(r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}', part):
                return part, 'directory_tcga'
            
            if part.startswith('Anonym_'):
                return part, 'directory_ukbonn'
            
            if part.startswith('case_') and re.match(r'case_\d+', part):
                return part, 'directory_kits'
            
            if re.match(r'^[A-Za-z0-9][-_A-Za-z0-9]{2,}$', part):
                if not part.startswith(('1.', '2.', 'CT_', 'MR_', 'PT_')):
                    if len(path_parts) - list(reversed(path_parts)).index(part) >= 3:
                        return part, 'directory_patient_folder'
        
        if len(path_parts) >= 4:
            candidate = path_parts[-4]
            if re.match(r'^[A-Za-z0-9][-_A-Za-z0-9]{2,}$', candidate):
                return candidate, 'directory_parent'
        
        return '', 'none'
    
    @staticmethod
    def _validate_patient_id(patient_id: str) -> bool:
        """Validate that patient ID is reasonable."""
        if not patient_id or len(patient_id) < 2:
            return False
        
        if patient_id.lower() in ['unknown', 'anonymous', 'none', 'test', 'patient']:
            return False
        
        if re.match(r'^[0-9.]+$', patient_id) and '.' in patient_id:
            return False
        
        return True
    
    @staticmethod
    def _sanitize_patient_id(patient_id: str) -> str:
        """Sanitize patient ID to safe filename format."""
        sanitized = re.sub(r'[^\w\-.]', '_', patient_id)
        sanitized = re.sub(r'_+', '_', sanitized)
        sanitized = sanitized.strip('_.')
        
        return sanitized if sanitized else 'UNKNOWN'
