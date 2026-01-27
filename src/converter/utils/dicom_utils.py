"""
DICOM reading and metadata utilities
"""

import SimpleITK as sitk
from typing import Optional


def get_metadata_value(reader: sitk.ImageSeriesReader, tag: str, default: str = "") -> str:
    try:
        if reader.HasMetaDataKey(0, tag):
            return reader.GetMetaData(0, tag).strip()
    except:
        pass
    return default


def read_dicom_series(series_path: str) -> Optional[tuple]:
    try:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(series_path)
        if not dicom_names:
            return None
        
        reader.SetFileNames(dicom_names)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        return reader, reader.Execute(), dicom_names
    except Exception:
        return None


def read_dicom_series_with_quality_check(series_files: list) -> Optional[tuple]:
    import sys
    import io
    import re
    
    try:
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(series_files)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        
        old_stderr = sys.stderr
        sys.stderr = captured_stderr = io.StringIO()
        try:
            image = reader.Execute()
        finally:
            sys.stderr = old_stderr
        
        stderr_text = captured_stderr.getvalue()
        quality_info = {'nonuniformity_mm': None, 'severity': 'ok', 'warning_message': None}
        
        match = re.search(r'maximum nonuniformity[:\s]+(\d+\.?\d*)', stderr_text, re.IGNORECASE)
        if match:
            nonuniformity = float(match.group(1))
            quality_info['nonuniformity_mm'] = nonuniformity
            quality_info['warning_message'] = stderr_text.strip()
            quality_info['severity'] = 'critical' if nonuniformity > 50 else 'moderate' if nonuniformity > 10 else 'minor'
        
        return reader, image, quality_info
    except Exception:
        return None


def convert_to_ras(image: sitk.Image) -> sitk.Image:
    orient_filter = sitk.DICOMOrientImageFilter()
    orient_filter.SetDesiredCoordinateOrientation("RAS")
    return orient_filter.Execute(image)


def embed_metadata(image: sitk.Image, reader: sitk.ImageSeriesReader, 
                   custom_metadata: dict) -> sitk.Image:
    from datetime import datetime
    
    metadata_tags = {
        'PatientID': '0010|0020',
        'PatientName': '0010|0010',
        'StudyDate': '0008|0020',
        'StudyTime': '0008|0030',
        'SeriesDescription': '0008|103e',
        'ProtocolName': '0018|1030',
        'Modality': '0008|0060',
        'Manufacturer': '0008|0070',
        'ManufacturerModelName': '0008|1090',
        'InstitutionName': '0008|0080',
        'SeriesNumber': '0020|0011',
        'AcquisitionNumber': '0020|0012',
        'SliceThickness': '0018|0050',
        'KVP': '0018|0060',
        'ContrastBolusAgent': '0018|0010',
    }
    
    for key, tag in metadata_tags.items():
        value = get_metadata_value(reader, tag, '')
        if value:
            image.SetMetaData(key, value)
    
    image.SetMetaData('ConversionDate', datetime.now().isoformat())
    image.SetMetaData('ConversionTool', 'UniversalDICOMConverter')
    image.SetMetaData('DatasetType', custom_metadata.get('dataset_type', 'generic'))
    image.SetMetaData('DetectedProtocol', custom_metadata.get('protocol', 'undefined'))
    image.SetMetaData('AssignedClass', custom_metadata.get('class_label', 'NG'))
    image.SetMetaData('CaseID', custom_metadata.get('case_id', ''))
    image.SetMetaData('ScanIndex', str(custom_metadata.get('scan_idx', 0)))
    return image
