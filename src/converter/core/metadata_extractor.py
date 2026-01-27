"""
DICOM metadata extraction
"""

import SimpleITK as sitk
from typing import Dict

from ..utils.dicom_utils import get_metadata_value
from ..utils.formatting import format_dicom_date, format_dicom_time, calculate_age


class MetadataExtractor:
    @staticmethod
    def extract_comprehensive_metadata(
        reader: sitk.ImageSeriesReader,
        image: sitk.Image,
        custom_metadata: Dict
    ) -> Dict:
        metadata = custom_metadata.copy()
        
        metadata['image_orientation'] = get_metadata_value(reader, '0020|0037', 'nan')
        metadata['sex'] = get_metadata_value(reader, '0010|0040', 'nan')
        metadata['DOB'] = format_dicom_date(get_metadata_value(reader, '0010|0030', 'nan'))
        metadata['age'] = get_metadata_value(reader, '0010|1010', 'nan')
        
        dos_raw = get_metadata_value(reader, '0008|0020', 'nan')
        metadata['DOS'] = format_dicom_date(dos_raw)
        metadata['study_time'] = format_dicom_time(get_metadata_value(reader, '0008|0030', 'nan'))
        metadata['series_description'] = get_metadata_value(reader, '0008|103e', 'nan')
        metadata['series_number'] = get_metadata_value(reader, '0020|0011', 'nan')
        metadata['modality'] = get_metadata_value(reader, '0008|0060', 'nan')
        metadata['manufacturer'] = get_metadata_value(reader, '0008|0070', 'nan')
        metadata['manufacturer_model'] = get_metadata_value(reader, '0008|1090', 'nan')
        metadata['institution'] = get_metadata_value(reader, '0008|0080', 'nan')
        metadata['kvp'] = get_metadata_value(reader, '0018|0060', 'nan')
        metadata['contrast_agent'] = get_metadata_value(reader, '0018|0010', 'nan')
        metadata['slice_thickness'] = get_metadata_value(reader, '0018|0050', 'nan')
        
        metadata['age_at_scan'] = (calculate_age(metadata['DOB'], metadata['DOS']) 
                                   if metadata['DOB'] != 'nan' and metadata['DOS'] != 'nan' else 'nan')
        return metadata
