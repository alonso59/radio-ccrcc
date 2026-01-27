# Universal DICOM to NIfTI Converter

Production-ready DICOM to NIfTI conversion with automatic dataset detection, protocol classification, and comprehensive metadata extraction.

## Features

- **Dataset Detection**: Auto-identifies TCGA, UKBonn, KiTS, and generic datasets
- **Protocol Classification**: Detects CT phases (non-contrast/arterial/venous) from metadata and filenames
- **Anatomical Filtering**: Processes AXIAL scans only, logs skipped series
- **Metadata Extraction**: Demographics, scanner specs, image properties (30+ fields)
- **Structured Output**: CSV manifest, organized class folders, RAS+ orientation
- **Quality Assurance**: Post-conversion validation with detailed logging

## Architecture

```
src/converter/
├── __init__.py              # Package exports
├── cli.py                   # Command-line interface
├── convert.py               # Convenience wrapper script
├── core/                    # Core conversion logic
│   ├── converter.py         # Main orchestration class
│   ├── dataset_detector.py  # Dataset type detection
│   ├── protocol_detector.py # CT protocol phase detection
│   ├── scan_plane_detector.py # Anatomical plane detection
│   ├── metadata_extractor.py # DICOM metadata extraction
│   └── validator.py         # Conversion validation
└── utils/                   # Utility functions
    ├── dicom_utils.py       # DICOM reading helpers
    └── formatting.py        # Date/time formatting
```

## Usage

### Command Line
```bash
python -m src.converter.cli \
  --input data/tcga_dicom/tcga_kirc \
  --output data/dataset/tcga_kirc_nii_v3 \
  --csv data/filtered_vessel_evaluation.csv
```

### Python API
```python
from src.converter import UniversalDICOMConverter

converter = UniversalDICOMConverter(
    input_dir='data/tcga_dicom/tcga_kirc',
    output_dir='data/dataset/tcga_kirc_nii_v3',
    classification_csv='data/filtered_vessel_evaluation.csv'
)
summary = converter.run()
```

### Arguments

- `-i, --input`: DICOM input directory (required)
- `-o, --output`: NIfTI output directory (required)
- `-c, --csv`: Patient classification CSV (optional)
- `--start-case-id`: Starting case number (default: 1)

## Output Structure

```
output_dir/
├── A/B/NG/                  # Classification folders
│   ├── 00_case_00001_0000.nii.gz  # Format: {scan_idx:02d}_{case_id}_0000.nii.gz
│   └── 01_case_00001_0000.nii.gz
├── conversion_summary.json  # Conversion log with errors/skipped series
└── manifest.csv            # Metadata table (30+ columns)
```

## Manifest Columns

| Category | Fields |
|----------|--------|
| **Identifiers** | filename, patient_id, case_id, class_label, scan_idx |
| **Protocol** | protocol (nc/art/ven/undefined), scan_plane, protocol_source, image_orientation |
| **Demographics** | sex, DOB, age, age_at_scan |
| **Acquisition** | DOS, study_time, series_description, series_number |
| **Scanner** | manufacturer, model, institution, kvp, contrast_agent |
| **Image** | spacing_x/y/z, dim_x/y/z, slice_thickness, num_slices |
| **Metadata** | conversion_date, dataset_type, output_path, validation_status |

## Error Handling

- **Skipped Series**: Non-AXIAL scans logged in `conversion_summary.json`
- **Failed Cases**: Conversion errors tracked with detailed logs
- **Validation**: Post-conversion integrity checks ensure data quality

## Examples

```bash
# TCGA with classification
python -m src.converter.cli \
  -i data/tcga_dicom/tcga_kirc \
  -o data/dataset/tcga_kirc_nii_v3 \
  -c data/filtered_vessel_evaluation.csv

# UKBonn without classification
python -m src.converter.cli \
  -i data/dataset/ukb2025_raw \
  -o data/dataset/ukb2025_nii

# Custom case numbering
python -m src.converter.cli \
  -i data/dicom \
  -o data/nifti \
  --start-case-id 100
```

## Extending

**Add Protocol Keywords** - Edit [protocol_detector.py](../src/converter/core/protocol_detector.py):
```python
NC_KEYWORDS = [..., 'your_keyword']
```

**Custom Metadata** - Extend [metadata_extractor.py](../src/converter/core/metadata_extractor.py):
```python
metadata['custom_field'] = get_metadata_value(reader, 'XXXX|XXXX', 'nan')
```

**New Dataset Type** - Update [dataset_detector.py](../src/converter/core/dataset_detector.py):
```python
if 'your_pattern' in patient_id:
    return 'your_dataset'
```
