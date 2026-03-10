# VOI Preprocessor

Modular pipeline for extracting kidney Volumes of Interest (VOIs) from CT scans with automatic kidney separation, resampling, and quality validation.

## Features

- **Dual Input Support**: NIfTI (.nii.gz) or NumPy (.npy) formats with auto-detection
- **Kidney Separation**: Automatic left/right kidney splitting based on tumor labels
- **Standardization**: RAS+ orientation, configurable spacing, HU clipping
- **Quality Filters**: Minimum voxel thresholds, boundary detection, HU range validation
- **Comprehensive Outputs**: Masks, segmentations, dataset manifests, fingerprints
- **Batch Processing**: Multi-patient/multi-scan with longitudinal data support

## Architecture

```
src/preprocessor/
├── __init__.py                  # Package exports
├── core/                        # Core pipeline logic
│   ├── config.py                # Configuration loading & validation
│   ├── pipeline.py              # Main VOIPreprocessor orchestrator
│   └── processing.py            # Image processing operations
├── utils/                       # Utilities
│   ├── common.py                # Constants & helper functions
│   ├── io.py                    # File discovery & I/O operations
│   └── metrics.py               # Tumor metrics & fingerprint generation
└── analysis/                    # Quality analysis
    ├── validators.py            # Edge case validators
    ├── sanity_check.py          # Post-processing sanity checks
    └── edge_case_analyzer.py    # Dataset-wide edge case analysis
```

## Usage

### Command Line
```bash
# Full preprocessing
python src/preprocessor.py --config config/preprocessor_config.yaml

# Recompute fingerprints from saved outputs (requires SAVE_IMAGES=true)
python src/preprocessor.py --config config/preprocessor_config.yaml --fingerprint-only

# Generate splits only
python src/preprocessor.py --config config/preprocessor_config.yaml --splits-only

# Edge case analysis only
python src/preprocessor.py --config config/preprocessor_config.yaml --analyze-only
```

### Python API
```python
from src.preprocessor import VOIPreprocessor, load_config

config = load_config('config/preprocessor_config.yaml')
preprocessor = VOIPreprocessor(config)
results = preprocessor.run_batch()
```

## Configuration

Edit [planner.yaml](../config/planner.yaml):

```yaml
# Input/Output (sources stay where they are)
IMAGE_FOLDER: "data/dataset/tcga_kirc_nii/"
MASK_FOLDER: "data/dataset/tcga_kirc_seg/"

# DatasetID-based output (recommended - nnUNet-like structure)
DATASET_ID: 320                     # Dataset320_TCGA_KITS
OUTPUT_BASE: "data/dataset"         # → data/dataset/Dataset320/voi

# Alternative: Explicit output path (legacy)
# OUTPUT_DIR: "data/dataset/tcga_kirc_tumor/"

# Processing
TARGET_SPACING: [1.0, 1.0, 1.0]    # mm/voxel
EXPANSION_MM: 1.0                   # bbox margin
MIN_VOI_SIZE: [128, 128, 128]       # minimum dimensions
HU_RANGE: [-200, 300]               # clipping range

# Output options
SAVE_MASK: true                     # Save mask arrays (uint8)
SAVE_SEGMENTATION: true             # Save masked CT images (float32)
SAVE_IMAGES: true                   # Save full VOI (required for fingerprint-only)

# Labels
LABELS:
  1: "kidney"
  2: "tumor"
BBOX_LABELS: [2]                    # labels for bbox computation
MASK_LABELS: [2]                    # labels to keep in output

# Quality Filters
MIN_KIDNEY_VOXELS: 20000
MIN_TUMOR_VOXELS: 100
BOUNDARY_MARGIN: 5                  # voxels

# Class Filter
CLASS_FILTER: ["A", "B"]            # null = all subfolders

# Splits (for generate_splits.py)
N_FOLDS: 5                          # cross-validation folds
TEST_RATIO: 0.1                     # test set ratio (0.0-1.0)
RANDOM_SEED: 42                     # reproducibility
```

## Pipeline Workflow

```
1. File Discovery
   ├─ Auto-detect format (NIfTI/NumPy)
   ├─ Match image-mask pairs by suffix
   └─ Filter by CLASS_FILTER subdirectories

2. Per-Case Processing
   ├─ Load image & mask
   ├─ Convert to RAS+ orientation
   ├─ Clip HU range [-200, 300]
   ├─ Resample to TARGET_SPACING
   ├─ Separate kidneys (left/right by centroid)
   └─ Extract VOI (bbox + expansion)

3. Quality Validation
   ├─ Check minimum voxel counts
   ├─ Validate VOI dimensions
   ├─ Detect boundary contact
   └─ Compute tumor metrics

4. Output Generation
   ├─ Save masks & segmentations (.npy)
   ├─ Generate dataset.json
   ├─ Compute dataset_fingerprint.json
   └─ Run sanity checks
```

## Output Structure

```
{OUTPUT_DIR}/                            # e.g., data/dataset/Dataset320/voi/
├── mask/{class}/{patient_id}/
│   └── {case_name}_{side}.npy           # Binary masks (uint8)
├── segmentation/{class}/{patient_id}/
│   └── {case_name}_{side}.npy           # Masked CT (float32, bg=-1000)
├── images/{class}/{patient_id}/
│   └── {case_name}_{side}.npy           # Full CT VOI (required for fingerprint-only)
├── dataset.json                         # Case manifest & source paths
├── dataset_fingerprint.json             # Stats from BBOX region (clamped HU range)
├── sanity_check.json                    # Validation report
└── splits_final.json                    # Train/val/test splits (images only)
```

### Filename Format
`{case_name}_{side}.npy`

Examples:
- `TCGA-B0-5399_L.npy` - Left kidney
- `00_case_00113_R.npy` - Right kidney, scan 00

## JSON Outputs

### dataset.json
Complete case registry with source paths:
```json
{
  "version": "1.1.0",
  "dataset_id": 320,
  "channel_names": {"0": "CT"},
  "labels": {"1": "kidney", "2": "tumor"},
  "source": {
    "image_folder": "data/dataset/tcga_kirc_nii/",
    "mask_folder": "data/dataset/tcga_kirc_seg/",
    "image_suffix": "_0000.nii.gz",
    "mask_suffix": ".nii.gz",
    "input_format": "nifti"
  },
  "statistics": {
    "num_cases": 1234,
    "num_patients": 456,
    "num_failed": 12
  },
  "patients": {
    "TCGA-B0-5399": {
      "class": "A",
      "cases": ["TCGA-B0-5399"],
      "kidneys": ["L", "R"],
      "files": {
        "mask": ["mask/A/TCGA-B0-5399/TCGA-B0-5399_L.npy", ...],
        "segmentation": ["segmentation/A/TCGA-B0-5399/TCGA-B0-5399_L.npy", ...]
      }
    }
  }
}
```

### dataset_fingerprint.json
Population-level statistics from BBOX region for normalization:
```json
{
  "fingerprint_type": "bbox",
  "foreground_intensity_properties_per_channel": {
    "0": {
      "mean": 45.2,
      "median": 42.1,
      "std": 28.5,
      "percentile_25_0": 30.5,
      "percentile_75_0": 58.3,
      "iqr": 27.8
    }
  },
  "normalization": {
    "method": "iqr",
    "percentile_25": 30.5,
    "percentile_75": 58.3,
    "median": 42.1,
    "iqr": 27.8
  },
  "voxel_collection_info": {
    "method": "bbox_only",
    "region": "BBOX (union of BBOX_LABELS)",
    "hu_range": [-200, 300],
    "total_voxels": 15234567,
    "filtered_voxels": 14987234,
    "filter_ratio": 0.984
  }
}
```

### sanity_check.json
Validation results per case:
```json
{
  "config": {
    "MIN_VOI_SIZE": [128, 128, 128],
    "HU_RANGE": [-200, 300],
    "BOUNDARY_MARGIN": 5
  },
  "summary": {
    "total_cases": 1234,
    "flagged_cases": 42,
    "flag_types": {
      "boundary_contact": 15,
      "small_tumor": 18,
      "anomalous_hu": 9
    }
  },
  "cases": {
    "TCGA-B0-5399_L": {
      "flags": ["Touches Z-max boundary"],
      "details": {...}
    }
  }
}
```

## Processing Details

### Kidney Separation
Kidneys are split by comparing tumor centroid X-coordinates:
- **Left kidney**: centroid_x < image_center_x
- **Right kidney**: centroid_x ≥ image_center_x

For multiple disconnected tumors per kidney, uses largest component.

### VOI Extraction
1. Compute bounding box from `BBOX_LABELS` (typically tumor=2)
2. Expand by `EXPANSION_MM` in all directions
3. Clip to image boundaries
4. Extract and validate against `MIN_VOI_SIZE`

### HU Normalization
- **Clipping**: Values outside [-200, 300] are clamped
- **Background**: Non-kidney regions set to -1000 HU in segmentations
- **Foreground**: Only voxels in [-200, 300] used for statistics

## Quality Validation

The pipeline flags but does **not reject** cases with:

| Validator | Detects | Threshold |
|-----------|---------|-----------|
| `validate_boundary_contact` | VOI touches image edges | `BOUNDARY_MARGIN` |
| `validate_small_tumor` | Insufficient tumor voxels | `MIN_TUMOR_VOXELS` |
| `validate_voi_size` | VOI smaller than minimum | `MIN_VOI_SIZE` |
| `validate_anomalous_hu` | Excessive out-of-range HU | `MIN_HU_IN_RANGE_RATIO` |

Flagged cases are logged in `sanity_check.json` for manual review.

## Examples

### Process TCGA Dataset
```bash
python -m src.preprocessor \
  --config config/preprocessor_config.yaml
```

### Custom Configuration
```python
from src.preprocessor import VOIPreprocessor

config = {
    'IMAGE_FOLDER': 'data/ukb_nii',
    'MASK_FOLDER': 'data/ukb_seg',
    'OUTPUT_DIR': 'data/ukb_voi',
    'TARGET_SPACING': [1.5, 1.5, 1.5],
    'EXPANSION_MM': 5.0,
    'MIN_VOI_SIZE': [96, 96, 96],
    'CLASS_FILTER': None,  # Process all classes
}

preprocessor = VOIPreprocessor(config)
results = preprocessor.run_batch()
```

### Load Existing Dataset
```python
from src.preprocessor import VOIPreprocessor, load_config

config = load_config('config/preprocessor_config.yaml')
preprocessor = VOIPreprocessor(config)
preprocessor.load_from_dataset()  # Loads from existing dataset.json
```

### Run Sanity Check
```python
from src.preprocessor import SanityChecker, load_config

config = load_config('config/preprocessor_config.yaml')
checker = SanityChecker(config, output_dir='data/tcga_kirc_tumor')
report = checker.check_dataset()

# Access flagged cases
for case_id, flags in report['cases'].items():
    if flags['flags']:
        print(f"{case_id}: {flags['flags']}")
```

## Extending

**Add Custom Validator** - Edit [validators.py](../src/preprocessor/analysis/validators.py):
```python
def validate_custom_metric(voi: np.ndarray, threshold: float) -> Dict:
    metric = compute_metric(voi)
    return {
        'valid': metric >= threshold,
        'flags': ['Custom issue'] if metric < threshold else [],
        'metric_value': metric
    }
```

**Modify Processing Pipeline** - Edit [processing.py](../src/preprocessor/core/processing.py):
```python
def custom_preprocessing(image: sitk.Image) -> sitk.Image:
    # Apply custom filters
    return processed_image
```

**Change Kidney Splitting Logic** - Edit [separate_kidneys()](../src/preprocessor/core/processing.py#L100):
```python
# Current: splits by X-coordinate centroid
# Custom: could use anatomical side labels if available
```
