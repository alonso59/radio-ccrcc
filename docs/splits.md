# Split Generation

This repo includes a small utility to generate **patient-level** train/val/test splits from a dataset folder on disk.

The script is: [src/generate_splits.py](../src/generate_splits.py)

## Expected Folder Structure

The generator scans `images/` and expects class subfolders:

```
DATA_FOLDER/
├── images/
│   ├── A/
│   │   └── <case_folder>/*.(npy|nii.gz)
│   ├── B/
│   └── ...
└── segmentation/              # optional; same layout as images/
    ├── A/
    └── ...
```

Notes:
- Splits are created **by patient_id**, not by individual files.
- `patient_id` is inferred from the case folder name:
  - `case_00093` → `case_00093`
  - `TCGA-XX-YYYY` → `TCGA-XX-YYYY`

## Usage

### Recommended: Use config file
```bash
# Parameters read from config (N_FOLDS, TEST_RATIO, RANDOM_SEED)
python src/generate_splits.py --config config/planner.yaml
```

### Legacy: Direct path
```bash
python src/generate_splits.py data/dataset/tcga_kirc_tumor \
  --test-ratio 0.2 \
  --n-folds 5 \
  --seed 42
```

## Outputs

The script writes JSON files into `DATA_FOLDER/`:

- `splits_images.json` (always)
- `splits_segmentation.json` (only if `segmentation/` exists)

Each JSON contains:
- `params`: `n_folds`, `test_ratio`, `random_seed`
- `ct_folds`: list of folds with `train_case_ids`, `val_case_ids`, and file lists
- `test_case_ids`: patient IDs for the test split
- `test_ct_files`: file list for the test split

## When to Use Which Split File

- Use `splits_images.json` when your pipeline reads from `images/`.
- Use `splits_segmentation.json` when you need matching file lists from `segmentation/`.
