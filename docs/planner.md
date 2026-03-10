# Dataset Planner

The planner coordinates dataset preparation steps without touching source data. It uses your config to find source paths and only writes outputs under `data/dataset/`.

## What it does

- **Preprocess**: VOI extraction into the standardized output folder
- **Fingerprint**: Recompute dataset statistics from saved VOI outputs (BBOX region, clamped HU range)
- **Splits**: Generate train/val/test splits from the output folder
- **Validate**: Check output folder structure is ready for downstream CNN workflows (TODO)

## Usage

```bash
# Full pipeline
python src/planner.py --config config/planner.yaml --all

# Preprocess only
python src/planner.py --config config/planner.yaml --preprocess

# Fingerprint only
python src/planner.py --config config/planner.yaml --fingerprint

# Splits only
python src/planner.py --config config/planner.yaml --splits

# Validate structure only
python src/planner.py --config config/planner.yaml --validate
```

## Configuration

The planner uses the same config as the preprocessor. Recommended: use `DATASET_ID` for nnUNet-like structure:

```yaml
DATASET_ID: 320                     # Dataset320_TCGA_KITS
OUTPUT_BASE: "data/dataset"         # → data/dataset/Dataset320/voi

# Split parameters (used by generate_splits.py)
N_FOLDS: 5
TEST_RATIO: 0.1
RANDOM_SEED: 42
```

See [planner.yaml](../config/planner.yaml) for full example.

## Notes

- The planner does **not** move or modify source data folders.
- For standalone fingerprinting, make sure `SAVE_IMAGES=true` during preprocessing.
- CNN-specific checks will be added later (TODO).
