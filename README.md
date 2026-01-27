# radio-ccrcc

Deep learning codebase for radiology-driven ccRCC research.

The repository supports an end-to-end workflow spanning **CT data curation**, **segmentation-assisted preprocessing**, **self-supervised representation learning**, and **downstream subtyping/classification experiments** (e.g., studying imaging phenotypes that correlate with histopathologic vascular branching patterns).

## Highlights

- **CT dataset preparation**: DICOM → NIfTI conversion, dataset manifests, and split generation.
- **Segmentation integration**: nnU-Net v2-compatible layout and workflows for kidney/tumor masking.
- **Preprocessing**: VOI (volume-of-interest) extraction and analysis utilities.
- **Deep learning training pipeline**: Hydra-configured training entrypoint and modular trainers/models.
- **Reproducible experiments**: YAML configs + structured outputs.

## Documentation

- Documentation index: [docs/index.md](docs/index.md)
- DICOM → NIfTI converter: [docs/converter.md](docs/converter.md)
- VOI preprocessor: [docs/preprocessor.md](docs/preprocessor.md)
- Split generation: [docs/splits.md](docs/splits.md)

## Getting Started

### Prerequisites

- Linux/macOS recommended
- Python (see `requirements.txt`)

### Install

```bash
pip install -r requirements.txt
```

### Typical Workflow

1. Convert DICOM to NIfTI
	 - See: [docs/converter.md](docs/converter.md)
2. Run preprocessing / VOI extraction
	 - See: [docs/preprocessor.md](docs/preprocessor.md)
3. Generate data splits
	 - See: [docs/splits.md](docs/splits.md)

4. Train models / run experiments
	 - Entry point: `python main.py` (Hydra)
	 - See “Training” below

## Configuration

- Main configuration lives in `config/` (e.g., `config/config.yaml`, `config/planner.yaml`).
- Project code lives under `src/`.

If you are new to the repo, start with the docs index and follow the pipeline in order.

## Repository Layout

- `main.py`: training entry point (Hydra)
- `src/models/`: model definitions (autoencoders, classifiers, etc.)
- `src/trainers/`: training loops and losses
- `src/dataloader/`: dataset + augmentation pipeline
- `src/preprocessor/` and `src/voi_preprocessor.py`: preprocessing / VOI extraction
- `src/converter/`: DICOM/NIfTI conversion utilities

## Training

Training is Hydra-driven. Common usage patterns:

```bash
python main.py
python main.py trainer.training_mode=vae
python main.py trainer.training_mode=classifier
```

There is also a Makefile shortcut used in this repo:

```bash
make train_autoencoder fold=0
```

See `config/config.yaml` and `config/model/model.yaml` for the full set of supported overrides.

## Segmentation (nnU-Net v2)

This repo includes utilities and folder structure compatible with nnU-Net v2. Example Makefile targets:

```bash
make train_seg_320_3d_fullres dev=0 fold=0
make segment dev=0 group=NG
```

These commands assume nnU-Net v2 is installed and that the expected nnU-Net environment variables / paths are configured for your machine.

## Datasets & Data Access

This repo contains utilities and metadata files to *work with* public datasets, but it does not redistribute the datasets themselves.

- **KiTS23 (Kidney Tumor Segmentation Challenge 2023)**: used commonly for kidney/renal tumor segmentation benchmarking.
- **TCGA-KIRC** (The Cancer Genome Atlas — Kidney Renal Clear Cell Carcinoma): radiology data accessed via the NCI Genomic Data Commons (GDC) and subject to TCGA/GDC terms.

When using these datasets, ensure you comply with their licenses/terms and cite the original sources.

## References (Please Cite)

If you use this repository in academic work, please also cite the upstream datasets and frameworks below.

- **KiTS23 dataset/challenge**: KiTS23 — Kidney Tumor Segmentation Challenge 2023.
	- Project page: https://kits-challenge.org/
- **TCGA-KIRC dataset**: The Cancer Genome Atlas (TCGA) — KIRC collection via the NCI GDC portal.
	- GDC portal: https://portal.gdc.cancer.gov/
- **nnU-Net v2 segmentation framework**: Fabian Isensee et al. (nnU-Net / nnU-Net v2).
	- Paper (nnU-Net): https://www.nature.com/articles/s41592-020-01008-z
	- Code (nnU-Net v2): https://github.com/MIC-DKFZ/nnUNet

### BibTeX (Convenience)

```bibtex
@misc{kits23,
	title        = {KiTS23: Kidney Tumor Segmentation Challenge 2023},
	howpublished = {\url{https://kits-challenge.org/}},
	note         = {Accessed 2026-01-27}
}

@misc{tcga_kirc_gdc,
	title        = {The Cancer Genome Atlas (TCGA) -- Kidney Renal Clear Cell Carcinoma (TCGA-KIRC)},
	howpublished = {\url{https://portal.gdc.cancer.gov/}},
	note         = {Accessed 2026-01-27}
}

@misc{nnunetv2,
	title        = {nnU-Net (v2): Self-configuring Framework for Medical Image Segmentation},
	author       = {Isensee, Fabian and others},
	howpublished = {\url{https://github.com/MIC-DKFZ/nnUNet}},
	note         = {Accessed 2026-01-27}
}

@article{isensee2021nnunet,
	title   = {nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
	author  = {Isensee, Fabian and Jaeger, Paul F. and Kohl, Simon A. A. and Petersen, Jens and Maier-Hein, Klaus H.},
	journal = {Nature Methods},
	year    = {2021},
	doi     = {10.1038/s41592-020-01008-z}
}
```

## Project Status

This repository contains both data preparation utilities and deep learning training code.

Planned documentation additions:

- Training / model documentation
- Evaluation documentation
