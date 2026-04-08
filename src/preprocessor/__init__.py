"""VOI Preprocessor: kidney VOI extraction from CT scans."""

__version__ = "4.0.0"


def load_config(path):
    """Load YAML config and derive OUTPUT_DIR from DATASET_ID when needed."""
    from pathlib import Path
    import yaml

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path) as f:
        cfg = yaml.safe_load(f)

    if not cfg.get("OUTPUT_DIR") and cfg.get("DATASET_ID"):
        base = cfg.get("OUTPUT_BASE", "data/dataset")
        cfg["OUTPUT_DIR"] = str(Path(base) / f"Dataset{cfg['DATASET_ID']}" / "voi")

    return cfg
