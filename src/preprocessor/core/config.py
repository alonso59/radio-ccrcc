from pathlib import Path
from typing import Dict, Any
import yaml


# =============================================================================
# DATASET ID UTILITIES
# =============================================================================
DEFAULT_OUTPUT_BASE = 'data/dataset'


def derive_output_dir(config: Dict[str, Any]) -> str:
    """
    Derive OUTPUT_DIR from DATASET_ID if not explicitly set.
    
    If DATASET_ID is provided and OUTPUT_DIR is not, computes:
        {OUTPUT_BASE}/Dataset{DATASET_ID}/voi
    
    Args:
        config: Configuration dictionary
        
    Returns:
        OUTPUT_DIR path string
    """
    if config.get('OUTPUT_DIR'):
        return config['OUTPUT_DIR']
    
    dataset_id = config.get('DATASET_ID')
    if dataset_id is None:
        raise ValueError("Either OUTPUT_DIR or DATASET_ID must be provided")
    
    output_base = config.get('OUTPUT_BASE', DEFAULT_OUTPUT_BASE)
    return str(Path(output_base) / f"Dataset{dataset_id}" / "voi")


# =============================================================================
# CONFIG LOADING
# =============================================================================
def load_config(config_path) -> Dict[str, Any]:
    config_path = Path(config_path)
    assert config_path.exists(), f"Config file not found: {config_path}"

    with open(config_path, 'r') as f:
        user_config = yaml.safe_load(f)
    
    # Derive OUTPUT_DIR from DATASET_ID if needed
    if not user_config.get('OUTPUT_DIR') and user_config.get('DATASET_ID'):
        user_config['OUTPUT_DIR'] = derive_output_dir(user_config)
        print(f"Derived OUTPUT_DIR from DATASET_ID: {user_config['OUTPUT_DIR']}")
    
    print(f"Loaded config from: {config_path}")
    return user_config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    required = ['IMAGE_FOLDER', 'MASK_FOLDER', 'TARGET_SPACING']
    for key in required:
        if config.get(key) is None:
            raise ValueError(f"Missing required config key: {key}")
    
    # OUTPUT_DIR can be derived from DATASET_ID
    if config.get('OUTPUT_DIR') is None and config.get('DATASET_ID') is None:
        raise ValueError("Either OUTPUT_DIR or DATASET_ID must be provided")
    
    spacing = config['TARGET_SPACING']
    if len(spacing) != 3 or any(s <= 0 for s in spacing):
        raise ValueError("TARGET_SPACING must have 3 positive values")
    
    if config['EXPANSION_MM'] < 0:
        raise ValueError("EXPANSION_MM must be non-negative")
    
    if config.get('MIN_VOI_SIZE'):
        if len(config['MIN_VOI_SIZE']) != 3:
            raise ValueError("MIN_VOI_SIZE must have 3 values")
    
    ratio = config.get('MIN_HU_IN_RANGE_RATIO', 0)
    if ratio < 0 or ratio > 1:
        raise ValueError("MIN_HU_IN_RANGE_RATIO must be in [0, 1]")
