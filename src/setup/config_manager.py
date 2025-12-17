"""
Configuration Manager - Handles configuration loading and validation.
Follows Single Responsibility Principle.
"""
from typing import List
from omegaconf import DictConfig
import hydra


class ConfigManager:
    """Manages configuration validation and preprocessing."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        
    def validate(self) -> None:
        """Validate configuration requirements."""
        required_sections = ['trainer', 'dataset', 'model']
        for section in required_sections:
            if section not in self.cfg:
                raise ValueError(f"Missing required configuration section: {section}")
        
        if not hasattr(self.cfg.trainer, 'training_mode'):
            raise ValueError("trainer.training_mode is required in configuration")
    
    def get_training_mode(self) -> str:
        """Get the normalized training mode."""
        return self.cfg.trainer.training_mode.lower()
    
    def resolve_paths(self) -> None:
        """Resolve relative paths to absolute paths."""
        if hasattr(self.cfg.dataset, 'splits_path'):
            self.cfg.dataset.splits_path = hydra.utils.to_absolute_path(
                self.cfg.dataset.splits_path
            )
    
    def get_experiment_name(self, default_prefix: str) -> str:
        """Get experiment name with fallback."""
        return (getattr(self.cfg.trainer, 'experiment_name', None) or 
                f"{default_prefix}_fold{self.cfg.dataset.fold}")
    
    def get_device(self) -> str:
        """Get device configuration."""
        return getattr(self.cfg.trainer, 'device', 'cuda')
    
    def should_use_dataparallel(self) -> bool:
        """Check if DataParallel should be used."""
        return getattr(self.cfg.trainer, 'use_dataparallel', True)
    
    def get_num_classes(self) -> int:
        """Get number of classes for classification."""
        return int(getattr(self.cfg.model, 'num_classes', 3))
    
    def get_class_names(self) -> List[str]:
        """Get class names for classification."""
        return getattr(self.cfg, 'class_names', ["A", "B"])
    
    def allow_resize(self) -> bool:
        """Check if resizing is allowed."""
        return getattr(self.cfg, 'allow_resize', False)
