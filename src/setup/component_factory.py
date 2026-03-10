"""
Component Factory - Creates models, loggers, callbacks, and dataloaders.
Follows Factory Pattern and Dependency Injection principles.
"""
from typing import Tuple, Dict, Any, Optional
import torch
from omegaconf import DictConfig
from torchinfo import summary

from ..models.autoencoder import Autoencoder
from ..models.classifier import Classifier3D
from ..dataloader.dataloader import DataLoaderFactory
from ..utils.callbacks import ModelCheckpoint
from ..utils.logger import TensorBoardLogger
import logging

class ComponentFactory:
    """Factory for creating training components."""
    
    @staticmethod
    def create_autoencoder(cfg: DictConfig, show_summary: bool = True) -> torch.nn.Module:
        """Create autoencoder model."""
        model = Autoencoder(cfg)
        if show_summary:
            summary(
                model, 
                input_size=(1, 1, 96, 96, 64),
                col_names=["input_size", "output_size", "num_params", "trainable"],
                depth=4
            )
        return model
    
    @staticmethod
    def create_classifier(num_classes: int) -> torch.nn.Module:
        """Create classifier model."""
        return Classifier3D(num_classes)
    
    @staticmethod
    def wrap_dataparallel(
        model: torch.nn.Module,
        device: str,
        use_dataparallel: bool
    ) -> torch.nn.Module:
        """Wrap model with DataParallel if needed."""
        if use_dataparallel and torch.cuda.device_count() > 1 and device == 'cuda':
            logging.info(f"🔥 Using DataParallel with {torch.cuda.device_count()} GPUs")
            return torch.nn.DataParallel(model)
        elif device == 'cuda':
            logging.info(f"💻 Using single GPU: cuda:0")
        return model
    
    @staticmethod
    def create_dataloaders(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Create train and validation dataloaders with normalization statistics.
        
        Returns:
            Tuple of (dataloaders_dict, normalization_stats_dict)
        """
        train_loader, val_loader, norm_stats = DataLoaderFactory.create_loaders(cfg)
        dataloaders = {'train': train_loader, 'val': val_loader}
        return dataloaders, norm_stats
    
    @staticmethod
    def create_logger(experiment_name: str) -> TensorBoardLogger:
        """Create TensorBoard logger."""
        # Hydra sets the working directory to the run directory; using a relative
        # root keeps TensorBoard artifacts inside the run folder.
        return TensorBoardLogger(
            root="tb",
            experiment_name=experiment_name,
            add_timestamp=False,
            use_date_structure=False,
        )
    
    @staticmethod
    def create_checkpoint_callback(
        checkpoint_dir: str,
        monitor: str,
        mode: str,
        filename: str,
        model: torch.nn.Module
    ) -> ModelCheckpoint:
        """Create model checkpoint callback."""
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            filename=filename
        )
        checkpoint_callback.set_model(model)
        return checkpoint_callback
