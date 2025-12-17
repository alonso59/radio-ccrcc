"""
Trainer Factory - Factory Pattern for creating trainers.
Follows the Factory design pattern and Dependency Inversion Principle.
"""
from typing import Dict, Any, Optional
from omegaconf import DictConfig
import torch

from .base_trainer import BaseTrainer
from .autoencoder_trainer import AutoencoderTrainer
from .adversarial_autoencoder_trainer import AdversarialAutoencoderTrainer
from .classifier_trainer import ClassifierTrainer


class TrainerFactory:
    """
    Factory for creating appropriate trainer based on configuration.
    Centralizes trainer creation logic following Single Responsibility Principle.
    """
    
    TRAINER_REGISTRY = {
        "vae": AutoencoderTrainer,
        "gan": AdversarialAutoencoderTrainer,
        "classifier": ClassifierTrainer,
    }
    
    @classmethod
    def create(
        cls,
        cfg: DictConfig,
        dataloaders: Dict[str, Any],
        device: torch.device,
        logger: Any,
        **kwargs
    ) -> BaseTrainer:
        """
        Create and return appropriate trainer based on configuration.
        
        Args:
            cfg: Hydra configuration object
            dataloaders: Dictionary containing train/val dataloaders
            device: Torch device (cuda/cpu)
            logger: Logger instance (e.g., TensorBoardLogger)
            **kwargs: Additional arguments specific to trainer type
            
        Returns:
            Configured trainer instance
            
        Raises:
            ValueError: If training_mode is not recognized
        """
        training_mode = cfg.trainer.training_mode.lower()
        
        if training_mode not in cls.TRAINER_REGISTRY:
            raise ValueError(
                f"Unknown training mode: '{training_mode}'. "
                f"Available modes: {list(cls.TRAINER_REGISTRY.keys())}"
            )
        
        trainer_class = cls.TRAINER_REGISTRY[training_mode]
        
        # Common arguments for all trainers
        common_args = {
            "cfg": cfg,
            "dataloaders": dataloaders,
            "device": device,
            "logger": logger,
            "callbacks": kwargs.get("callbacks", None),
            "max_epochs": kwargs.get("max_epochs", None),
            "normalization_stats": kwargs.get("normalization_stats", None),  # NEW
        }
        
        # Mode-specific trainer creation
        if training_mode in ["autoencoder", "vae"]:
            return cls._create_autoencoder_trainer(common_args, kwargs)
        
        elif training_mode in ["adversarial", "gan"]:
            return cls._create_adversarial_trainer(common_args, kwargs)
        
        elif training_mode == "classifier":
            return cls._create_classifier_trainer(common_args, kwargs)
        
        else:
            raise ValueError(f"Unhandled training mode: {training_mode}")
    
    @staticmethod
    def _create_autoencoder_trainer(
        common_args: Dict[str, Any],
        kwargs: Dict[str, Any]
    ) -> AutoencoderTrainer:
        """Create pure autoencoder trainer."""
        if "model" not in kwargs:
            raise ValueError("'model' must be provided for autoencoder training")
        
        return AutoencoderTrainer(
            model=kwargs["model"],
            **common_args
        )
    
    @staticmethod
    def _create_adversarial_trainer(
        common_args: Dict[str, Any],
        kwargs: Dict[str, Any]
    ) -> AdversarialAutoencoderTrainer:
        """Create adversarial autoencoder trainer."""
        if "model" not in kwargs:
            raise ValueError("'model' must be provided for adversarial training")
        
        return AdversarialAutoencoderTrainer(
            model=kwargs["model"],
            metrics=kwargs.get("metrics", None),
            **common_args
        )
    
    @staticmethod
    def _create_classifier_trainer(
        common_args: Dict[str, Any],
        kwargs: Dict[str, Any]
    ) -> ClassifierTrainer:
        """Create classifier trainer with frozen autoencoder."""
        if "model_auto" not in kwargs:
            raise ValueError("'model_auto' must be provided for classifier training")
        if "model_class" not in kwargs:
            raise ValueError("'model_class' must be provided for classifier training")
        
        return ClassifierTrainer(
            model_auto=kwargs["model_auto"],
            model_class=kwargs["model_class"],
            class_names=kwargs.get("class_names", None),
            allow_resize=kwargs.get("allow_resize", False),
            **common_args
        )
    
    @classmethod
    def register_trainer(cls, name: str, trainer_class: type) -> None:
        """
        Register a new trainer type.
        Allows for extension without modifying core code (Open/Closed Principle).
        
        Args:
            name: Name identifier for the trainer
            trainer_class: Trainer class (must inherit from BaseTrainer)
        """
        if not issubclass(trainer_class, BaseTrainer):
            raise TypeError(f"{trainer_class} must inherit from BaseTrainer")
        
        cls.TRAINER_REGISTRY[name] = trainer_class
        print(f"Registered trainer '{name}': {trainer_class.__name__}")
    
    @classmethod
    def get_available_modes(cls) -> list:
        """Return list of available training modes."""
        return list(cls.TRAINER_REGISTRY.keys())
