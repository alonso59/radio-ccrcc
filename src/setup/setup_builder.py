"""
Training Setup Builder - Builds training configurations using Builder Pattern.
Follows Builder Pattern, Single Responsibility, and Open/Closed Principles.
"""
from typing import Dict, Any, List, Optional
import torch
from omegaconf import DictConfig

from .component_factory import ComponentFactory
from .config_manager import ConfigManager
from ..trainers.trainer_factory import TrainerFactory
from ..utils.logger import TensorBoardLogger

class TrainingSetup:
    """Data class holding all training components."""
    
    def __init__(self, trainer: Any, logger: TensorBoardLogger):
        self.trainer = trainer
        self.logger = logger


class BaseSetupBuilder:
    """Base builder for common training setup steps."""
    
    def __init__(self, cfg: DictConfig, config_manager: ConfigManager):
        self.cfg = cfg
        self.config_manager = config_manager
        self.model = None
        self.dataloaders = None
        self.norm_stats = None  # NEW: Store normalization statistics
        self.logger = None
        self.callbacks = []
        self.device = None
    
    def build_dataloaders(self) -> 'BaseSetupBuilder':
        """Build data loaders and extract normalization statistics."""
        self.dataloaders, self.norm_stats = ComponentFactory.create_dataloaders(self.cfg.dataset)
        return self
    
    def build_device(self) -> 'BaseSetupBuilder':
        """Setup device."""
        self.device = torch.device(self.config_manager.get_device())
        return self
    
    def build_logger(self, experiment_prefix: str) -> 'BaseSetupBuilder':
        """Build logger."""
        experiment_name = self.config_manager.get_experiment_name(experiment_prefix)
        self.logger = ComponentFactory.create_logger(experiment_name)
        return self
    
    def build(self) -> TrainingSetup:
        """Build final training setup. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement build()")


class AutoencoderSetupBuilder(BaseSetupBuilder):
    """Builder for autoencoder training setup."""
    
    def build_model(self) -> 'AutoencoderSetupBuilder':
        """Build autoencoder model."""
        self.model = ComponentFactory.create_autoencoder(self.cfg)
        self.model = ComponentFactory.wrap_dataparallel(
            self.model,
            self.config_manager.get_device(),
            self.config_manager.should_use_dataparallel()
        )
        return self
    
    def build_callbacks(self) -> 'AutoencoderSetupBuilder':
        """Build callbacks."""
        checkpoint_callback = ComponentFactory.create_checkpoint_callback(
            checkpoint_dir=self.cfg.trainer.checkpoint_dir,
            monitor='val_loss',
            mode='min',
            filename='best_autoencoder.pth',
            model=self.model
        )
        self.callbacks = [checkpoint_callback]
        return self
    
    def build(self) -> TrainingSetup:
        """Build autoencoder training setup."""
        if self.dataloaders is None or self.device is None or self.logger is None:
            raise RuntimeError("Build steps must be called before build()")
        
        trainer = TrainerFactory.create(
            cfg=self.cfg,
            dataloaders=self.dataloaders,
            device=self.device,
            logger=self.logger,
            callbacks=self.callbacks,
            normalization_stats=self.norm_stats,
            model=self.model
        )
        return TrainingSetup(trainer, self.logger)


class AdversarialSetupBuilder(BaseSetupBuilder):
    """Builder for adversarial autoencoder training setup."""
    
    def build_model(self) -> 'AdversarialSetupBuilder':
        """Build adversarial autoencoder model."""
        self.model = ComponentFactory.create_autoencoder(self.cfg)
        self.model = ComponentFactory.wrap_dataparallel(
            self.model,
            self.config_manager.get_device(),
            self.config_manager.should_use_dataparallel()
        )
        return self
    
    def build_callbacks(self) -> 'AdversarialSetupBuilder':
        """Build callbacks."""
        checkpoint_callback = ComponentFactory.create_checkpoint_callback(
            checkpoint_dir=self.cfg.trainer.checkpoint_dir,
            monitor='val_recon',
            mode='min',
            filename='best_adversarial.pth',
            model=self.model
        )
        self.callbacks = [checkpoint_callback]
        return self
    
    def build(self) -> TrainingSetup:
        """Build adversarial training setup."""
        if self.dataloaders is None or self.device is None or self.logger is None:
            raise RuntimeError("Build steps must be called before build()")
        
        trainer = TrainerFactory.create(
            cfg=self.cfg,
            dataloaders=self.dataloaders,
            device=self.device,
            logger=self.logger,
            callbacks=self.callbacks,
            normalization_stats=self.norm_stats,
            model=self.model
        )
        return TrainingSetup(trainer, self.logger)


class ClassifierSetupBuilder(BaseSetupBuilder):
    """Builder for classifier training setup."""
    
    def __init__(self, cfg: DictConfig, config_manager: ConfigManager):
        super().__init__(cfg, config_manager)
        self.model_auto = None
        self.model_class = None
    
    def build_model(self) -> 'ClassifierSetupBuilder':
        """Build autoencoder and classifier models."""
        # Autoencoder
        self.model_auto = ComponentFactory.create_autoencoder(self.cfg, show_summary=False)
        self.model_auto = ComponentFactory.wrap_dataparallel(
            self.model_auto,
            self.config_manager.get_device(),
            self.config_manager.should_use_dataparallel()
        )
        
        # Classifier
        num_classes = self.config_manager.get_num_classes()
        self.model_class = ComponentFactory.create_classifier(num_classes)
        self.model_class = ComponentFactory.wrap_dataparallel(
            self.model_class,
            self.config_manager.get_device(),
            self.config_manager.should_use_dataparallel()
        )
        return self
    
    def build_callbacks(self) -> 'ClassifierSetupBuilder':
        """Build callbacks."""
        if self.model_class is None:
            raise RuntimeError("Model must be built before callbacks")
        
        checkpoint_callback = ComponentFactory.create_checkpoint_callback(
            checkpoint_dir=self.cfg.trainer.checkpoint_dir,
            monitor='val_loss',
            mode='min',
            filename='best_classifier.pth',
            model=self.model_class
        )
        self.callbacks = [checkpoint_callback]
        return self
    
    def build(self) -> TrainingSetup:
        """Build classifier training setup."""
        if self.dataloaders is None or self.device is None or self.logger is None:
            raise RuntimeError("Build steps must be called before build()")
        
        trainer = TrainerFactory.create(
            cfg=self.cfg,
            dataloaders=self.dataloaders,
            device=self.device,
            logger=self.logger,
            callbacks=self.callbacks,
            normalization_stats=self.norm_stats,
            model_auto=self.model_auto,
            model_class=self.model_class,
            class_names=self.config_manager.get_class_names(),
            allow_resize=self.config_manager.allow_resize()
        )
        return TrainingSetup(trainer, self.logger)


class SetupBuilderDirector:
    """Director that orchestrates the building process."""
    
    BUILDER_REGISTRY = {
        "autoencoder": (AutoencoderSetupBuilder, "autoencoder"),
        "vae": (AutoencoderSetupBuilder, "autoencoder"),
        "adversarial": (AdversarialSetupBuilder, "adversarial"),
        "gan": (AdversarialSetupBuilder, "adversarial"),
        "classifier": (ClassifierSetupBuilder, "classifier"),
    }
    
    def __init__(self, cfg: DictConfig, config_manager: ConfigManager):
        self.cfg = cfg
        self.config_manager = config_manager
    
    def construct_setup(self, training_mode: str) -> TrainingSetup:
        """Construct complete training setup based on mode."""
        if training_mode not in self.BUILDER_REGISTRY:
            available_modes = list(self.BUILDER_REGISTRY.keys())
            raise ValueError(
                f"Unknown training mode: '{training_mode}'. "
                f"Available modes: {available_modes}"
            )
        
        builder_class, experiment_prefix = self.BUILDER_REGISTRY[training_mode]
        builder = builder_class(self.cfg, self.config_manager)
        
        # Execute build steps in order
        return (builder
                .build_dataloaders()
                .build_device()
                .build_logger(experiment_prefix)
                .build_model()
                .build_callbacks()
                .build())
