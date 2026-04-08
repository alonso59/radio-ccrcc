"""Builders for assembling training setups."""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig

from .component_factory import ComponentFactory
from .config_manager import ConfigManager
from ..trainers.trainer_factory import TrainerFactory
from ..utils.logger import TensorBoardLogger
from ..utils.results_callback import ResultsFolderCallback
from ..utils.training_modes import normalize_training_mode


class TrainingSetup:
    """Container holding the trainer and logger."""

    def __init__(self, trainer: Any, logger: TensorBoardLogger):
        self.trainer = trainer
        self.logger = logger


class BaseSetupBuilder:
    """Shared builder utilities for training setup."""

    def __init__(self, cfg: DictConfig, config_manager: ConfigManager):
        self.cfg = cfg
        self.config_manager = config_manager
        self.logger: TensorBoardLogger | None = None
        self.device: torch.device | None = None
        self.dataloaders = None
        self.norm_stats = None
        self.callbacks: list[Any] = []

    def build_dataloaders(self) -> "BaseSetupBuilder":
        self.dataloaders, self.norm_stats = ComponentFactory.create_dataloaders(self.cfg.dataset)
        return self

    def build_device(self) -> "BaseSetupBuilder":
        self.device = torch.device(self.config_manager.get_device())
        return self

    def build_logger(self, experiment_prefix: str) -> "BaseSetupBuilder":
        experiment_name = self.config_manager.get_experiment_name(experiment_prefix)
        self.logger = ComponentFactory.create_logger(experiment_name)
        return self

    def build(self) -> TrainingSetup:
        raise NotImplementedError

    def _create_standard_callbacks(self, model: torch.nn.Module, filename: str) -> list[Any]:
        checkpoint = ComponentFactory.create_checkpoint_callback(
            checkpoint_dir=self.cfg.trainer.checkpoint_dir,
            monitor=str(getattr(self.cfg.trainer, "monitor_metric", "val_loss")),
            mode=str(getattr(self.cfg.trainer, "monitor_mode", "min")),
            filename=filename,
            model=model,
        )
        results = ResultsFolderCallback(self.cfg)
        return [checkpoint, results]

    def _attach_callback_context(self, trainer: Any, model: torch.nn.Module | None = None) -> None:
        for callback in self.callbacks:
            if model is not None and hasattr(callback, "set_model"):
                callback.set_model(model)
            if self.logger is not None and hasattr(callback, "set_logger"):
                callback.set_logger(self.logger)
            if hasattr(callback, "set_trainer"):
                callback.set_trainer(trainer)

    def _validate_common_state(self) -> None:
        if self.dataloaders is None or self.device is None or self.logger is None:
            raise RuntimeError("Build steps must be called before build()")


class AutoencoderSetupBuilder(BaseSetupBuilder):
    """Builder for all representation-learning modes."""

    def __init__(self, cfg: DictConfig, config_manager: ConfigManager):
        super().__init__(cfg, config_manager)
        self.model: torch.nn.Module | None = None

    def build_model(self) -> "AutoencoderSetupBuilder":
        self.model = ComponentFactory.create_autoencoder(self.cfg)
        self.model = ComponentFactory.wrap_dataparallel(
            self.model,
            self.config_manager.get_device(),
            self.config_manager.should_use_dataparallel(),
        )
        return self

    def build_callbacks(self) -> "AutoencoderSetupBuilder":
        if self.model is None:
            raise RuntimeError("Model must be built before callbacks")
        mode_name = str(self.cfg.trainer.training_mode).lower()
        self.callbacks = self._create_standard_callbacks(self.model, f"best_{mode_name}.pth")
        return self

    def build(self) -> TrainingSetup:
        self._validate_common_state()
        if self.model is None:
            raise RuntimeError("Model must be built before build()")

        trainer = TrainerFactory.create(
            cfg=self.cfg,
            dataloaders=self.dataloaders,
            device=self.device,
            logger=self.logger,
            callbacks=self.callbacks,
            normalization_stats=self.norm_stats,
            model=self.model,
        )
        self._attach_callback_context(trainer, model=self.model)
        return TrainingSetup(trainer, self.logger)


class ClassifierSetupBuilder(BaseSetupBuilder):
    """Builder for classifier training."""

    def __init__(self, cfg: DictConfig, config_manager: ConfigManager):
        super().__init__(cfg, config_manager)
        self.model_auto: torch.nn.Module | None = None
        self.model_class: torch.nn.Module | None = None

    def build_model(self) -> "ClassifierSetupBuilder":
        self.model_auto = ComponentFactory.create_autoencoder(self.cfg, show_summary=False)
        self.model_auto = ComponentFactory.wrap_dataparallel(
            self.model_auto,
            self.config_manager.get_device(),
            self.config_manager.should_use_dataparallel(),
        )
        self.model_class = ComponentFactory.create_classifier(self.config_manager.get_num_classes())
        self.model_class = ComponentFactory.wrap_dataparallel(
            self.model_class,
            self.config_manager.get_device(),
            self.config_manager.should_use_dataparallel(),
        )
        return self

    def build_callbacks(self) -> "ClassifierSetupBuilder":
        if self.model_class is None:
            raise RuntimeError("Model must be built before callbacks")
        self.callbacks = self._create_standard_callbacks(self.model_class, "best_classifier.pth")
        return self

    def build(self) -> TrainingSetup:
        self._validate_common_state()
        if self.model_auto is None or self.model_class is None:
            raise RuntimeError("Models must be built before build()")

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
            allow_resize=self.config_manager.allow_resize(),
        )
        self._attach_callback_context(trainer, model=self.model_class)
        return TrainingSetup(trainer, self.logger)


class SetupBuilderDirector:
    """Orchestrate the setup build sequence."""

    BUILDER_REGISTRY = {
        "autoencoder": (AutoencoderSetupBuilder, "autoencoder"),
        "classifier": (ClassifierSetupBuilder, "classifier"),
    }

    def __init__(self, cfg: DictConfig, config_manager: ConfigManager):
        self.cfg = cfg
        self.config_manager = config_manager

    def construct_setup(self, training_mode: str) -> TrainingSetup:
        canonical_mode = normalize_training_mode(training_mode)
        if canonical_mode not in self.BUILDER_REGISTRY:
            available = sorted(self.BUILDER_REGISTRY)
            raise ValueError(f"Unknown training mode: {training_mode!r}. Available modes: {available}")

        builder_class, experiment_prefix = self.BUILDER_REGISTRY[canonical_mode]
        builder = builder_class(self.cfg, self.config_manager)
        return (
            builder
            .build_dataloaders()
            .build_device()
            .build_logger(experiment_prefix)
            .build_model()
            .build_callbacks()
            .build()
        )
