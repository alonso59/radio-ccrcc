"""Factory for creating trainers from the normalized training mode."""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch
from omegaconf import DictConfig

from .base_trainer import BaseTrainer
from .classifier_trainer import ClassifierTrainer
from .trainer import AutoencoderTrainer
from ..utils.training_modes import normalize_training_mode, supported_training_modes

logger = logging.getLogger(__name__)


class TrainerFactory:
    """Create the correct trainer for the configured training mode."""

    TRAINER_REGISTRY = {
        "autoencoder": AutoencoderTrainer,
        "classifier": ClassifierTrainer,
    }

    @classmethod
    def create(
        cls,
        cfg: DictConfig,
        dataloaders: Dict[str, Any],
        device: torch.device,
        logger: Any,
        **kwargs: Any,
    ) -> BaseTrainer:
        training_mode = normalize_training_mode(cfg.trainer.training_mode)
        trainer_class = cls.TRAINER_REGISTRY.get(training_mode)
        if trainer_class is None:
            raise ValueError(
                f"Unknown training mode: {training_mode!r}. Available modes: {supported_training_modes()}"
            )

        common_args = {
            "cfg": cfg,
            "dataloaders": dataloaders,
            "device": device,
            "logger": logger,
            "callbacks": kwargs.get("callbacks"),
            "max_epochs": kwargs.get("max_epochs"),
            "normalization_stats": kwargs.get("normalization_stats"),
        }
        return cls._create_autoencoder_trainer(common_args, kwargs) if training_mode == "autoencoder" else cls._create_classifier_trainer(common_args, kwargs)

    @staticmethod
    def _create_autoencoder_trainer(common_args: Dict[str, Any], kwargs: Dict[str, Any]) -> AutoencoderTrainer:
        if "model" not in kwargs:
            raise ValueError("'model' must be provided for autoencoder training")
        return AutoencoderTrainer(model=kwargs["model"], **common_args)

    @staticmethod
    def _create_classifier_trainer(common_args: Dict[str, Any], kwargs: Dict[str, Any]) -> ClassifierTrainer:
        if "model_auto" not in kwargs:
            raise ValueError("'model_auto' must be provided for classifier training")
        if "model_class" not in kwargs:
            raise ValueError("'model_class' must be provided for classifier training")
        return ClassifierTrainer(
            model_auto=kwargs["model_auto"],
            model_class=kwargs["model_class"],
            class_names=kwargs.get("class_names"),
            allow_resize=kwargs.get("allow_resize", False),
            **common_args,
        )

    @classmethod
    def register_trainer(cls, name: str, trainer_class: type[BaseTrainer]) -> None:
        if not issubclass(trainer_class, BaseTrainer):
            raise TypeError(f"{trainer_class} must inherit from BaseTrainer")
        cls.TRAINER_REGISTRY[normalize_training_mode(name)] = trainer_class
        logger.info("Registered trainer '%s': %s", name, trainer_class.__name__)

    @classmethod
    def get_available_modes(cls) -> list[str]:
        return supported_training_modes()
