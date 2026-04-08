"""Lazy exports for training modules."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["BaseTrainer", "AutoencoderTrainer", "ClassifierTrainer", "TrainerFactory"]

_EXPORTS = {
    "BaseTrainer": ".base_trainer",
    "AutoencoderTrainer": ".trainer",
    "ClassifierTrainer": ".classifier_trainer",
    "TrainerFactory": ".trainer_factory",
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name], __name__)
    return getattr(module, name)
