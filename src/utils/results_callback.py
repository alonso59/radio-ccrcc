"""Training callback for maintaining run artifacts in the results folder."""

from __future__ import annotations

from typing import Any, Optional

from omegaconf import DictConfig

from .results_artifacts import ResultsArtifactsManager


class ResultsFolderCallback:
    """Thin callback adapter around the results artifact manager."""

    def __init__(self, cfg: DictConfig):
        self.artifacts = ResultsArtifactsManager(cfg)

    def set_model(self, model: Any) -> None:
        self.artifacts.set_model(model)

    def set_logger(self, logger_instance: Any) -> None:
        self.artifacts.set_logger(logger_instance)

    def set_trainer(self, trainer: Any) -> None:
        self.artifacts.set_trainer(trainer)

    def on_fit_start(self, trainer: Optional[Any] = None) -> None:
        self.artifacts.on_fit_start(trainer)

    def on_epoch_end(self, epoch: int, train_metrics: Any, val_metrics: Any) -> None:
        self.artifacts.on_epoch_end(epoch, train_metrics, val_metrics)

    def on_fit_end(self) -> None:
        self.artifacts.on_fit_end()
