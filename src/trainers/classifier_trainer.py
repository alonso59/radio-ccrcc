"""Classifier trainer on frozen autoencoder representations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig

from .base_trainer import BaseTrainer
from ..utils.scheduler import get_scheduler

logger = logging.getLogger(__name__)

ARRAY_METRIC_KEYS = ("predictions", "labels", "probabilities")


@dataclass(frozen=True)
class ClassifierSettings:
    """Normalized classifier trainer settings."""

    num_classes: int
    metric_log_interval: int
    weight_decay: float
    optimizer_name: str
    learning_rate: float
    class_weights: Optional[Sequence[float]]

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> "ClassifierSettings":
        return cls(
            num_classes=int(getattr(cfg.model, "num_classes", 2)),
            metric_log_interval=int(getattr(cfg.trainer, "metric_log_interval", 5)),
            weight_decay=float(getattr(cfg.optimizer, "weight_decay", 0.0)),
            optimizer_name=str(cfg.optimizer.name),
            learning_rate=float(cfg.optimizer.lr),
            class_weights=getattr(cfg, "class_weights", None),
        )


class ClassifierTrainer(BaseTrainer):
    """Trainer for classification using a frozen autoencoder encoder."""

    def __init__(
        self,
        cfg: DictConfig,
        model_auto: torch.nn.Module,
        model_class: torch.nn.Module,
        dataloaders: Dict[str, Any],
        device: torch.device,
        logger: Any,
        callbacks: Optional[list] = None,
        max_epochs: Optional[int] = None,
        class_names: Optional[Sequence[str]] = None,
        allow_resize: bool = False,
    ):
        super().__init__(cfg, dataloaders, device, logger, callbacks, max_epochs)
        self.model_auto = model_auto
        self.model_class = model_class
        self.allow_resize = allow_resize
        self.settings = ClassifierSettings.from_cfg(cfg)
        self.class_names = list(class_names) if class_names else [f"Class_{i}" for i in range(self.settings.num_classes)]
        if len(self.class_names) != self.settings.num_classes:
            raise ValueError(f"Number of class names ({len(self.class_names)}) != num_classes ({self.settings.num_classes})")

        self.setup_models()
        self.setup_optimizers()
        self.setup_schedulers()
        self.setup_criteria()

    def setup_models(self) -> None:
        self.model_auto = self.model_auto.to(self.device)
        self.model_class = self.model_class.to(self.device)
        for parameter in self.model_auto.parameters():
            parameter.requires_grad = False
        self.model_auto.eval()

    def setup_optimizers(self) -> None:
        optimizer_cls = getattr(optim, self.settings.optimizer_name)
        self.optimizer = optimizer_cls(
            self.model_class.parameters(),
            lr=self.settings.learning_rate,
            weight_decay=self.settings.weight_decay,
        )

    def setup_schedulers(self) -> None:
        scheduler_cfg = self.cfg.get("scheduler")
        self.scheduler = get_scheduler(self.optimizer, scheduler_cfg) if scheduler_cfg is not None else None

    def setup_criteria(self) -> None:
        class_weights = self.settings.class_weights
        weight_tensor = None
        if class_weights is not None:
            weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32, device=self.device)
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)

    def set_train_mode(self, train: bool) -> None:
        self.model_class.train(train)
        self.model_auto.eval()

    def train_step(self, epoch: int, batch: Any) -> Dict[str, Any]:
        inputs, labels = self._get_inputs_and_labels(batch)
        with torch.no_grad():
            mu_volume = self._encode_mu_volume(inputs)

        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model_class(mu_volume)
        self._validate_logits(logits)
        loss = self.criterion(logits, labels)
        loss.backward()
        self.optimizer.step()
        return self._classification_metrics(logits, labels, loss)

    def validation_step(self, epoch: int, batch: Any) -> Dict[str, Any]:
        inputs, labels = self._get_inputs_and_labels(batch)
        with torch.no_grad():
            mu_volume = self._encode_mu_volume(inputs)
            logits = self.model_class(mu_volume)
            self._validate_logits(logits)
            loss = self.criterion(logits, labels)
            return self._classification_metrics(logits, labels, loss)

    def step_schedulers(self, epoch: int) -> None:
        if self.scheduler is not None:
            self.scheduler.step()

    def on_epoch_end(self, epoch: int, train_metrics: Dict[str, Any], val_metrics: Dict[str, Any]) -> None:
        labels = val_metrics.get("labels")
        predictions = val_metrics.get("predictions")
        probabilities = val_metrics.get("probabilities")
        if labels is not None and predictions is not None and labels.size > 0:
            precision, recall, f1 = self._precision_recall_f1(predictions, labels)
            self.logger.add_scalar("Precision_macro/val", precision, epoch)
            self.logger.add_scalar("Recall_macro/val", recall, epoch)
            self.logger.add_scalar("F1_macro/val", f1, epoch)

            if epoch % self.settings.metric_log_interval == 0:
                self._log_classifier_figures(epoch, labels, predictions, probabilities)

        super().on_epoch_end(epoch, train_metrics, val_metrics)

    def _checkpoint_modules(self) -> Dict[str, torch.nn.Module]:
        return {"model_auto": self.model_auto, "model_class": self.model_class}

    def _checkpoint_optimizers(self) -> Dict[str, Any]:
        return {"optimizer": self.optimizer}

    def _checkpoint_schedulers(self) -> Dict[str, Any]:
        return {"scheduler": self.scheduler} if self.scheduler is not None else {}

    def _initialize_epoch_metrics(self) -> Dict[str, Any]:
        return {"loss": 0.0, "accuracy": 0.0, "predictions": [], "labels": [], "probabilities": []}

    def _average_epoch_metrics(self, epoch_metrics: Dict[str, Any], n_batches: int) -> Dict[str, Any]:
        averaged: Dict[str, Any] = {}
        denominator = max(1, n_batches)
        for key, value in epoch_metrics.items():
            if key in ARRAY_METRIC_KEYS:
                averaged[key] = self._concatenate_metric_list(value)
            else:
                averaged[key] = value / denominator
        return averaged

    def _classification_metrics(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss: torch.Tensor,
    ) -> Dict[str, Any]:
        probabilities = F.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)
        accuracy = (predictions == labels).float().mean().item()
        return {
            "loss": loss.item(),
            "accuracy": accuracy,
            "predictions": predictions.cpu(),
            "labels": labels.cpu(),
            "probabilities": probabilities.cpu(),
        }

    def _concatenate_metric_list(self, values: list[Any]) -> np.ndarray:
        if not values:
            return np.array([])
        if isinstance(values[0], torch.Tensor):
            return torch.cat(values, dim=0).numpy()
        return np.concatenate(values, axis=0)

    def _validate_logits(self, logits: torch.Tensor) -> None:
        if logits.ndim != 2:
            raise RuntimeError(f"Classifier must output [B,C]; got {tuple(logits.shape)}")

    @torch.no_grad()
    def _encode_mu_volume(self, inputs: torch.Tensor) -> torch.Tensor:
        _, mu, _ = self.model_auto(inputs)
        if mu.ndim == 4:
            return mu.unsqueeze(2)
        if mu.ndim != 5:
            raise RuntimeError(f"Unexpected mu shape: {tuple(mu.shape)}")
        return mu

    def _get_inputs_and_labels(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = batch["ct"]["data"].to(self.device)
        labels = batch["label"]
        if isinstance(labels, (list, tuple)) and labels and isinstance(labels[0], str):
            labels = [self.class_names.index(label) for label in labels]
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long, device=self.device)
        else:
            labels = labels.to(self.device)
        return inputs, labels

    def _precision_recall_f1(self, predictions: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
        precisions, recalls, f1_scores = [], [], []
        for class_index in range(self.settings.num_classes):
            true_positive = np.sum((labels == class_index) & (predictions == class_index))
            false_positive = np.sum((labels != class_index) & (predictions == class_index))
            false_negative = np.sum((labels == class_index) & (predictions != class_index))

            precision = true_positive / (true_positive + false_positive + 1e-12)
            recall = true_positive / (true_positive + false_negative + 1e-12)
            f1 = 2 * precision * recall / (precision + recall + 1e-12)

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1_scores))

    def _log_classifier_figures(
        self,
        epoch: int,
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: Optional[np.ndarray],
    ) -> None:
        try:
            self.logger.log_confusion_matrix(labels, predictions, self.class_names, step=epoch, tag="val")
        except Exception as exc:
            logger.warning("Confusion matrix logging failed: %s", exc)

        if probabilities is None:
            return

        try:
            self.logger.log_roc_auc(labels, probabilities, step=epoch, tag="val", class_names=self.class_names)
        except Exception as exc:
            logger.warning("ROC curve logging failed: %s", exc)
