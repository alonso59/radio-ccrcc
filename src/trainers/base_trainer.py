"""Shared training lifecycle and epoch-loop utilities."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from tqdm import tqdm

from ..utils.checkpoints import (
    export_trainer_checkpoint,
    resume_trainer_if_available,
    sync_callback_resume_state,
)
from ..utils.monitoring import is_improved, parse_monitor_name, resolve_monitor_value

logger = logging.getLogger(__name__)

try:
    from pytorch_msssim import ssim
except ImportError:  # pragma: no cover - optional dependency
    def ssim(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
        mae = torch.mean(torch.abs(x - y))
        score = 1.0 - (mae / max(float(data_range), 1e-6))
        return torch.clamp(score, min=0.0, max=1.0)


class BaseTrainer(ABC):
    """Abstract trainer defining the shared training lifecycle."""

    def __init__(
        self,
        cfg: DictConfig,
        dataloaders: Dict[str, Any],
        device: torch.device,
        logger: Any,
        callbacks: Optional[List[Any]] = None,
        max_epochs: Optional[int] = None,
        normalization_stats: Optional[Dict[str, float]] = None,
    ):
        self.cfg = cfg
        self.dataloaders = dataloaders
        self.device = device
        self.logger = logger
        self.callbacks = callbacks or []
        self.max_epochs = int(max_epochs if max_epochs is not None else cfg.trainer.max_epochs)
        self.norm_stats = normalization_stats or {}
        self.global_step = 0
        self.start_epoch = 0
        self.resume_info: Dict[str, Any] = {"resume_mode": "fresh"}
        self.last_completed_epoch = -1
        self.monitor_metric = str(getattr(cfg.trainer, "monitor_metric", "val_loss"))
        self.monitor_mode = str(getattr(cfg.trainer, "monitor_mode", "min")).lower()
        if self.monitor_mode not in {"min", "max"}:
            raise ValueError(f"Unsupported trainer.monitor_mode: {self.monitor_mode!r}")

        self.best_monitor_value: Optional[float] = None
        self._monitor_warning_emitted = False
        self.metrics = {"PSNR": self._psnr_torch, "SSIM": self._ssim_torch}

    @abstractmethod
    def setup_models(self) -> None:
        """Initialize and configure models."""

    @abstractmethod
    def setup_optimizers(self) -> None:
        """Initialize optimizers."""

    @abstractmethod
    def setup_schedulers(self) -> None:
        """Initialize learning-rate schedulers."""

    @abstractmethod
    def setup_criteria(self) -> None:
        """Initialize loss functions."""

    @abstractmethod
    def train_step(self, epoch: int, batch: Any) -> Dict[str, Any]:
        """Execute one training step."""

    @abstractmethod
    def validation_step(self, epoch: int, batch: Any) -> Dict[str, Any]:
        """Execute one validation step."""

    @abstractmethod
    def _initialize_epoch_metrics(self) -> Dict[str, Any]:
        """Return the initial epoch metric container."""

    @abstractmethod
    def set_train_mode(self, train: bool) -> None:
        """Toggle models between training and evaluation mode."""

    def fit(self) -> None:
        """Execute the full training loop."""
        self.on_fit_start()

        if self.start_epoch >= self.max_epochs:
            logger.info(
                "Checkpoint epoch %d already satisfies max_epochs=%d. Skipping training loop.",
                self.start_epoch,
                self.max_epochs,
            )
            self.on_fit_end()
            return

        for epoch in range(self.start_epoch, self.max_epochs):
            start_time = time.time()
            train_metrics = self._run_epoch(self.dataloaders["train"], epoch, train=True)
            val_metrics = self._run_epoch(self.dataloaders["val"], epoch, train=False)

            self._log_epoch_metrics(epoch, train_metrics, val_metrics)
            self._check_best_model(train_metrics, val_metrics)
            self.on_epoch_end(epoch, train_metrics, val_metrics)
            self.step_schedulers(epoch)
            self._print_epoch_summary(epoch, train_metrics, val_metrics, time.time() - start_time)
            self.last_completed_epoch = epoch

        self.on_fit_end()

    def _run_epoch(self, dataloader: Any, epoch: int, train: bool = True) -> Dict[str, Any]:
        """Run one training or validation epoch."""
        mode = "train" if train else "val"
        self.set_train_mode(train)
        self.on_epoch_start(epoch, train)
        epoch_metrics = self._initialize_epoch_metrics()

        with torch.set_grad_enabled(train):
            for batch in tqdm(dataloader, desc=f"{mode.capitalize()} Epoch {epoch}", ncols=100, ascii=True):
                step_metrics = self.train_step(epoch, batch) if train else self.validation_step(epoch, batch)
                self._accumulate_epoch_metrics(epoch_metrics, step_metrics)
                self._log_step_metrics(mode, step_metrics)
                self.global_step += 1

        return self._average_epoch_metrics(epoch_metrics, len(dataloader))

    def step_schedulers(self, epoch: int) -> None:
        """Step learning-rate schedulers. Override when needed."""

    def on_fit_start(self) -> None:
        """Hook called before training starts."""
        resume_trainer_if_available(self)
        sync_callback_resume_state(self.callbacks, self.best_monitor_value)
        self._invoke_callbacks("on_fit_start", self)

    def on_fit_end(self) -> None:
        """Hook called after training ends."""
        self._invoke_callbacks("on_fit_end")

    def on_epoch_start(self, epoch: int, is_train: bool) -> None:
        """Hook called at the start of each epoch."""

    def on_epoch_end(self, epoch: int, train_metrics: Dict[str, Any], val_metrics: Dict[str, Any]) -> None:
        """Hook called at the end of each epoch."""
        self._invoke_callbacks("on_epoch_end", epoch, train_metrics, val_metrics)

    def export_checkpoint_state(
        self,
        epoch: int,
        checkpoint_kind: str,
        monitor_value: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Build a full trainer checkpoint payload."""
        return export_trainer_checkpoint(self, epoch, checkpoint_kind, monitor_value)

    def _checkpoint_modules(self) -> Dict[str, torch.nn.Module]:
        """Return checkpointed model modules."""
        model = getattr(self, "model", None)
        return {"model": model} if isinstance(model, torch.nn.Module) else {}

    def _checkpoint_optimizers(self) -> Dict[str, Any]:
        """Return checkpointed optimizers."""
        return self._stateful_attrs("optimizer")

    def _checkpoint_schedulers(self) -> Dict[str, Any]:
        """Return checkpointed schedulers."""
        return self._stateful_attrs("scheduler")

    def _checkpoint_scalers(self) -> Dict[str, Any]:
        """Return checkpointed AMP scalers."""
        return self._stateful_attrs("scaler")

    def _stateful_attrs(self, prefix: str) -> Dict[str, Any]:
        """Collect `state_dict`/`load_state_dict` objects from trainer attributes."""
        return {
            name: value
            for name, value in self.__dict__.items()
            if name.startswith(prefix) and hasattr(value, "state_dict") and hasattr(value, "load_state_dict")
        }

    def _accumulate_epoch_metrics(self, epoch_metrics: Dict[str, Any], step_metrics: Dict[str, Any]) -> None:
        """Merge one step's metrics into the epoch accumulator."""
        for key, value in step_metrics.items():
            target = epoch_metrics.get(key)
            if isinstance(target, list):
                target.append(value)
            elif key in epoch_metrics:
                epoch_metrics[key] += value
            else:
                epoch_metrics[key] = value

    def _average_epoch_metrics(self, epoch_metrics: Dict[str, Any], n_batches: int) -> Dict[str, Any]:
        """Average accumulated metrics over an epoch."""
        denominator = max(1, n_batches)
        averaged: Dict[str, Any] = {}
        for key, value in epoch_metrics.items():
            averaged[key] = value if isinstance(value, list) else value / denominator
        return averaged

    def _log_epoch_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, Any],
        val_metrics: Dict[str, Any],
    ) -> None:
        """Log epoch-level scalar metrics."""
        scalar_train = self._scalar_metrics(train_metrics)
        scalar_val = self._scalar_metrics(val_metrics)
        for key, train_value in scalar_train.items():
            val_value = scalar_val.get(key)
            if val_value is not None:
                self.logger.log_scalars(key.capitalize(), {"train": train_value, "val": val_value}, epoch)

    def _log_step_metrics(self, mode: str, metrics: Dict[str, Any]) -> None:
        """Log scalar step metrics at a configurable interval."""
        interval = int(getattr(self.cfg.trainer, "log_step_interval", 50))
        if self.global_step % interval != 0:
            return
        for key, value in self._scalar_metrics(metrics).items():
            self.logger.add_scalar(f"step_{mode}_{key}", value, self.global_step)

    def _print_epoch_summary(
        self,
        epoch: int,
        train_metrics: Dict[str, Any],
        val_metrics: Dict[str, Any],
        epoch_time: float,
    ) -> None:
        """Print a compact epoch summary."""
        logger.info("Epoch %d | Time: %.2fs", epoch, epoch_time)
        logger.info("Train: %s", self._format_metrics(train_metrics))
        logger.info("Val:   %s", self._format_metrics(val_metrics))

    def _format_metrics(self, metrics: Dict[str, Any], max_items: int = 5) -> str:
        """Format scalar metrics for logging."""
        items = list(self._scalar_metrics(metrics).items())[:max_items]
        return ", ".join(f"{key}={value:.4f}" for key, value in items)

    def _scalar_metrics(self, metrics: Mapping[str, Any]) -> Dict[str, float]:
        """Filter a metric mapping down to scalar values."""
        scalars: Dict[str, float] = {}
        for key, value in metrics.items():
            scalar = self._coerce_scalar(value)
            if scalar is not None:
                scalars[key] = scalar
        return scalars

    def _coerce_scalar(self, value: Any) -> Optional[float]:
        """Convert scalar-like values to float and ignore structured metrics."""
        if isinstance(value, (int, float)):
            return float(value)
        if torch.is_tensor(value) and value.ndim == 0:
            return float(value.item())
        if hasattr(value, "item"):
            try:
                return float(value.item())
            except (TypeError, ValueError):
                return None
        return None

    def _check_best_model(self, train_metrics: Dict[str, Any], val_metrics: Dict[str, Any]) -> None:
        """Update tracked best metric when the monitored value improves."""
        current_value = resolve_monitor_value(self.monitor_metric, train_metrics, val_metrics)
        if current_value is None:
            if not self._monitor_warning_emitted:
                split, _ = parse_monitor_name(self.monitor_metric)
                metrics = train_metrics if split == "train" else val_metrics
                logger.warning(
                    "Configured monitor_metric '%s' did not resolve to a scalar metric. Available %s keys: %s",
                    self.monitor_metric,
                    split,
                    sorted(metrics.keys()) if metrics is not None else [],
                )
                self._monitor_warning_emitted = True
            return

        if is_improved(current_value, self.best_monitor_value, self.monitor_mode):
            self.best_monitor_value = current_value
            logger.info("New best %s (%s): %.4f", self.monitor_metric, self.monitor_mode, current_value)

    def _invoke_callbacks(self, method_name: str, *args: Any) -> None:
        """Invoke a lifecycle hook on all registered callbacks."""
        for callback in self.callbacks:
            method = getattr(callback, method_name, None)
            if callable(method):
                method(*args)

    def _psnr_torch(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(img1, img2)
        if mse == 0:
            return torch.tensor(float("inf"))
        data_range = 2.0
        return 10 * torch.log10(data_range**2 / mse)

    def _ssim_torch(self, x: torch.Tensor, x_hat: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        return ssim(x, x_hat, data_range=2.0)

    def _compute_quality_metrics(self, x_recon: torch.Tensor, x: torch.Tensor) -> Dict[str, float]:
        """Compute reconstruction quality metrics."""
        metrics: Dict[str, float] = {}
        with torch.no_grad():
            x_recon = x_recon.float()
            x = x.float()
            for metric_name, metric_fn in self.metrics.items():
                metric_value = metric_fn(x_recon, x)
                metrics[metric_name] = float(metric_value.item() if hasattr(metric_value, "item") else metric_value)
            if "PSNR" in metrics:
                metrics["PSNR_count"] = 1
        return metrics
