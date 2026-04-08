"""Training callbacks."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .monitoring import is_improved, parse_monitor_name, resolve_monitor_value
from .checkpoints import export_trainer_checkpoint

logger = logging.getLogger(__name__)


class ModelCheckpoint:
    """Callback to persist full trainer checkpoints for resume and evaluation."""

    def __init__(
        self,
        dirpath: str,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        filename: str = "best_model.pth",
    ):
        self.dirpath = dirpath
        self.monitor = str(monitor)
        self.mode = str(mode).lower()
        self.save_best_only = save_best_only
        self.filename = filename
        self.best: Optional[float] = None
        self.trainer: Optional[Any] = None
        self.model: Optional[torch.nn.Module] = None
        self._monitor_warning_emitted = False
        if self.mode not in {"min", "max"}:
            raise ValueError(f"Unsupported checkpoint mode: {self.mode!r}. Expected 'min' or 'max'.")
        self.run_dir = Path(os.getcwd()).resolve()
        self.checkpoint_dir = Path(self.dirpath)
        if not self.checkpoint_dir.is_absolute():
            self.checkpoint_dir = (self.run_dir / self.checkpoint_dir).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(
        self,
        epoch: int,
        train_metrics: Dict[str, Any],
        val_metrics: Dict[str, Any],
    ) -> None:
        """Save the latest checkpoint every epoch and the best checkpoint when improved."""
        monitor_value = resolve_monitor_value(self.monitor, train_metrics, val_metrics)
        if monitor_value is None:
            if not self._monitor_warning_emitted:
                split, _ = parse_monitor_name(self.monitor)
                metrics = train_metrics if split == "train" else val_metrics
                available = sorted(metrics.keys()) if metrics is not None else []
                logger.warning(
                    "Checkpoint monitor '%s' did not resolve to a scalar metric. Available %s keys: %s",
                    self.monitor,
                    split,
                    available,
                )
                self._monitor_warning_emitted = True
            self._save_checkpoint(epoch, checkpoint_kind="last", monitor_value=None)
            return

        self._save_checkpoint(epoch, checkpoint_kind="last", monitor_value=float(monitor_value))
        improved = is_improved(monitor_value, self.best, self.mode)
        if improved:
            self.best = float(monitor_value)
            self._save_checkpoint(epoch, checkpoint_kind="best", monitor_value=float(monitor_value))

    def _save_checkpoint(self, epoch: int, checkpoint_kind: str, monitor_value: Optional[float]) -> None:
        """Persist a checkpoint payload and update compatibility aliases."""
        payload = self._build_payload(epoch, checkpoint_kind, monitor_value)
        canonical_name = f"checkpoint_{checkpoint_kind}.pth"
        targets = [
            self.checkpoint_dir / canonical_name,
            self.run_dir / canonical_name,
        ]

        if checkpoint_kind == "best" and self.filename and self.filename != canonical_name:
            targets.append(self.checkpoint_dir / self.filename)

        for target in targets:
            self._write_payload_atomic(target, payload)

        logger.info("Saved %s checkpoint for epoch %d to %s", checkpoint_kind, epoch, targets[0])

    def _build_payload(self, epoch: int, checkpoint_kind: str, monitor_value: Optional[float]) -> Dict[str, Any]:
        """Build a full trainer checkpoint or fall back to weights-only export."""
        if self.trainer is not None:
            return export_trainer_checkpoint(
                self.trainer,
                epoch=epoch,
                checkpoint_kind=checkpoint_kind,
                monitor_value=monitor_value,
            )

        if self.model is None:
            raise RuntimeError("ModelCheckpoint requires either a trainer or a model")

        model_to_save = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        return {
            "epoch": epoch,
            "model_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in model_to_save.state_dict().items()
            },
            "best": self.best,
            "best_monitor_value": self.best,
            "monitor_metric": self.monitor,
            "monitor_mode": self.mode,
            "resume_metadata": {
                "checkpoint_kind": checkpoint_kind,
                "monitor_value": monitor_value,
            },
        }

    def _write_payload_atomic(self, path: Path, payload: Dict[str, Any]) -> None:
        """Write a checkpoint atomically to avoid partial files."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def set_model(self, model: torch.nn.Module) -> None:
        """Attach a model for legacy fallback checkpoints."""
        self.model = model

    def set_trainer(self, trainer: Any) -> None:
        """Attach the owning trainer for full-state checkpointing."""
        self.trainer = trainer
        if hasattr(trainer, "best_monitor_value"):
            self.best = trainer.best_monitor_value
