"""Run-artifact management for history, plots, aliases, and metadata."""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

from .metric_plots import save_metric_plots

logger = logging.getLogger(__name__)

try:
    from hydra.core.hydra_config import HydraConfig
except Exception:  # pragma: no cover
    HydraConfig = None  # type: ignore


class ResultsArtifactsManager:
    """Persist canonical run artifacts under the Hydra results directory."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.model: Optional[Any] = None
        self.logger: Optional[Any] = None
        self.trainer: Optional[Any] = None
        self.run_dir = self._resolve_run_dir()
        self._history_seeded = False

        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "tb").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "progress").mkdir(parents=True, exist_ok=True)
        self.write_debug_json()

    def set_model(self, model: Any) -> None:
        self.model = model

    def set_logger(self, logger_instance: Any) -> None:
        self.logger = logger_instance

    def set_trainer(self, trainer: Any) -> None:
        self.trainer = trainer

    def on_fit_start(self, trainer: Optional[Any] = None) -> None:
        if trainer is not None:
            self.trainer = trainer
        self.seed_history_from_resume()
        self.flush_history_outputs()

    def on_epoch_end(self, epoch: int, train_metrics: Any, val_metrics: Any) -> None:
        self.flush_history_outputs()

    def on_fit_end(self) -> None:
        self.flush_history_outputs()
        self.write_checkpoint_final_alias()
        self.create_latest_symlink()
        self.write_debug_json()

    def flush_history_outputs(self) -> None:
        """Persist history files and refresh publication-style plots."""
        if self.logger is None:
            return

        self.logger.flush()
        recorder = self.logger.get_scalar_recorder()
        if recorder.is_empty():
            return

        recorder.save_json(self.run_dir / "history.json", metadata=self.history_metadata())
        recorder.save_csv(self.run_dir / "history.csv")
        save_metric_plots(recorder.as_series_dict(), self.run_dir / "progress")

    def seed_history_from_resume(self) -> None:
        """Seed the current recorder from a previous run when resuming."""
        if self._history_seeded or self.logger is None or self.trainer is None:
            return

        resume_info = getattr(self.trainer, "resume_info", {}) or {}
        source_run_dir = resume_info.get("source_run_dir")
        if not source_run_dir:
            self._history_seeded = True
            return

        source_dir = Path(str(source_run_dir))
        recorder = self.logger.get_scalar_recorder()
        history_path = source_dir / "history.json"
        if history_path.exists() and recorder.load_json(history_path):
            logger.info("Seeded history recorder from %s", history_path)
        elif resume_info.get("resume_mode") == "warm_start":
            tb_root = source_dir / "tb"
            if tb_root.exists() and recorder.import_tensorboard(tb_root):
                logger.info("Seeded history recorder from TensorBoard events in %s", tb_root)

        self._history_seeded = True

    def history_metadata(self) -> dict[str, Any]:
        """Build metadata stored alongside canonical scalar history."""
        resume_info = getattr(self.trainer, "resume_info", {}) if self.trainer is not None else {}
        return {
            "run_dir": str(self.run_dir),
            "created_at": datetime.now().isoformat(),
            "resume": resume_info,
            "trainer": {
                "training_mode": self.cfg.trainer.get("training_mode"),
                "monitor_metric": self.cfg.trainer.get("monitor_metric"),
                "monitor_mode": self.cfg.trainer.get("monitor_mode"),
                "max_epochs": self.cfg.trainer.get("max_epochs"),
            },
            "dataset": {
                "dataset_id": self.cfg.dataset.get("dataset_id"),
                "fold": self.cfg.dataset.get("fold"),
            },
        }

    def write_checkpoint_final_alias(self) -> None:
        """Keep a compatibility alias for the final last-checkpoint."""
        source = self.run_dir / "checkpoint_last.pth"
        if not source.exists():
            source = self.run_dir / "checkpoints" / "checkpoint_last.pth"
        if source.exists():
            shutil.copy2(source, self.run_dir / "checkpoint_final.pth")

    def create_latest_symlink(self) -> None:
        """Create or update the fold-level `latest` symlink."""
        try:
            fold_dir = self.run_dir.parent.parent
            latest_link = fold_dir / "latest"
            if latest_link.is_symlink():
                latest_link.unlink()
            elif latest_link.exists():
                if latest_link.is_dir():
                    shutil.rmtree(latest_link)
                else:
                    latest_link.unlink()

            latest_link.symlink_to(self.run_dir.relative_to(fold_dir))
        except Exception as exc:
            logger.warning("Could not create 'latest' symlink: %s", exc)

    def write_debug_json(self) -> None:
        """Write a compact run-level debug snapshot."""
        payload: dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "cwd": os.getcwd(),
            "run_dir": str(self.run_dir),
            "results": {
                "root": self.cfg.get("results", {}).get("root") if isinstance(self.cfg.get("results"), dict) else None,
                "experiment": self.cfg.get("results", {}).get("experiment") if isinstance(self.cfg.get("results"), dict) else None,
            },
            "dataset": {
                "dataset_id": self.cfg.dataset.get("dataset_id"),
                "fold": self.cfg.dataset.get("fold"),
            },
            "trainer": {
                "training_mode": self.cfg.trainer.get("training_mode"),
                "max_epochs": self.cfg.trainer.get("max_epochs"),
                "checkpoint_dir": self.cfg.trainer.get("checkpoint_dir"),
                "monitor_metric": self.cfg.trainer.get("monitor_metric"),
                "monitor_mode": self.cfg.trainer.get("monitor_mode"),
                "resume": getattr(self.trainer, "resume_info", None),
            },
            "config": OmegaConf.to_container(self.cfg, resolve=True),
        }
        (self.run_dir / "debug.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _resolve_run_dir(self) -> Path:
        if HydraConfig is not None:
            try:
                return Path(HydraConfig.get().runtime.output_dir)
            except Exception:
                pass
        return Path(os.getcwd())
