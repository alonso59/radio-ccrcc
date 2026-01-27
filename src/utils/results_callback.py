from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
from omegaconf import DictConfig, OmegaConf

try:
    from hydra.core.hydra_config import HydraConfig
except Exception:  # pragma: no cover
    HydraConfig = None  # type: ignore


@dataclass(frozen=True)
class _BestCheckpointSpec:
    filename: str


_MODE_TO_BEST = {
    "vae": _BestCheckpointSpec(filename="best_autoencoder.pth"),
    "gan": _BestCheckpointSpec(filename="best_adversarial.pth"),
    "classifier": _BestCheckpointSpec(filename="best_classifier.pth"),
}


class ResultsFolderCallback:
    """Writes/keeps a standardized results folder *during* training.

    Assumes Hydra is configured so the current run directory is under `results/`.

    Creates/maintains in the run directory:
      - checkpoint_best.pth (synced from trainer's best checkpoint)
      - checkpoint_final.pth (written at fit end)
      - debug.json
      - progress.png (empty placeholder; TODO)
      - tb/ (created here; populated by TensorBoard logger)
      - .hydra/ (created by Hydra automatically)

    This is intentionally lightweight and avoids refactoring trainer logic.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.model: Optional[torch.nn.Module] = None
        self.run_dir = self._resolve_run_dir()
        self._last_best_mtime: Optional[float] = None

        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "tb").mkdir(parents=True, exist_ok=True)

        progress_path = self.run_dir / "progress.png"
        if not progress_path.exists():
            progress_path.write_bytes(b"")

        self._write_debug_json()

    def set_model(self, model: torch.nn.Module) -> None:
        self.model = model

    def on_epoch_end(self, epoch: int, train_loss: float, val_loss: float) -> None:
        best_src = self._best_checkpoint_source_path()
        if best_src is None or not best_src.exists():
            return

        mtime = best_src.stat().st_mtime
        if self._last_best_mtime is None or mtime > self._last_best_mtime:
            shutil.copy2(best_src, self.run_dir / "checkpoint_best.pth")
            self._last_best_mtime = mtime

    def on_fit_end(self) -> None:
        if self.model is None:
            return

        model_to_save = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        final_path = self.run_dir / "checkpoint_final.pth"

        torch.save(
            {
                "epoch": int(self.cfg.trainer.max_epochs) - 1,
                "model_state_dict": model_to_save.state_dict(),
                "saved_at": datetime.now().isoformat(),
            },
            final_path,
        )
        
        # Create/update 'latest' symlink for evaluation discovery
        self._create_latest_symlink()

    def _resolve_run_dir(self) -> Path:
        if HydraConfig is not None:
            try:
                return Path(HydraConfig.get().runtime.output_dir)
            except Exception:
                pass
        return Path(os.getcwd())

    def _best_checkpoint_source_path(self) -> Optional[Path]:
        mode = str(self.cfg.trainer.training_mode)
        spec = _MODE_TO_BEST.get(mode)
        if spec is None:
            return None

        checkpoint_dir = Path(str(self.cfg.trainer.checkpoint_dir))
        checkpoint_dir = (self.run_dir / checkpoint_dir).resolve()
        return checkpoint_dir / spec.filename

    def _create_latest_symlink(self) -> None:
        """Create/update 'latest' symlink at fold level pointing to this run.
        
        Structure: results/experiment/fold_N/latest -> YYYY-MM-DD/HH-MM-SS
        This allows evaluators to discover the most recent successful run per fold.
        """
        try:
            # Navigate up from run_dir to fold level
            # run_dir: results/Dataset320_ldm_vae/fold_0/2026-01-27/15-10-46
            # fold_dir: results/Dataset320_ldm_vae/fold_0
            fold_dir = self.run_dir.parent.parent
            latest_link = fold_dir / "latest"
            
            # Remove existing symlink/file
            if latest_link.is_symlink():
                latest_link.unlink()
            elif latest_link.exists():
                # Handle case where 'latest' is a regular file/dir
                import shutil
                if latest_link.is_dir():
                    shutil.rmtree(latest_link)
                else:
                    latest_link.unlink()
            
            # Create relative symlink for portability
            # Target: YYYY-MM-DD/HH-MM-SS (relative to fold_dir)
            relative_path = self.run_dir.relative_to(fold_dir)
            latest_link.symlink_to(relative_path)
            
            print(f"[ResultsCallback] Created symlink: {latest_link.name} -> {relative_path}")
        except Exception as e:
            print(f"[ResultsCallback] Warning: Could not create 'latest' symlink: {e}")

    def _write_debug_json(self) -> None:
        debug: dict[str, Any] = {
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
            },
            "config": OmegaConf.to_container(self.cfg, resolve=True),
        }

        (self.run_dir / "debug.json").write_text(json.dumps(debug, indent=2, sort_keys=True), encoding="utf-8")
