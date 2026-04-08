"""Checkpoint save/load helpers shared by trainers and evaluation."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
from omegaconf import OmegaConf

from .training_modes import legacy_checkpoint_modes

logger = logging.getLogger(__name__)

try:
    from hydra.utils import to_absolute_path
except Exception:  # pragma: no cover
    to_absolute_path = None  # type: ignore


def export_trainer_checkpoint(
    trainer: Any,
    epoch: int,
    checkpoint_kind: str,
    monitor_value: Optional[float] = None,
) -> Dict[str, Any]:
    """Build a full trainer checkpoint payload."""
    modules = trainer._checkpoint_modules()
    optimizers = trainer._checkpoint_optimizers()
    schedulers = trainer._checkpoint_schedulers()
    scalers = trainer._checkpoint_scalers()
    primary_state = next(iter(modules.values())).state_dict() if modules else {}

    return {
        "epoch": int(epoch),
        "global_step": int(trainer.global_step),
        "best_monitor_value": trainer.best_monitor_value,
        "monitor_metric": trainer.monitor_metric,
        "monitor_mode": trainer.monitor_mode,
        "val_loss": trainer.best_monitor_value,
        "best": trainer.best_monitor_value,
        "cfg": OmegaConf.to_container(trainer.cfg, resolve=True),
        "model_state_dict": to_cpu(primary_state),
        "model_state_dicts": {
            name: to_cpu(module.state_dict())
            for name, module in modules.items()
        },
        "optimizer_state_dict": {
            name: to_cpu(optimizer.state_dict())
            for name, optimizer in optimizers.items()
        },
        "scheduler_state_dict": {
            name: to_cpu(scheduler.state_dict())
            for name, scheduler in schedulers.items()
        },
        "scaler_state_dict": {
            name: to_cpu(scaler.state_dict())
            for name, scaler in scalers.items()
        },
        "resume_metadata": {
            "saved_at": datetime.now().isoformat(),
            "checkpoint_kind": checkpoint_kind,
            "monitor_value": monitor_value,
            "resume_mode": trainer.resume_info.get("resume_mode", "fresh"),
            "resume_checkpoint_path": trainer.resume_info.get("checkpoint_path"),
            "resume_source_run_dir": trainer.resume_info.get("source_run_dir"),
            "run_dir": str(resolve_current_run_dir()),
        },
    }


def resume_trainer_if_available(trainer: Any) -> None:
    """Restore trainer state only when an explicit resume checkpoint is configured."""
    checkpoint_path = resolve_configured_resume_checkpoint(trainer.cfg)
    if checkpoint_path is None:
        trainer.start_epoch = 0
        trainer.resume_info = {"resume_mode": "fresh"}
        logger.info("No trainer.resume_checkpoint_path configured - starting from epoch 0")
        return

    checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
    assert_resume_compatible(trainer.cfg, checkpoint)
    resume_mode = load_checkpoint_payload(trainer, checkpoint)
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    source_run_dir = resolve_checkpoint_run_dir(checkpoint_path)

    trainer.start_epoch = max(0, checkpoint_epoch + 1)
    trainer.resume_info = {
        "resume_mode": resume_mode,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "start_epoch": trainer.start_epoch,
        "source_run_dir": str(source_run_dir),
    }

    if resume_mode == "full_resume":
        logger.info("Resuming from epoch %d - checkpoint: %s", trainer.start_epoch, checkpoint_path)
    else:
        logger.warning(
            "Warm-starting from legacy checkpoint at epoch %d - optimizer/scheduler/scaler state was reinitialized: %s",
            trainer.start_epoch,
            checkpoint_path,
        )


def resolve_configured_resume_checkpoint(cfg: Any) -> Optional[Path]:
    """Return the explicitly configured checkpoint path, if any."""
    configured = getattr(cfg.trainer, "resume_checkpoint_path", None)
    if configured in (None, "", False):
        return None

    checkpoint_path = Path(str(configured)).expanduser()
    if not checkpoint_path.is_absolute():
        if to_absolute_path is not None:
            try:
                checkpoint_path = Path(to_absolute_path(str(checkpoint_path))).resolve()
            except Exception:
                checkpoint_path = (resolve_current_run_dir() / checkpoint_path).resolve()
        else:
            checkpoint_path = (resolve_current_run_dir() / checkpoint_path).resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Configured resume checkpoint does not exist: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise ValueError(f"Configured resume checkpoint is not a file: {checkpoint_path}")
    return checkpoint_path


def resolve_checkpoint_run_dir(checkpoint_path: Path) -> Path:
    """Infer the Hydra run directory from a checkpoint file path."""
    path = checkpoint_path.resolve()
    if path.parent.name == "checkpoints":
        return path.parent.parent
    return path.parent


def sync_callback_resume_state(callbacks: list[Any], best_monitor_value: Optional[float]) -> None:
    """Propagate restored best-monitor state to callbacks that track it."""
    for callback in callbacks:
        if hasattr(callback, "best"):
            callback.best = best_monitor_value


def resolve_best_checkpoint(run_dir: Path, training_mode: str) -> Optional[Path]:
    """Resolve the preferred checkpoint for evaluation."""
    candidates = [
        run_dir / "checkpoint_best.pth",
        run_dir / "checkpoints" / "checkpoint_best.pth",
    ]
    candidates.extend(run_dir / "checkpoints" / f"best_{mode}.pth" for mode in legacy_checkpoint_modes(training_mode))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_checkpoint_payload(trainer: Any, checkpoint: Mapping[str, Any]) -> str:
    """Load a checkpoint into the trainer and return the resume mode."""
    if can_full_resume(trainer, checkpoint):
        load_model_states(trainer._checkpoint_modules(), checkpoint)
        load_named_state_bundle(checkpoint.get("optimizer_state_dict"), trainer._checkpoint_optimizers(), "optimizer")
        load_named_state_bundle(checkpoint.get("scheduler_state_dict"), trainer._checkpoint_schedulers(), "scheduler")
        load_named_state_bundle(
            checkpoint.get("scaler_state_dict"),
            trainer._checkpoint_scalers(),
            "scaler",
            required=False,
        )
        trainer.global_step = int(checkpoint.get("global_step", 0))
        trainer.best_monitor_value = extract_best_monitor_value(checkpoint)
        return "full_resume"

    load_available_model_states(trainer._checkpoint_modules(), checkpoint)
    trainer.global_step = int(checkpoint.get("global_step", 0)) if "global_step" in checkpoint else 0
    trainer.best_monitor_value = extract_best_monitor_value(checkpoint)
    return "warm_start"


def can_full_resume(trainer: Any, checkpoint: Mapping[str, Any]) -> bool:
    """Return True when the checkpoint contains all exact-resume state."""
    modules = trainer._checkpoint_modules()
    if not modules:
        return False

    model_states = checkpoint.get("model_state_dicts")
    if len(modules) == 1:
        if model_states is None and checkpoint.get("model_state_dict") is None:
            return False
    elif not isinstance(model_states, Mapping) or not all(name in model_states for name in modules):
        return False

    if not bundle_has_required_states(checkpoint.get("optimizer_state_dict"), trainer._checkpoint_optimizers(), "optimizer"):
        return False
    if not bundle_has_required_states(checkpoint.get("scheduler_state_dict"), trainer._checkpoint_schedulers(), "scheduler"):
        return False
    if not bundle_has_required_states(checkpoint.get("scaler_state_dict"), trainer._checkpoint_scalers(), "scaler"):
        return False
    return True


def bundle_has_required_states(bundle: Any, objects: Mapping[str, Any], bundle_kind: str) -> bool:
    """Return whether a state bundle covers the expected objects."""
    if not objects:
        return True
    if len(objects) == 1 and is_direct_state_dict(bundle, bundle_kind):
        return True
    return isinstance(bundle, Mapping) and all(name in bundle for name in objects)


def load_model_states(modules: Mapping[str, torch.nn.Module], checkpoint: Mapping[str, Any]) -> None:
    """Load all expected model state dicts from a checkpoint."""
    model_states = checkpoint.get("model_state_dicts")
    model_states = model_states if isinstance(model_states, Mapping) else {}

    for index, (name, module) in enumerate(modules.items()):
        state_dict = model_states.get(name)
        if state_dict is None and index == 0:
            state_dict = checkpoint.get("model_state_dict")
        if state_dict is None:
            raise ValueError(f"Checkpoint is missing model state for '{name}'")
        module.load_state_dict(state_dict)


def load_available_model_states(modules: Mapping[str, torch.nn.Module], checkpoint: Mapping[str, Any]) -> None:
    """Load any compatible model weights from a checkpoint for warm-starting."""
    model_states = checkpoint.get("model_state_dicts")
    model_states = model_states if isinstance(model_states, Mapping) else {}
    loaded_any = False

    for index, (name, module) in enumerate(modules.items()):
        state_dict = model_states.get(name)
        if state_dict is None and index == 0:
            state_dict = checkpoint.get("model_state_dict")
        if state_dict is None:
            continue
        module.load_state_dict(state_dict)
        loaded_any = True

    if not loaded_any:
        raise ValueError("Checkpoint does not contain any compatible model weights")


def load_named_state_bundle(
    bundle: Any,
    objects: Mapping[str, Any],
    bundle_kind: str,
    required: bool = True,
) -> None:
    """Load optimizer, scheduler, or scaler state bundles."""
    if not objects:
        return
    if bundle is None:
        if required:
            raise ValueError(f"Checkpoint is missing {bundle_kind} state")
        return
    if len(objects) == 1 and is_direct_state_dict(bundle, bundle_kind):
        next(iter(objects.values())).load_state_dict(bundle)
        return
    if not isinstance(bundle, Mapping):
        if required:
            raise ValueError(f"Checkpoint {bundle_kind} state has invalid format")
        return

    for name, obj in objects.items():
        state = bundle.get(name)
        if state is None:
            if required:
                raise ValueError(f"Checkpoint is missing {bundle_kind} state for '{name}'")
            continue
        obj.load_state_dict(state)


def is_direct_state_dict(bundle: Any, bundle_kind: str) -> bool:
    """Heuristically detect a direct state dict for single-object checkpoints."""
    if not isinstance(bundle, Mapping):
        return False
    if bundle_kind == "optimizer":
        return "state" in bundle and "param_groups" in bundle
    if bundle_kind == "scheduler":
        return "last_epoch" in bundle or "_last_lr" in bundle
    if bundle_kind == "scaler":
        return "scale" in bundle or "_scale" in bundle or "growth_factor" in bundle
    return False


def assert_resume_compatible(cfg: Any, checkpoint: Mapping[str, Any]) -> None:
    """Abort when the saved configuration is architecture-incompatible."""
    saved_cfg = checkpoint.get("cfg")
    if not isinstance(saved_cfg, Mapping):
        return

    current_cfg = OmegaConf.to_container(cfg, resolve=True)
    saved_mode = str(saved_cfg.get("trainer", {}).get("training_mode", "")).lower()
    current_mode = str(current_cfg.get("trainer", {}).get("training_mode", "")).lower()
    if saved_mode and saved_mode != current_mode:
        raise ValueError(f"Checkpoint training_mode mismatch: saved={saved_mode!r}, current={current_mode!r}")
    if saved_cfg.get("model") != current_cfg.get("model"):
        raise ValueError("Checkpoint model configuration is incompatible with the current run")


def extract_best_monitor_value(checkpoint: Mapping[str, Any]) -> Optional[float]:
    """Extract the stored best-monitor scalar from a checkpoint."""
    for key in ("best_monitor_value", "val_loss", "best"):
        value = checkpoint.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def resolve_current_run_dir() -> Path:
    """Resolve the active Hydra run directory."""
    return Path(os.getcwd()).resolve()
def to_cpu(obj: Any) -> Any:
    """Recursively clone tensors to CPU for portable checkpoints."""
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {key: to_cpu(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_cpu(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(to_cpu(value) for value in obj)
    return obj
