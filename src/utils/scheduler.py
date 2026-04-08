"""Learning rate scheduler factory."""
import logging

import torch.optim as optim
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

_SUPPORTED = ("CosineAnnealingLR", "StepLR", "ReduceLROnPlateau")


def get_scheduler(optimizer: optim.Optimizer, scheduler_cfg: DictConfig) -> optim.lr_scheduler._LRScheduler:
    """Return a PyTorch LR scheduler from a Hydra config node.

    Args:
        optimizer: The optimizer to wrap.
        scheduler_cfg: Config node with at minimum a ``name`` key.
            Remaining keys are forwarded as keyword arguments to the scheduler.

    Returns:
        An instantiated LR scheduler.

    Raises:
        ValueError: If ``scheduler_cfg.name`` is not one of the supported schedulers.
    """
    name = scheduler_cfg.get("name", "CosineAnnealingLR")

    if name == "CosineAnnealingLR":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_cfg.get("T_max", 1000),
            eta_min=scheduler_cfg.get("eta_min", 0.0),
        )
    elif name == "StepLR":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_cfg.get("step_size", 10),
            gamma=scheduler_cfg.get("gamma", 0.5),
        )
    elif name == "ReduceLROnPlateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_cfg.get("mode", "min"),
            factor=scheduler_cfg.get("factor", 0.1),
            patience=scheduler_cfg.get("patience", 10),
        )
    else:
        raise ValueError(
            f"Unknown scheduler '{name}'. Supported schedulers: {_SUPPORTED}"
        )
