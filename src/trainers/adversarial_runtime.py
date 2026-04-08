"""Adversarial training helpers for the autoencoder trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from monai.losses.adversarial_loss import PatchAdversarialLoss
from monai.networks.nets.patchgan_discriminator import MultiScalePatchDiscriminator

from ..utils.scheduler import get_scheduler

try:
    from torch.amp.grad_scaler import GradScaler
except ImportError:  # pragma: no cover - torch version dependent
    from torch.cuda.amp import GradScaler


@dataclass(frozen=True)
class AdversarialSettings:
    """Configuration required for adversarial training."""

    enabled: bool
    warmup_epochs: Optional[int]
    weight: float
    use_amp: bool
    amp_device: str
    optimizer_name: str
    generator_lr: float
    discriminator_lr: float
    generator_betas: tuple[float, float]
    discriminator_betas: tuple[float, float]
    weight_decay: float


def build_adversarial_runtime(
    settings: AdversarialSettings,
    device: torch.device,
    scheduler_cfg: Any,
) -> "BaseAdversarialRuntime":
    """Create the enabled or disabled adversarial runtime."""
    if not settings.enabled:
        return DisabledAdversarialRuntime()
    return AdversarialRuntime(settings, device, scheduler_cfg)


class BaseAdversarialRuntime:
    """No-op adversarial runtime."""

    enabled = False
    active = False

    def on_epoch_start(self, epoch: int) -> None:
        """Update runtime state at the start of an epoch."""

    def generator_loss(self, reconstruction: torch.Tensor) -> torch.Tensor:
        """Return the generator adversarial loss."""
        return torch.tensor(0.0, device=reconstruction.device)

    def update_discriminator(self, inputs: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
        """Update the discriminator and return its loss."""
        return torch.tensor(0.0, device=inputs.device)

    def discriminator_loss(self, inputs: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
        """Return the discriminator loss without updating weights."""
        return torch.tensor(0.0, device=inputs.device)

    def step_scheduler(self, logger: Any, epoch: int) -> None:
        """Advance runtime schedulers."""

    def checkpoint_modules(self) -> Dict[str, torch.nn.Module]:
        """Return checkpointed modules."""
        return {}

    def checkpoint_optimizers(self) -> Dict[str, Any]:
        """Return checkpointed optimizers."""
        return {}

    def checkpoint_schedulers(self) -> Dict[str, Any]:
        """Return checkpointed schedulers."""
        return {}

    def checkpoint_scalers(self) -> Dict[str, Any]:
        """Return checkpointed scalers."""
        return {}


class DisabledAdversarialRuntime(BaseAdversarialRuntime):
    """Disabled adversarial runtime."""


class AdversarialRuntime(BaseAdversarialRuntime):
    """Active adversarial runtime with discriminator state."""

    enabled = True

    def __init__(self, settings: AdversarialSettings, device: torch.device, scheduler_cfg: Any):
        self.settings = settings
        self.active = False
        optimizer_cls = getattr(torch.optim, settings.optimizer_name)

        self.discriminator = MultiScalePatchDiscriminator(
            num_d=2,
            num_layers_d=2,
            spatial_dims=3,
            channels=32,
            in_channels=1,
            out_channels=1,
            minimum_size_im=64,
        ).to(device)
        self.loss = PatchAdversarialLoss(criterion="least_squares")
        self.optimizer_d = optimizer_cls(
            params=self.discriminator.parameters(),
            lr=settings.discriminator_lr,
            betas=settings.discriminator_betas,
            weight_decay=settings.weight_decay,
        )
        self.scheduler_d = get_scheduler(self.optimizer_d, scheduler_cfg)
        self.scaler_d = self._create_grad_scaler()

    def on_epoch_start(self, epoch: int) -> None:
        self.active = epoch >= int(self.settings.warmup_epochs or 0)

    def generator_loss(self, reconstruction: torch.Tensor) -> torch.Tensor:
        if not self.active:
            return torch.tensor(0.0, device=reconstruction.device)

        self._set_requires_grad(False)
        outputs, _ = self.discriminator(reconstruction.contiguous().float())
        losses = [self.loss(logits, target_is_real=True, for_discriminator=False) for logits in outputs]
        self._set_requires_grad(True)
        return torch.stack(losses).sum() / len(losses)

    def update_discriminator(self, inputs: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
        if not self.active:
            return torch.tensor(0.0, device=inputs.device)

        self.optimizer_d.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self.settings.amp_device, enabled=self.settings.use_amp):
            loss_d = self.discriminator_loss(inputs, reconstruction)
        self.scaler_d.scale(loss_d).backward()
        self.scaler_d.step(self.optimizer_d)
        self.scaler_d.update()
        return loss_d

    def discriminator_loss(self, inputs: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
        outputs_fake, _ = self.discriminator(reconstruction.contiguous().float().detach())
        outputs_real, _ = self.discriminator(inputs.contiguous().float().detach())

        fake_loss = sum(self.loss(logits, target_is_real=False, for_discriminator=True) for logits in outputs_fake)
        real_loss = sum(self.loss(logits, target_is_real=True, for_discriminator=True) for logits in outputs_real)
        return self.settings.weight * 0.5 * ((fake_loss / len(outputs_fake)) + (real_loss / len(outputs_real)))

    def step_scheduler(self, logger: Any, epoch: int) -> None:
        if self.active:
            self.scheduler_d.step()
            logger.add_scalar("LR/discriminator", self.optimizer_d.param_groups[0]["lr"], epoch)

    def checkpoint_modules(self) -> Dict[str, torch.nn.Module]:
        return {"discriminator": self.discriminator}

    def checkpoint_optimizers(self) -> Dict[str, Any]:
        return {"optimizer_d": self.optimizer_d}

    def checkpoint_schedulers(self) -> Dict[str, Any]:
        return {"scheduler_d": self.scheduler_d}

    def checkpoint_scalers(self) -> Dict[str, Any]:
        return {"scaler_d": self.scaler_d}

    def _set_requires_grad(self, requires_grad: bool) -> None:
        for parameter in self.discriminator.parameters():
            parameter.requires_grad_(requires_grad)

    def _create_grad_scaler(self) -> GradScaler:
        try:
            return GradScaler(self.settings.amp_device, enabled=self.settings.use_amp)
        except TypeError:  # pragma: no cover - torch version dependent
            return GradScaler(enabled=self.settings.use_amp)
