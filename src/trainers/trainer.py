"""Autoencoder trainer with optional adversarial and monitoring helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
import torch.optim as optim
from monai.losses.perceptual import PerceptualLoss
from omegaconf import DictConfig
from torch.nn import L1Loss

from .adversarial_runtime import AdversarialSettings, build_adversarial_runtime
from .base_trainer import BaseTrainer
from .representation_monitoring import MonitoringSettings, RepresentationMonitor
from ..utils.scheduler import get_scheduler

try:
    from torch.amp.grad_scaler import GradScaler
except ImportError:  # pragma: no cover - torch version dependent
    from torch.cuda.amp import GradScaler


@dataclass(frozen=True)
class AutoencoderSettings:
    """Normalized trainer settings used throughout the autoencoder flow."""

    use_masked_loss: bool
    grad_clip_norm: float
    kl_weight: float
    perceptual_weight: float
    tumor_weight: float
    non_tumor_weight: float
    use_amp: bool
    amp_device: str
    optimizer_name: str
    generator_lr: float
    generator_betas: tuple[float, float]
    weight_decay: float
    adversarial: AdversarialSettings
    monitoring: MonitoringSettings

    @classmethod
    def from_cfg(cls, cfg: DictConfig, device: torch.device) -> "AutoencoderSettings":
        trainer_cfg = cfg.trainer
        optimizer_cfg = cfg.optimizer
        loss_cfg = cfg.loss_weights
        amp_device = device.type if device.type == "cuda" else "cpu"
        warmup_epochs = trainer_cfg.adversarial_warmup_epochs

        return cls(
            use_masked_loss=bool(getattr(trainer_cfg, "use_masked_loss", False)),
            grad_clip_norm=float(getattr(trainer_cfg, "grad_clip_norm", 1.0)),
            kl_weight=float(loss_cfg.kl),
            perceptual_weight=float(loss_cfg.perceptual),
            tumor_weight=float(loss_cfg.w_tumor),
            non_tumor_weight=float(loss_cfg.w_nontumor),
            use_amp=bool(trainer_cfg.use_amp),
            amp_device=amp_device,
            optimizer_name=str(optimizer_cfg.name),
            generator_lr=float(optimizer_cfg.lr),
            generator_betas=tuple(optimizer_cfg.betas),
            weight_decay=float(getattr(optimizer_cfg, "weight_decay", 0.0)),
            adversarial=AdversarialSettings(
                enabled=warmup_epochs is not None,
                warmup_epochs=warmup_epochs,
                weight=float(loss_cfg.adversarial),
                use_amp=bool(trainer_cfg.use_amp),
                amp_device=amp_device,
                optimizer_name=str(optimizer_cfg.name),
                generator_lr=float(optimizer_cfg.lr),
                discriminator_lr=float(optimizer_cfg.lr_d),
                generator_betas=tuple(optimizer_cfg.betas),
                discriminator_betas=tuple(optimizer_cfg.betas_d),
                weight_decay=float(getattr(optimizer_cfg, "weight_decay", 0.0)),
            ),
            monitoring=MonitoringSettings(
                collect_latents=bool(getattr(trainer_cfg, "collect_latents", True)),
                umap_log_interval=int(getattr(trainer_cfg, "umap_log_interval", 10)),
                image_log_interval=int(getattr(trainer_cfg, "image_log_interval", 5)),
                probe_interval=int(getattr(trainer_cfg, "probe_interval", 0)),
                probe_max_batches=int(getattr(trainer_cfg, "probe_max_batches", 20)),
                probe_epochs=int(getattr(trainer_cfg, "probe_epochs", 50)),
                probe_lr=float(getattr(trainer_cfg, "probe_lr", 1e-2)),
                probe_l2=float(getattr(trainer_cfg, "probe_l2", 1e-4)),
            ),
        )


class AutoencoderTrainer(BaseTrainer):
    """Trainer for autoencoders with optional adversarial training."""

    def __init__(
        self,
        cfg: DictConfig,
        model: torch.nn.Module,
        dataloaders: Dict[str, Any],
        device: torch.device,
        logger: Any,
        callbacks: Optional[list] = None,
        max_epochs: Optional[int] = None,
        normalization_stats: Optional[Dict[str, float]] = None,
    ):
        super().__init__(cfg, dataloaders, device, logger, callbacks, max_epochs, normalization_stats)
        self.model = model.to(device)
        self.settings = AutoencoderSettings.from_cfg(cfg, device)
        self.adversarial = build_adversarial_runtime(self.settings.adversarial, device, cfg.scheduler)
        self.monitoring = RepresentationMonitor(
            self.settings.monitoring,
            logger=self.logger,
            model=self.model,
            dataloaders=self.dataloaders,
            device=self.device,
        )

        self.setup_optimizers()
        self.setup_schedulers()
        self.setup_criteria()
        self.scaler_g = self._create_grad_scaler()

    def setup_models(self) -> None:
        """No-op: models are already provided to the trainer."""

    def setup_optimizers(self) -> None:
        optimizer_cls = getattr(optim, self.settings.optimizer_name)
        self.optimizer_g = optimizer_cls(
            params=self.model.parameters(),
            lr=self.settings.generator_lr,
            betas=self.settings.generator_betas,
            weight_decay=self.settings.weight_decay,
        )

    def setup_schedulers(self) -> None:
        self.scheduler_g = get_scheduler(self.optimizer_g, self.cfg.scheduler)

    def setup_criteria(self) -> None:
        self.l1_loss = L1Loss()
        self.loss_perceptual = PerceptualLoss(
            spatial_dims=3,
            network_type="squeeze",
            is_fake_3d=True,
            fake_3d_ratio=0.2,
        ).to(self.device)

    def set_train_mode(self, train: bool) -> None:
        self.model.train(train)
        if hasattr(self.adversarial, "discriminator"):
            self.adversarial.discriminator.train(train)

    def train_step(self, epoch: int, batch: Any) -> Dict[str, float]:
        inputs, phase, multilabel_mask, tumor_mask = self._unpack_batch(batch)

        self.optimizer_g.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self.settings.amp_device, enabled=self.settings.use_amp):
            reconstruction, z_mu, _, recons_loss, p_loss, klloss, loss_g = self._forward_and_losses(
                inputs,
                tumor_mask,
                phase=phase,
            )
            gen_adv = self.adversarial.generator_loss(reconstruction)

        self.scaler_g.scale(loss_g).backward()
        self.scaler_g.unscale_(self.optimizer_g)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.settings.grad_clip_norm)
        self.scaler_g.step(self.optimizer_g)
        self.scaler_g.update()

        loss_d = self.adversarial.update_discriminator(inputs, reconstruction)
        self.monitoring.record_batch("train", batch, inputs, reconstruction, multilabel_mask, z_mu)

        metrics = self._make_loss_metrics(loss_g, loss_d, recons_loss, klloss, p_loss, gen_adv)
        with torch.no_grad():
            metrics.update(self._compute_quality_metrics(reconstruction, inputs))
        return metrics

    def validation_step(self, epoch: int, batch: Any) -> Dict[str, float]:
        inputs, phase, multilabel_mask, tumor_mask = self._unpack_batch(batch)
        self.model.eval()

        with torch.no_grad():
            with torch.autocast(device_type=self.settings.amp_device, enabled=self.settings.use_amp):
                reconstruction, z_mu, _, recons_loss, p_loss, klloss, loss_g = self._forward_and_losses(
                    inputs,
                    tumor_mask,
                    phase=phase,
                )
                gen_adv = self.adversarial.generator_loss(reconstruction)

            loss_d = self.adversarial.discriminator_loss(inputs, reconstruction) if self.adversarial.active else torch.tensor(
                0.0,
                device=inputs.device,
            )
            self.monitoring.record_batch("val", batch, inputs, reconstruction, multilabel_mask, z_mu)

            metrics = self._make_loss_metrics(loss_g, loss_d, recons_loss, klloss, p_loss, gen_adv)
            metrics.update(self._compute_quality_metrics(reconstruction, inputs))
            return metrics

    def step_schedulers(self, epoch: int) -> None:
        self.scheduler_g.step()
        self.logger.add_scalar("LR/generator", self.optimizer_g.param_groups[0]["lr"], epoch)
        self.adversarial.step_scheduler(self.logger, epoch)

    def on_epoch_start(self, epoch: int, is_train: bool) -> None:
        self.adversarial.on_epoch_start(epoch)
        self.monitoring.on_epoch_start(epoch, is_train)

    def on_epoch_end(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        self.monitoring.on_epoch_end(epoch)
        super().on_epoch_end(epoch, train_metrics, val_metrics)

    def _checkpoint_modules(self) -> Dict[str, torch.nn.Module]:
        modules = {"model": self.model}
        modules.update(self.adversarial.checkpoint_modules())
        return modules

    def _checkpoint_optimizers(self) -> Dict[str, Any]:
        optimizers = {"optimizer_g": self.optimizer_g}
        optimizers.update(self.adversarial.checkpoint_optimizers())
        return optimizers

    def _checkpoint_schedulers(self) -> Dict[str, Any]:
        schedulers = {"scheduler_g": self.scheduler_g}
        schedulers.update(self.adversarial.checkpoint_schedulers())
        return schedulers

    def _checkpoint_scalers(self) -> Dict[str, Any]:
        scalers = {"scaler_g": self.scaler_g}
        scalers.update(self.adversarial.checkpoint_scalers())
        return scalers

    def _create_grad_scaler(self) -> GradScaler:
        try:
            return GradScaler(self.settings.amp_device, enabled=self.settings.use_amp)
        except TypeError:  # pragma: no cover - torch version dependent
            return GradScaler(enabled=self.settings.use_amp)

    def _initialize_epoch_metrics(self) -> Dict[str, float]:
        metrics = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "PSNR_count": 0}
        for metric_name in self.metrics:
            metrics[metric_name] = 0.0
        return metrics

    def _average_epoch_metrics(self, epoch_metrics: Dict[str, float], n_batches: int) -> Dict[str, float]:
        quality_keys = set(self.metrics.keys())
        psnr_computed = epoch_metrics.get("PSNR_count", 0) > 0
        averaged: Dict[str, float] = {}

        for key, value in epoch_metrics.items():
            if key == "PSNR_count" or (key in quality_keys and not psnr_computed):
                continue
            averaged[key] = value / max(1, n_batches)

        if psnr_computed and "PSNR" in epoch_metrics:
            averaged["PSNR"] = epoch_metrics["PSNR"] / epoch_metrics["PSNR_count"]
        return averaged

    def _unpack_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = batch["ct"]["data"].to(self.device)
        phase = batch["phase"].to(self.device)
        multilabel_mask = batch["mask"]["data"].to(self.device)
        tumor_mask = (multilabel_mask == 2).float()

        if inputs.shape != tumor_mask.shape:
            raise ValueError(f"Shape mismatch: {inputs.shape} vs {tumor_mask.shape}")
        return inputs, phase, multilabel_mask, tumor_mask

    def _forward_and_losses(
        self,
        inputs: torch.Tensor,
        tumor_mask: torch.Tensor,
        phase: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        reconstruction, z_mu, z_sigma = self.model(inputs, phase=phase)
        recons_loss, p_loss = self._compute_recon_losses(inputs, reconstruction, tumor_mask)
        klloss = self.kl_loss(z_mu, z_sigma)
        loss_g = recons_loss + self.settings.kl_weight * klloss + self.settings.perceptual_weight * p_loss
        return reconstruction, z_mu, z_sigma, recons_loss, p_loss, klloss, loss_g

    def _compute_recon_losses(
        self,
        inputs: torch.Tensor,
        reconstruction: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.settings.use_masked_loss:
            recon_loss = self._weighted_masked_loss(inputs, reconstruction, mask)
        else:
            recon_loss = self.l1_loss(reconstruction, inputs)
        return recon_loss, self.loss_perceptual(reconstruction, inputs)

    def _weighted_masked_loss(
        self,
        inputs: torch.Tensor,
        reconstruction: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        l1_losses = F.l1_loss(inputs, reconstruction, reduction="none")
        tumor_mask = mask.float()
        non_tumor_mask = 1.0 - tumor_mask

        tumor_loss = (l1_losses * tumor_mask).sum() / tumor_mask.sum().clamp_min(1e-6)
        non_tumor_loss = (l1_losses * non_tumor_mask).sum() / non_tumor_mask.sum().clamp_min(1e-6)
        return (self.settings.tumor_weight * tumor_loss) + (self.settings.non_tumor_weight * non_tumor_loss)

    def kl_loss(self, z_mu: torch.Tensor, z_sigma: torch.Tensor) -> torch.Tensor:
        klloss = 0.5 * torch.sum(
            z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2).clamp(min=1e-6)) - 1,
            dim=[1, 2, 3, 4],
        )
        return torch.sum(klloss) / klloss.shape[0]

    def _make_loss_metrics(
        self,
        loss_g: torch.Tensor,
        loss_d: torch.Tensor,
        recon_loss: torch.Tensor,
        klloss: torch.Tensor,
        perceptual_loss: torch.Tensor,
        gen_adv: torch.Tensor,
    ) -> Dict[str, float]:
        return {
            "loss": loss_g.item(),
            "loss_d": loss_d.item(),
            "recon": recon_loss.item(),
            "kl": klloss.item(),
            "perceptual": perceptual_loss.item(),
            "g_adv": gen_adv.item(),
        }
