"""Adversarial Autoencoder Trainer with GAN-based regularization."""
from typing import Dict, Any, Optional, Callable
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from omegaconf import DictConfig

from .base_trainer import BaseTrainer
from .loss import ELBOLoss, Perceptual
from ..dataloader.dataloader import map_label_to_category
from ..models.discriminator import MultiScaleDiscriminator
from monai.losses.adversarial_loss import PatchAdversarialLoss

class AdversarialAutoencoderTrainer(BaseTrainer):
    """Trainer for adversarial autoencoder combining reconstruction, KL, adversarial, and perceptual losses."""
    
    def __init__(self, cfg: DictConfig, model: torch.nn.Module, dataloaders: Dict[str, Any],
                 device: torch.device, logger: Any, callbacks: Optional[list] = None,
                 max_epochs: Optional[int] = None, metrics: Optional[Dict[str, Callable]] = None,
                 normalization_stats: Optional[Dict[str, float]] = None):
        super().__init__(cfg, dataloaders, device, logger, callbacks, max_epochs, normalization_stats)
        self.generator = model.to(device)
        self.discriminator = MultiScaleDiscriminator(cfg).to(device)
        self.use_amp = getattr(cfg.trainer, "use_amp", False)
        self.metrics = {
            "PSNR": self._psnr_torch,
            "SSIM": self._ssim_torch
        }
        
        # Latent collection for UMAP
        self.collect_latents = getattr(cfg.trainer, 'collect_latents', True)
        self.latent_collection = {'train': [], 'val': []}
        self.label_collection = {'train': [], 'val': []}
        self._collect_this_epoch = False

        self.setup_optimizers()
        self.setup_schedulers()
        self.setup_criteria()

    def setup_models(self) -> None:
        """Models initialized in __init__."""
        pass
        
    def setup_optimizers(self) -> None:
        """Initialize optimizers with selective weight decay."""
        cfg_opt = self.cfg.optimizer
        betas = self._parse_betas(getattr(cfg_opt, "betas", (0.0, 0.9)))
        
        self.optimizer_g = torch.optim.AdamW(self._get_param_groups(self.generator),
                                             lr=float(getattr(cfg_opt, "lr_g", 1e-4)), betas=betas,
                                             weight_decay=float(getattr(cfg_opt, "weight_decay_g", 0.0)))
        self.optimizer_d = torch.optim.AdamW(self._get_param_groups(self.discriminator),
                                             lr=float(getattr(cfg_opt, "lr_d", 4e-4)), betas=betas)
        
        self.scaler_g = GradScaler(enabled=self.use_amp)
        self.scaler_d = GradScaler(enabled=self.use_amp)
    
    def setup_schedulers(self) -> None:
        """Initialize cosine annealing schedulers."""
        self.scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_g, T_max=self.max_epochs, eta_min=1e-6)
        self.scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_d, T_max=self.max_epochs, eta_min=1e-6)
    
    def setup_criteria(self) -> None:
        """Initialize loss functions and weights."""
        self.ae_loss = ELBOLoss().to(self.device)
        self.perc_loss = Perceptual().to(self.device)
        self.adv_loss = PatchAdversarialLoss(criterion="hinge", reduction="mean").to(self.device)
        
        lw = self.cfg.loss_weights
        self.w_recon, self.w_kl = float(lw.recon), float(lw.kl)
        self.w_adv, self.w_perc = float(lw.adversarial), float(getattr(lw, "perceptual", 0.0))
        self.n_critic = int(getattr(self.cfg.gan, "n_critic", 1))
    
    def set_train_mode(self, train: bool) -> None:
        """Set models to train/eval mode."""
        self.generator.train(train)
        self.discriminator.train(train)
    
    def train_step(self, batch: Any) -> Dict[str, float]:
        """Execute adversarial training step."""
        x = batch["ct"]["data"].to(self.device)
        x_recon, z_mu, z_sigma = self.generator(x)
        self._collect_latents(z_mu, batch, 'train')
        
        # Train discriminator
        adv_d_total = self._train_discriminator(x, x_recon)
        
        # Train generator
        self.optimizer_g.zero_grad(set_to_none=True)
        beta = self.w_kl
        recon, kl = self.ae_loss(x_recon, x, mu=z_mu, sigma=z_sigma, beta=beta)
        
        with autocast(enabled=self.use_amp):
            adv_g = self.adv_loss(self._flatten_logits(self.discriminator(x_recon)), 
                                  target_is_real=True, for_discriminator=False)
            perc = self.perc_loss(x_recon, x) if self.w_perc > 0 else torch.tensor(0.0, device=self.device)
        
        loss_g = recon + kl + self.w_adv * adv_g + self.w_perc * perc
        self.scaler_g.scale(loss_g).backward()
        self.scaler_g.step(self.optimizer_g)
        self.scaler_g.update()
        
        return self._compute_metrics(loss_g, recon, kl, adv_g, perc, adv_d_total, x_recon, x)
    
    def validation_step(self, batch: Any) -> Dict[str, float]:
        """Execute validation step."""
        x = batch["ct"]["data"].to(self.device)
        
        with torch.no_grad():
            x_recon, z_mu, z_sigma = self.generator(x)
            
            # Collect sample images for logging (efficient - during validation)
            if self.collect_img:
                self.random_idx = torch.randint(0, x.shape[0], (1,)).item()
                self.rand_input = x[self.random_idx].detach().cpu()
                self.rand_recon = x_recon[self.random_idx].detach().cpu()
            
            self._collect_latents(z_mu, batch, 'val')
            
            beta = self.w_kl
            recon, kl = self.ae_loss(x_recon, x, mu=z_mu, sigma=z_sigma, beta=beta)
            
            with autocast(enabled=self.use_amp):
                perc = self.perc_loss(x_recon, x) if self.w_perc > 0 else torch.tensor(0.0, device=self.device)
            
            loss = recon + kl + self.w_perc * perc
        
        self.collect_img = False  # Only collect once per epoch
        return self._compute_metrics(loss, recon, kl, torch.tensor(0.0), perc, 0.0, x_recon, x)
    
    def step_schedulers(self) -> None:
        """Step learning rate schedulers."""
        self.scheduler_g.step()
        self.scheduler_d.step()
    
    def on_fit_start(self) -> None:
        """Initialize collection state."""
        super().on_fit_start()
        self._collect_this_epoch = False
        self.collect_img = False
        self.epoch_count = 0
    
    def on_epoch_start(self, epoch: int, is_train: bool) -> None:
        """Determine latent collection and clear buffers."""
        self.epoch_count = epoch
        image_log_interval = getattr(self.cfg.trainer, 'image_log_interval', 5)
        self._collect_this_epoch = self.collect_latents and (epoch % getattr(self.cfg.trainer, 'umap_log_interval', 10) == 0)
        self.collect_img = (epoch % image_log_interval == 0)
        if self._collect_this_epoch:
            split = 'train' if is_train else 'val'
            self.latent_collection[split], self.label_collection[split] = [], []
    
    def on_epoch_end(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        """Log images and UMAP at intervals."""
        image_log_interval = getattr(self.cfg.trainer, 'image_log_interval', 5)
        if epoch % image_log_interval == 0:
            self._log_sample_images(epoch)
        if self._collect_this_epoch:
            self._log_latent_umap(epoch)
        super().on_epoch_end(epoch, train_metrics, val_metrics)
    
    # --- Helper Methods ---
    
    def _train_discriminator(self, x_real: torch.Tensor, x_fake: torch.Tensor) -> float:
        """Train discriminator for n_critic iterations."""
        total_loss = 0.0
        for _ in range(self.n_critic):
            self.optimizer_d.zero_grad(set_to_none=True)
            with autocast(enabled=self.use_amp):
                d_real = self._flatten_logits(self.discriminator(x_real))
                d_fake = self._flatten_logits(self.discriminator(x_fake.detach()))
                loss = 0.5 * (self.adv_loss(d_real, target_is_real=True, for_discriminator=True) +
                             self.adv_loss(d_fake, target_is_real=False, for_discriminator=True))
            self.scaler_d.scale(loss).backward()
            self.scaler_d.step(self.optimizer_d)
            self.scaler_d.update()
            total_loss += loss.item()
        return total_loss
    
    def _collect_latents(self, z_mu: torch.Tensor, batch: Dict[str, Any], split: str) -> None:
        """Collect latents and labels for UMAP."""
        if self._collect_this_epoch:
            z_mu_fp32 = z_mu.detach().float()
            if not (torch.isnan(z_mu_fp32).any() or torch.isinf(z_mu_fp32).any()):
                self.latent_collection[split].append(z_mu_fp32.cpu().numpy())
                if 'label' in batch:
                    labels = [map_label_to_category(lbl) for lbl in batch['label']]
                    self.label_collection[split].append(np.array(labels))
    
    def _compute_metrics(self, loss: torch.Tensor, recon: torch.Tensor, kl: torch.Tensor,
                        adv_g: torch.Tensor, perc: torch.Tensor, adv_d: float,
                        x_recon: torch.Tensor, x: torch.Tensor) -> Dict[str, float]:
        """Compute all metrics: loss components + quality metrics (HU or IQR-normalized space)."""
        # Loss components
        metrics = {
            "loss": loss.item(),
            "recon": recon.item(),
            "kl": kl.item(),
            "adv_g": adv_g.item() if isinstance(adv_g, torch.Tensor) else adv_g,
            "adv_d": adv_d,
            "perc": perc.item() if isinstance(perc, torch.Tensor) else perc,
        }
        
        # Quality metrics (PSNR, SSIM) - computed in HU space if stats available
        metrics.update(self._compute_quality_metrics(x_recon, x))
        
        return metrics
    
    def _initialize_epoch_metrics(self) -> Dict[str, float]:
        """Initialize metrics for epoch aggregation."""
        return {k: 0.0 for k in ["loss", "recon", "kl", "adv_g", "adv_d", "perc", "PSNR", "PSNR_count"] + list(self.metrics.keys())}
    
    def _average_epoch_metrics(self, epoch_metrics: Dict[str, float], n_batches: int) -> Dict[str, float]:
        """Average metrics over epoch."""
        n = max(1, n_batches)
        averaged = {k: v / n for k, v in epoch_metrics.items() if k != 'PSNR_count'}
        if epoch_metrics.get('PSNR_count', 0) > 0:
            averaged['PSNR'] = epoch_metrics['PSNR'] / epoch_metrics['PSNR_count']
        return averaged
    
    def _log_latent_umap(self, epoch: int) -> None:
        """Log UMAP visualization and clear buffers."""
        for split in ['train', 'val']:
            if self.latent_collection[split]:
                self.logger.log_latent_umap(latents=self.latent_collection[split],
                                           labels=self.label_collection[split] or None,
                                           step=epoch, tag=split)
                self.latent_collection[split], self.label_collection[split] = [], []
    
    def _log_sample_images(self, epoch: int, tag: str = "val") -> None:
        """Log cached sample images (efficient - no extra forward pass)."""
        self.logger.log_axial_figure(
            self.rand_input,
            self.rand_recon,
            step=epoch,
            tag=tag,
            norm_stats=self.norm_stats
        )
    
    @staticmethod
    def _parse_betas(betas_raw) -> tuple:
        """Parse betas from various formats (str/list/tuple)."""
        if isinstance(betas_raw, str):
            import ast
            parsed = ast.literal_eval(betas_raw)
            return (float(parsed[0]), float(parsed[1]))
        return (float(betas_raw[0]), float(betas_raw[1])) if isinstance(betas_raw, (list, tuple)) else (0.0, 0.9)
    
    @staticmethod
    def _get_param_groups(module):
        """Yield parameter groups with selective weight decay."""
        for name, p in module.named_parameters():
            if p.requires_grad:
                yield {"params": [p], "weight_decay": 0.0 if name.endswith(".bias") or "norm" in name.lower() else None}
    
    @staticmethod
    def _flatten_logits(out):
        """Flatten nested discriminator outputs to list of tensors."""
        if isinstance(out, torch.Tensor):
            return [out]
        flat, stack = [], [out]
        while stack:
            v = stack.pop()
            (flat.append(v) if isinstance(v, torch.Tensor) else stack.extend(v)) if isinstance(v, (torch.Tensor, list, tuple)) else None
        if not flat:
            raise TypeError("Discriminator returned no tensors.")
        return flat
