"""
Pure Autoencoder Trainer (VAE/AE without adversarial component).
Handles reconstruction + KL divergence loss.
"""
from typing import Dict, Any, Optional, Callable
import numpy as np
import torch
import torch.optim as optim
from omegaconf import DictConfig

from torch.nn import L1Loss
from monai.losses.perceptual import PerceptualLoss
from monai.losses.adversarial_loss import PatchAdversarialLoss
from monai.networks.nets.patchgan_discriminator import MultiScalePatchDiscriminator

from .base_trainer import BaseTrainer
from ..dataloader.dataloader import map_label_to_category
from ..utils.scheduler import get_scheduler
import torch.nn.functional as F

class AutoencoderTrainer(BaseTrainer):
    """
    Trainer for pure Autoencoder (VAE) without adversarial training.
    Optimizes reconstruction + KL divergence.
    """
    
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
        self.model = model
        
        
        # Latent space collection for UMAP
        self.collect_latents = getattr(cfg.trainer, 'collect_latents', True)
        self.latent_collection = {'train': [], 'val': []}
        self.label_collection = {'train': [], 'val': []}
        self.discriminator = MultiScalePatchDiscriminator(
            num_d=2, 
            num_layers_d=2, 
            spatial_dims=3, 
            channels=32, 
            in_channels=1, 
            out_channels=1,
            minimum_size_im=64  # Match minimum dimension of input (96,96,64)
        )
        self.discriminator.to(device)

        # Cached config
        self.use_masked_loss = getattr(self.cfg.trainer, 'use_masked_loss', False)
        self.kl_w = self.cfg.loss_weights.kl
        self.perc_w = self.cfg.loss_weights.perceptual
        self.adv_w = self.cfg.loss_weights.adversarial
        self.adv_warmup_epochs = self.cfg.adversarial_warmup_epochs

        self.samples = {"train": {}, "val": {}}
        
        # Initialize components
        self.setup_models()
        self.setup_optimizers()
        self.setup_schedulers()
        self.setup_criteria()
        
    def setup_models(self) -> None:
        """Move model to device."""
        self.model = self.model.to(self.device)
    
    def setup_optimizers(self) -> None:
        """Initialize optimizer for autoencoder."""
        OptimClass = getattr(optim, self.cfg.optimizer.name)
        
        self.optimizer_g = OptimClass(params=self.model.parameters(), 
                                      lr=self.cfg.optimizer.lr_g, 
                                      betas=self.cfg.optimizer.betas_g,
                                      weight_decay=getattr(self.cfg.optimizer, 'weight_decay', 0.0))
        
        self.optimizer_d = OptimClass(params=self.discriminator.parameters(), 
                                      lr=self.cfg.optimizer.lr_d, 
                                      betas=self.cfg.optimizer.betas_d,
                                      weight_decay=getattr(self.cfg.optimizer, 'weight_decay', 0.0))
    
    def setup_schedulers(self) -> None:
        """Initialize learning rate scheduler."""
        self.scheduler_g = get_scheduler(self.optimizer_g, self.cfg.scheduler)
        self.scheduler_d = get_scheduler(self.optimizer_d, self.cfg.scheduler)

    def setup_criteria(self) -> None:
        self.l1_loss = L1Loss()
        self.adv_loss = PatchAdversarialLoss(criterion="least_squares")
        self.loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2)
        self.loss_perceptual.to(self.device)
    
    def set_train_mode(self, train: bool) -> None:
        """Set model to train or eval mode."""
        self.model.train(train)
    
    
    def kl_loss(self, z_mu, z_sigma):
        klloss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
        return torch.sum(klloss) / klloss.shape[0]
    
    def _apply_tumor_mask(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply binary tumor mask to tensor for masked loss computation.
        
        Args:
            tensor: Input tensor (B, C, H, W, D)
            mask: Binary mask (B, 1, H, W, D) with 1=tumor, 0=background
            
        Returns:
            Masked tensor with same shape (non-tumor regions zeroed)
        """
        return tensor * mask.float()

    def _set_discriminator_requires_grad(self, requires_grad: bool) -> None:
        for p in self.discriminator.parameters():
            p.requires_grad_(requires_grad)

    def _compute_recon_losses(
        self,
        inputs: torch.Tensor,
        reconstruction: torch.Tensor,
        tumor_mask: torch.Tensor,
        weight_k: float = 1.0,
        weight_tr: float = 10.0
        ):
        if self.use_masked_loss:
            l1_per_pixel = torch.abs(reconstruction.float() - inputs.float())
            weights = torch.full_like(tumor_mask, weight_k, dtype=torch.float32).to(inputs.device)
            weights[tumor_mask == 1] = weight_tr
            weighted_l1_tensor = l1_per_pixel * weights
            recons_loss = torch.mean(weighted_l1_tensor)
        else:
            recons_loss = self.l1_loss(reconstruction.float(), inputs.float())
        p_loss = self.loss_perceptual(reconstruction.float(), inputs.float())
        return recons_loss, p_loss

    def _forward_and_losses(
        self,
        inputs: torch.Tensor,
        tumor_mask: torch.Tensor,
        phase: Optional[torch.Tensor] = None
        ):
        reconstruction, z_mu, z_sigma = self.model(inputs, phase=phase)
        recons_loss, p_loss = self._compute_recon_losses(inputs, reconstruction, tumor_mask)
        klloss = self.kl_loss(z_mu, z_sigma)
        loss_g = recons_loss + self.kl_w * klloss + self.perc_w * p_loss
        return reconstruction, z_mu, z_sigma, recons_loss, p_loss, klloss, loss_g

    def _gen_adv_loss(self, reconstruction: torch.Tensor, epoch: int) -> torch.Tensor:
        if epoch < self.adv_warmup_epochs:
            return torch.tensor(0.0, device=reconstruction.device)
        self._set_discriminator_requires_grad(False)
        # MultiScalePatchDiscriminator returns (out_list, intermediate_features)
        out_list, _ = self.discriminator(reconstruction.contiguous().float())
        # Average adversarial loss across all scales
        adv_losses = [self.adv_loss(logits, target_is_real=True, for_discriminator=False) for logits in out_list]
        gen_adv = torch.stack(adv_losses).sum() / len(adv_losses)
        self._set_discriminator_requires_grad(True)
        return gen_adv

    def _discriminator_loss(self, inputs: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
        # MultiScalePatchDiscriminator returns (out_list, intermediate_features)
        out_fake, _ = self.discriminator(reconstruction.contiguous().detach())
        out_real, _ = self.discriminator(inputs.contiguous().detach())
        # Average loss across all scales
        loss_d_fake = sum([self.adv_loss(logits, target_is_real=False, for_discriminator=True) for logits in out_fake]) / len(out_fake)
        loss_d_real = sum([self.adv_loss(logits, target_is_real=True, for_discriminator=True) for logits in out_real]) / len(out_real)
        return self.adv_w * 0.5 * (loss_d_fake + loss_d_real)

    def _update_discriminator(self, inputs: torch.Tensor, reconstruction: torch.Tensor, epoch: int) -> torch.Tensor:
        if epoch < self.adv_warmup_epochs:
            return torch.tensor(0.0, device=inputs.device)
        self.optimizer_d.zero_grad(set_to_none=True)
        loss_d = self._discriminator_loss(inputs, reconstruction)
        loss_d.backward()
        self.optimizer_d.step()
        return loss_d

    def _store_sample(self, split, inputs, reconstruction, mask):
        rand_idx = torch.randint(0, inputs.shape[0], (1,)).item()
        self.samples[split] = {
            "input": inputs[rand_idx].detach().cpu(),
            "recon": reconstruction[rand_idx].detach().cpu(),
            "mask": mask[rand_idx].detach().cpu(),
        }
    
    @torch.no_grad()
    def _extract_embeddings(self, loader, max_batches: int):
        self.model.eval()

        E, Y = [], []
        n = 0
        for batch in loader:
            x = batch["ct"]["data"].to(self.device)
            y = [map_label_to_category(lbl) for lbl in batch["label"]]
            _, z_mu, _ = self.model(x)
            e = z_mu.mean(dim=(2,3,4))
            e = F.normalize(e, dim=1)

            E.append(e.cpu())
            Y.append(torch.as_tensor(y))

            n += 1
            if n >= max_batches:
                break

        E = torch.cat(E, dim=0).float()
        Y = torch.cat(Y, dim=0)
        return E, Y
    
    def _linear_probe_classification(self, Etr, Ytr, Eval, Yval, num_epochs=50, lr=1e-2, l2=1e-4):
        # binaria vs multiclase
        y_unique = torch.unique(Ytr)
        is_binary = (len(y_unique) == 2)

        in_dim = Etr.shape[1]
        if is_binary:
            head = torch.nn.Linear(in_dim, 1).to(self.device)
            loss_fn = torch.nn.BCEWithLogitsLoss()
            Ytr_t = Ytr.float().to(self.device).view(-1, 1)
            Yval_t = Yval.float().to(self.device).view(-1, 1)
        else:
            n_classes = int(torch.max(Ytr).item() + 1)
            head = torch.nn.Linear(in_dim, n_classes).to(self.device)
            loss_fn = torch.nn.CrossEntropyLoss()
            Ytr_t = Ytr.long().to(self.device)
            Yval_t = Yval.long().to(self.device)

        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=l2)

        Etr_d = Etr.to(self.device)
        Eval_d = Eval.to(self.device)

        head.train()
        for _ in range(num_epochs):
            opt.zero_grad(set_to_none=True)
            logits = head(Etr_d)
            loss = loss_fn(logits, Ytr_t)
            loss.backward()
            opt.step()

        head.eval()
        with torch.no_grad():
            logits = head(Eval_d)
            if is_binary:
                probs = torch.sigmoid(logits).squeeze(1).cpu()
                preds = (probs > 0.5).long()
                acc = (preds == Yval.long()).float().mean().item()
                return {"probe_acc": acc}
            else:
                preds = torch.argmax(logits, dim=1).cpu()
                acc = (preds == Yval.long()).float().mean().item()
                return {"probe_acc": acc}

    def train_step(self, epoch, batch: Any) -> Dict[str, float]:
        inputs = batch["ct"]["data"].to(self.device)
        phase = batch["phase"].to(self.device)
        multilabel_mask = batch['mask']['data'].to(self.device)
        # Extract binary tumor mask: label==2
        tumor_mask = (multilabel_mask == 2).float()
        # Verify CT and mask patches have matching shapes
        assert inputs.shape == tumor_mask.shape, \
            f"CT and tumor mask patch shape mismatch: {inputs.shape} vs {tumor_mask.shape}"
        
        if epoch == self.adv_warmup_epochs :
            OptimClass = getattr(optim, self.cfg.optimizer.name)
            self.optimizer_g = OptimClass(params=self.model.parameters(), 
                                      lr=self.cfg.optimizer.lr_g, 
                                      betas=[0.0, 0.999],
                                      weight_decay=getattr(self.cfg.optimizer, 'weight_decay', 0.0))

        self.optimizer_g.zero_grad(set_to_none=True)

        reconstruction, z_mu, z_sigma, recons_loss, p_loss, klloss, loss_g = self._forward_and_losses(inputs, tumor_mask, phase=phase)

        gen_adv = self._gen_adv_loss(reconstruction, epoch)

        loss_g.backward()
        self.optimizer_g.step()
        # --------------------
        # Discriminator update
        # --------------------
        loss_d = self._update_discriminator(inputs, reconstruction, epoch)
        
        # Collect latents for UMAP (only z_mu, not variance)
        if self.collect_latents and hasattr(self, '_collect_this_epoch'):
            if self._collect_this_epoch:
                self.latent_collection['train'].append(z_mu.detach().cpu().numpy())
                labels = [map_label_to_category(lbl) for lbl in batch['label']]
                self.label_collection['train'].append(np.array(labels))
    
        # Store random training sample for visualization
        if self.collect_img:
            self._store_sample("train", inputs, reconstruction, multilabel_mask)

        metrics = {
            "loss": loss_g.item(),
            "loss_d": loss_d.item(),
            "recon": recons_loss.item(),
            "kl": klloss.item() * self.kl_w,
            "perceptual": p_loss.item(),
            "g_adv": gen_adv.item(),
        }
        metrics.update(self._compute_quality_metrics(reconstruction, inputs))
        return metrics

    def validation_step(self, epoch, batch: Any) -> Dict[str, float]:
        inputs = batch["ct"]["data"].to(self.device)
        phase = batch["phase"].to(self.device)
        multilabel_mask = batch['mask']['data'].to(self.device)
        # Extract binary tumor mask: label==2
        tumor_mask = (multilabel_mask == 2).float()
        # Verify CT and mask patches have matching shapes
        assert inputs.shape == tumor_mask.shape, \
            f"CT and tumor mask patch shape mismatch: {inputs.shape} vs {tumor_mask.shape}"
        
        self.model.eval()
        # self.discriminator.eval()

        with torch.no_grad():
            reconstruction, z_mu, z_sigma, recons_loss, p_loss, klloss, loss_g = self._forward_and_losses(inputs, tumor_mask, phase=phase)

            gen_adv = self._gen_adv_loss(reconstruction, epoch)
            loss_d = self._discriminator_loss(inputs, reconstruction) if epoch > self.adv_warmup_epochs else torch.tensor(0.0, device=inputs.device)

            metrics = {
                "loss": loss_g.item(),
                "loss_d": loss_d.item(),
                "recon": recons_loss.item(),
                "kl": klloss.item() * self.kl_w,
                "perceptual": p_loss.item(),
                "g_adv": gen_adv.item(),
            }
            metrics.update(self._compute_quality_metrics(reconstruction, inputs))
            
            # Logging samples for visualization
            if self.collect_img:
                self._store_sample("val", inputs, reconstruction, multilabel_mask)


            if self.collect_latents and hasattr(self, "_collect_this_epoch") and self._collect_this_epoch:
                self.latent_collection["val"].append(z_mu.detach().cpu().numpy())
                labels = [map_label_to_category(lbl) for lbl in batch["label"]]
                self.label_collection["val"].append(np.array(labels))
                    
        return metrics

    
    def _initialize_epoch_metrics(self) -> Dict[str, float]:
        """Initialize metrics dictionary for epoch."""
        metrics = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "PSNR_count": 0}
        for metric_name in self.metrics:
            metrics[metric_name] = 0.0
        return metrics
    
    def _average_epoch_metrics(self, epoch_metrics: Dict[str, float], n_batches: int) -> Dict[str, float]:
        """Average accumulated metrics over the epoch."""
        averaged_metrics = {k: v / max(1, n_batches) for k, v in epoch_metrics.items() if k != 'PSNR_count'}
        if 'PSNR' in epoch_metrics and epoch_metrics['PSNR_count'] > 0:
            averaged_metrics['PSNR'] = epoch_metrics['PSNR'] / epoch_metrics['PSNR_count']
        return averaged_metrics

    def step_schedulers(self) -> None:
        """Step LR schedulers for G and D (epoch-based)."""
        self.scheduler_g.step()
        self.scheduler_d.step()
    
    def on_fit_start(self) -> None:
        """Hook called before training starts."""
        super().on_fit_start()
        self._collect_this_epoch = False
        self.collect_img = False
        
    def on_epoch_start(self, epoch: int, is_train: bool) -> None:
        """Hook called at the start of each epoch."""
        # Determine if we should collect latents this epoch
        umap_interval = getattr(self.cfg.trainer, 'umap_log_interval', 10)
        self._collect_this_epoch = self.collect_latents and (epoch % umap_interval == 0)
        image_log_interval = getattr(self.cfg.trainer, 'image_log_interval', 5)

        self.collect_img = (epoch % image_log_interval == 0)
        
        # Clear previous collections if starting new collection
        if self._collect_this_epoch:
            split = 'train' if is_train else 'val'
            self.latent_collection[split] = []
            self.label_collection[split] = []
    
    def on_epoch_end(
        self, 
        epoch: int, 
        train_metrics: Dict[str, float], 
        val_metrics: Dict[str, float]
    ) -> None:
        """Hook for epoch end - log images, UMAP, and call callbacks."""
        # Log sample images every N epochs
        image_log_interval = getattr(self.cfg.trainer, 'image_log_interval', 5)
        if epoch % image_log_interval == 0:
            if self.samples["val"]:
                self._log_sample_images(epoch, tag="val")
            if self.samples["train"]:
                self._log_sample_images(epoch, tag="train")
        
        # Log UMAP if latents were collected
        if self._collect_this_epoch and self.collect_latents:
            self._log_latent_umap(epoch)
        
        # ---- PROBE ----
        probe_interval = getattr(self.cfg.trainer, "probe_interval", 0)
        if probe_interval and (epoch % probe_interval == 0):
            max_batches = getattr(self.cfg.trainer, "probe_max_batches", 20)
            probe_epochs = getattr(self.cfg.trainer, "probe_epochs", 50)
            probe_lr = getattr(self.cfg.trainer, "probe_lr", 1e-2)
            probe_l2 = getattr(self.cfg.trainer, "probe_l2", 1e-4)

            Etr, Ytr = self._extract_embeddings(self.dataloaders["train"], max_batches)
            Eval, Yval = self._extract_embeddings(self.dataloaders["val"], max_batches)

            probe_metrics = self._linear_probe_classification(Etr, Ytr, Eval, Yval,
                                                            num_epochs=probe_epochs, lr=probe_lr, l2=probe_l2)

            for k, v in probe_metrics.items():
                self.logger.add_scalar(f"probe_{k}", v, epoch)


        # callbacks base
        super().on_epoch_end(epoch, train_metrics, val_metrics)
    
    def _log_latent_umap(self, epoch: int) -> None:
        """Log UMAP visualization of latent space."""
        for split in ['train', 'val']:
            if len(self.latent_collection[split]) > 0:
                latents = self.latent_collection[split]
                labels = self.label_collection[split] if len(self.label_collection[split]) > 0 else None
                
                self.logger.log_latent_umap(
                    latents=latents,
                    labels=labels,
                    step=epoch,
                    tag=split
                )
                
                # Clear after logging to save memory
                self.latent_collection[split] = []
                self.label_collection[split] = []
    
    def _log_sample_images(self, epoch: int, tag: str = "val") -> None:
        sample = self.samples.get(tag, {})
        if not sample:
            return
        self.logger.log_axial_figure(
            sample["input"],
            sample["recon"],
            step=epoch,
            tag=tag,
            norm_stats=self.norm_stats,
            mask=sample["mask"],
        )


