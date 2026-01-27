"""
Pure Autoencoder Trainer (VAE/AE without adversarial component).
Handles reconstruction + KL divergence loss.
"""
from typing import Dict, Any, Optional, Callable
import numpy as np
import torch
import torch.optim as optim
from omegaconf import DictConfig


from .base_trainer import BaseTrainer
from .loss import ELBOLoss
from ..dataloader.dataloader import map_label_to_category
from ..utils.scheduler import get_scheduler

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
        self.optimizer = OptimClass(
            self.model.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=getattr(self.cfg.optimizer, 'weight_decay', 0.0),
            betas=tuple(getattr(self.cfg.optimizer, 'betas', (0.9, 0.999))),
        )
    
    def setup_schedulers(self) -> None:
        """Initialize learning rate scheduler."""
        self.scheduler = get_scheduler(self.optimizer, self.cfg.scheduler)
    
    def setup_criteria(self) -> None:
        """Initialize loss function."""
        print("beta KL:", self.cfg.loss_weights.kl)
        self.criterion = ELBOLoss().to(self.device)

    def set_train_mode(self, train: bool) -> None:
        """Set model to train or eval mode."""
        self.model.train(train)
    
    def train_step(self, batch: Any) -> Dict[str, float]:
        """Execute one training step."""
        inputs = batch['ct']['data'].to(self.device)
        # mask = batch['mask']['data'].to(self.device)
        # Forward pass
        self.optimizer.zero_grad()
        reconstruction, z_mu, z_sigma = self.model(inputs)
        
        # Collect latents for UMAP (only z_mu, not variance)
        if self.collect_latents and hasattr(self, '_collect_this_epoch'):
            if self._collect_this_epoch:
                self.latent_collection['train'].append(z_mu.detach().cpu().numpy())
                if 'label' in batch:
                    # batch['label'] is a list of strings, convert to integers
                    labels = [map_label_to_category(lbl) for lbl in batch['label']]
                    self.label_collection['train'].append(np.array(labels))
        
        # Get current beta
        current_beta = self.cfg.loss_weights.kl
        
        recon, kl = self.criterion(
            reconstruction, 
            inputs, 
            mu=z_mu, 
            sigma=z_sigma,
            beta=current_beta
        )
        loss = recon # + kl
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Compute metrics
        metrics = {
            "loss": loss.item(),
            "recon": recon.item(),
            "kl": kl.item()
        }
        metrics.update(self._compute_quality_metrics(reconstruction, inputs))
        print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        return metrics
    
    def validation_step(self, batch: Any) -> Dict[str, float]:
        """Execute one validation step."""
        inputs = batch['ct']['data'].to(self.device)
        
        # mask = batch['mask']['data'].to(self.device)
        with torch.no_grad():
            # Forward pass
            reconstruction, z_mu, z_sigma = self.model(inputs)
            
            if self.collect_img:
                self.random_idx = torch.randint(0, inputs.shape[0], (1,)).item()
                self.rand_input = inputs[self.random_idx].detach().cpu()
                self.rand_recon = reconstruction[self.random_idx].detach().cpu()
            # Collect latents for UMAP
            if self.collect_latents and hasattr(self, '_collect_this_epoch'):
                if self._collect_this_epoch:
                    self.latent_collection['val'].append(z_mu.detach().cpu().numpy())
                    if 'label' in batch:
                        # batch['label'] is a list of strings, convert to integers
                        labels = [map_label_to_category(lbl) for lbl in batch['label']]
                        self.label_collection['val'].append(np.array(labels))
            
            # Compute loss
            current_beta = self.cfg.loss_weights.kl
            
            recon, kl = self.criterion(
            reconstruction, 
            inputs,
            mu=z_mu,
            sigma=z_sigma,
            beta=current_beta
        )
            loss = recon # + kl
            # Compute metrics
            metrics = {
                "loss": loss.item(),
                "recon": recon.item(),
                "kl": kl.item()
            }
            metrics.update(self._compute_quality_metrics(reconstruction, inputs))
        self.collect_img = False  # Only collect once per epoch
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
        """Step the learning rate scheduler."""
        if self.scheduler is not None:
            # ReduceLROnPlateau needs metrics, others don't
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # Will be called from on_epoch_end with val metrics
                pass
            else:
                self.scheduler.step()
    
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
            self._log_sample_images(self.rand_input, self.rand_recon, epoch)
        
        # Log UMAP if latents were collected
        if self._collect_this_epoch and self.collect_latents:
            self._log_latent_umap(epoch)
        
        # Step ReduceLROnPlateau if needed
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            monitor_metric = getattr(self.cfg.trainer, 'monitor_metric', 'loss')
            metric_value = val_metrics.get(monitor_metric, val_metrics['loss'])
            self.scheduler.step(metric_value)
        
        # Call parent implementation for callbacks
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
    
    def _log_sample_images(self, rand_input: torch.Tensor, rand_recon: torch.Tensor, epoch: int, tag: str="val") -> None:
        self.logger.log_axial_figure(
            rand_input,
            rand_recon, 
            step=epoch, 
            tag=tag,
            norm_stats=self.norm_stats
        )


