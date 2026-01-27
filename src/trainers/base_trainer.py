"""
Base Trainer following SOLID principles.
- Single Responsibility: Each trainer handles one type of training
- Open/Closed: Extensible for new training strategies without modifying base
- Liskov Substitution: All trainers can be used interchangeably
- Interface Segregation: Clean interface for training lifecycle
- Dependency Inversion: Depends on abstractions (callbacks, loggers)
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
import time
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from pytorch_msssim import ssim
import torch.nn.functional as F
import logging

class BaseTrainer(ABC):
    """
    Abstract base trainer defining the training lifecycle.
    All concrete trainers should inherit from this class.
    """
    
    def __init__(
        self,
        cfg: DictConfig,
        dataloaders: Dict[str, Any],
        device: torch.device,
        logger: Any,
        callbacks: Optional[List[Any]] = None,
        max_epochs: Optional[int] = None,
        normalization_stats: Optional[Dict[str, float]] = None,
    ):
        self.cfg = cfg
        self.dataloaders = dataloaders
        self.device = device
        self.logger = logger
        self.callbacks = callbacks or []
        self.max_epochs = int(max_epochs if max_epochs is not None else cfg.trainer.max_epochs)
        self.norm_stats = normalization_stats or {}  # Store normalization statistics
        self.global_step = 0
        self.best_val_metric = float('inf')
        self.metrics = {
            "PSNR": self._psnr_torch,
            "SSIM": self._ssim_torch
        }
        
    @abstractmethod
    def setup_models(self) -> None:
        """Initialize and configure models."""
        pass
    
    @abstractmethod
    def setup_optimizers(self) -> None:
        """Initialize optimizers."""
        pass
    
    @abstractmethod
    def setup_schedulers(self) -> None:
        """Initialize learning rate schedulers."""
        pass
    
    @abstractmethod
    def setup_criteria(self) -> None:
        """Initialize loss functions."""
        pass
    
    @abstractmethod
    def train_step(self, batch: Any) -> Dict[str, float]:
        """
        Execute one training step.
        
        Args:
            batch: Input batch from dataloader
            
        Returns:
            Dictionary of metrics for this step
        """
        pass
    
    @abstractmethod
    def validation_step(self, batch: Any) -> Dict[str, float]:
        """
        Execute one validation step.
        
        Args:
            batch: Input batch from dataloader
            
        Returns:
            Dictionary of metrics for this step
        """
        pass
    
    def fit(self) -> None:
        """Main training loop - Template Method pattern."""
        self.on_fit_start()
        
        for epoch in range(self.max_epochs):
            start_time = time.time()
            
            # Training epoch
            train_metrics = self._run_epoch(
                self.dataloaders['train'], 
                epoch, 
                train=True
            )
            
            # Validation epoch
            val_metrics = self._run_epoch(
                self.dataloaders['val'], 
                epoch, 
                train=False
            )
            
            # Logging
            self._log_epoch_metrics(epoch, train_metrics, val_metrics)
            
            # Epoch end callbacks
            self.on_epoch_end(epoch, train_metrics, val_metrics)
            
            # Scheduler step
            self.step_schedulers()
            
            # Print summary
            self._print_epoch_summary(epoch, train_metrics, val_metrics, time.time() - start_time)
            
            # Check for best model
            self._check_best_model(epoch, val_metrics)
        
        self.on_fit_end()
    
    def _run_epoch(
        self, 
        dataloader: Any, 
        epoch: int, 
        train: bool = True
    ) -> Dict[str, float]:
        """
        Run one epoch of training or validation.
        
        Args:
            dataloader: DataLoader for this epoch
            epoch: Current epoch number
            train: Whether this is training (True) or validation (False)
            
        Returns:
            Dictionary of aggregated metrics for the epoch
        """
        mode = "train" if train else "val"
        self.set_train_mode(train)
        
        # Call epoch start hook
        self.on_epoch_start(epoch, train)
        
        epoch_metrics = self._initialize_epoch_metrics()
        n_batches = len(dataloader)
        
        with torch.set_grad_enabled(train):
            for batch in tqdm(
                dataloader, 
                desc=f"{mode.capitalize()} Epoch {epoch}", 
                ncols=100, 
                ascii=True
            ):
                # Execute step
                if train:
                    step_metrics = self.train_step(batch)
                else:
                    step_metrics = self.validation_step(batch)
                
                # Accumulate metrics
                for key, value in step_metrics.items():
                    if key in epoch_metrics:
                        epoch_metrics[key] += value
                    else:
                        epoch_metrics[key] = value
                
                # Log step metrics
                self._log_step_metrics(mode, step_metrics)
                self.global_step += 1
        
        # Average metrics
        epoch_metrics = self._average_epoch_metrics(epoch_metrics, n_batches)
        
        return epoch_metrics
    
    @abstractmethod
    def _initialize_epoch_metrics(self) -> Dict[str, float]:
        """Initialize metrics dictionary for epoch."""
        pass
    
    @abstractmethod
    def set_train_mode(self, train: bool) -> None:
        """Set models to train or eval mode."""
        pass
    
    def step_schedulers(self) -> None:
        """Step learning rate schedulers. Override if needed."""
        pass
    
    def on_fit_start(self) -> None:
        """Hook called before training starts."""
        pass
    
    def on_fit_end(self) -> None:
        """Hook called after training ends."""
        for cb in self.callbacks:
            if hasattr(cb, 'on_fit_end'):
                cb.on_fit_end()
    
    def on_epoch_start(self, epoch: int, is_train: bool) -> None:
        """Hook called at the start of each epoch."""
        pass
    
    def on_epoch_end(
        self, 
        epoch: int, 
        train_metrics: Dict[str, float], 
        val_metrics: Dict[str, float]
    ) -> None:
        """Hook called at the end of each epoch."""
        for cb in self.callbacks:
            if hasattr(cb, 'on_epoch_end'):
                cb.on_epoch_end(
                    epoch, 
                    train_metrics.get("loss"), 
                    val_metrics.get("loss")
                )
    
    def _log_epoch_metrics(
        self, 
        epoch: int, 
        train_metrics: Dict[str, float], 
        val_metrics: Dict[str, float]
    ) -> None:
        """Log epoch-level metrics."""
        for key in train_metrics:
            if key in val_metrics:
                self.logger.log_scalars(
                    key.capitalize(), 
                    {"train": train_metrics[key], "val": val_metrics[key]}, 
                    epoch
                )
    
    def _log_step_metrics(self, mode: str, metrics: Dict[str, float]) -> None:
        """Log step-level metrics."""
        for key, value in metrics.items():
            self.logger.add_scalar(f"step_{mode}_{key}", value, self.global_step)
    
    def _average_epoch_metrics(
        self, 
        epoch_metrics: Dict[str, float], 
        n_batches: int
    ) -> Dict[str, float]:
        """Average accumulated metrics over the epoch."""
        return {k: v / max(1, n_batches) for k, v in epoch_metrics.items()}
    
    def _print_epoch_summary(
        self, 
        epoch: int, 
        train_metrics: Dict[str, float], 
        val_metrics: Dict[str, float], 
        epoch_time: float
    ) -> None:
        """Print summary of epoch results."""
        logging.info(f"\nEpoch {epoch} | Time: {epoch_time:.2f}s")
        logging.info(f"Train metrics: {self._format_metrics(train_metrics)}")
        logging.info(f"Val metrics:   {self._format_metrics(val_metrics)}")
    
    def _format_metrics(self, metrics: Dict[str, float], max_items: int = 5) -> str:
        """Format metrics dictionary for printing."""
        items = list(metrics.items())[:max_items]
        return ", ".join([f"{k}={v:.4f}" for k, v in items])
    
    def _check_best_model(self, epoch: int, val_metrics: Dict[str, float]) -> None:
        """Check if current model is the best and update tracking."""
        monitor_metric = getattr(self.cfg.trainer, 'monitor_metric', 'loss')
        current_val = val_metrics.get(monitor_metric, float('inf'))
        
        if current_val < self.best_val_metric:
            self.best_val_metric = current_val
            logging.info(f"✓ New best {monitor_metric}: {self.best_val_metric:.4f}")

    def _denormalize_to_hu(self, x_iqr: torch.Tensor) -> torch.Tensor:
        """
        Convert IQR-normalized data back to HU space.
        
        Args:
            x_iqr: IQR-normalized tensor (range ~[-3, 3])
        
        Returns:
            Tensor in HU space
        """
        if not self.norm_stats:
            return x_iqr  # Fallback: no denormalization if stats unavailable
        
        median = self.norm_stats['median']
        iqr = self.norm_stats['iqr']
        return x_iqr * iqr + median

    def _psnr_torch(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Compute PSNR in HU space (or IQR space as fallback).
        
        Medical imaging standard: compute in HU for clinical interpretability
        and literature comparability.
        
        Args:
            img1: First image tensor
            img2: Second image tensor
        
        Returns:
            PSNR value in dB
        """
        from src.dataloader.augmentations import P_LOW, P_HIGH
        
        # if self.norm_stats:
        #     # Convert to HU space for clinical relevance
        #     img1_hu = self._denormalize_to_hu(img1).clamp(P_LOW, P_HIGH)
        #     img2_hu = self._denormalize_to_hu(img2).clamp(P_LOW, P_HIGH)
        #     data_range = float(P_HIGH - P_LOW)  # 500 HU
        #     mse = F.mse_loss(img1_hu, img2_hu)
        # else:
        # Fallback: IQR-normalized space
        clip_val = 3.0
        img1 = img1.clamp(-clip_val, clip_val)
        img2 = img2.clamp(-clip_val, clip_val)
        data_range = 2 * clip_val  # 6.0
        mse = F.mse_loss(img1, img2)
        
        if mse == 0:
            return torch.tensor(float('inf'))
        
        return 10 * torch.log10(data_range**2 / mse)
    
    def _ssim_torch(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        reduction: str = "mean"
    ) -> torch.Tensor:
        """
        Compute SSIM in HU space (or IQR space as fallback).
        
        Medical imaging standard: compute in HU for clinical interpretability
        and literature comparability.
        
        Args:
            x: Original image tensor
            x_hat: Reconstructed image tensor
            win_size: Window size for SSIM computation
            win_sigma: Gaussian window standard deviation
            reduction: Reduction method ('mean' or 'sum')
        
        Returns:
            SSIM value in range [0, 1]
        """
        from src.dataloader.augmentations import P_LOW, P_HIGH
        
        # if self.norm_stats:
        #     # Convert to HU space
        #     x_hu = self._denormalize_to_hu(x).clamp(P_LOW, P_HIGH)
        #     x_hat_hu = self._denormalize_to_hu(x_hat).clamp(P_LOW, P_HIGH)
            
        #     # Normalize to [0, 1] for SSIM
        #     data_range_hu = float(P_HIGH - P_LOW)
        #     x_norm = (x_hu - P_LOW) / data_range_hu
        #     x_hat_norm = (x_hat_hu - P_LOW) / data_range_hu
        #     ssim_data_range = 1.0
        # else:
        # Fallback: IQR-normalized space
        clip_val = 3.0
        x_hat = x_hat.clamp(-clip_val, clip_val)
        x = x.clamp(-clip_val, clip_val)
        
        # Map [-3, 3] to [0, 1]
        x_norm = (x + clip_val) / (2 * clip_val)
        x_hat_norm = (x_hat + clip_val) / (2 * clip_val)
        ssim_data_range = 1.0
        
        size_avg = (reduction == "mean")
        
        return ssim(
            x_norm, x_hat_norm,
            data_range=ssim_data_range
        )
    
    def _compute_quality_metrics(self, x_recon: torch.Tensor, x: torch.Tensor) -> Dict[str, float]:
        """
        Compute quality metrics (PSNR, SSIM) for reconstruction.
        
        Metrics are computed in HU space if normalization stats available,
        otherwise falls back to IQR-normalized space.
        
        Args:
            x_recon: Reconstructed images
            x: Original images
        
        Returns:
            Dictionary with metrics including PSNR_count for proper averaging
        """
        metrics = {}
        
        
        with torch.no_grad():
            for metric_name, metric_fn in self.metrics.items():
                metric_value = metric_fn(x_recon, x)
                metrics[metric_name] = float(metric_value.item() if hasattr(metric_value, 'item') else metric_value)
            
            # Add PSNR count for proper epoch averaging
            if "PSNR" in metrics:
                metrics["PSNR_count"] = 1
        
        return metrics
