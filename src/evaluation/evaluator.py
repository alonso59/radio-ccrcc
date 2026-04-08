"""
Model evaluator for k-fold cross-validation.
Handles train/val/test evaluation with memory-efficient test loading.
"""

import torch
import numpy as np
from typing import Dict, Optional, Any
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

from .metrics import MetricsCalculator

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluates model performance on train, validation, and test sets.
    """
    
    def __init__(
        self,
        device: torch.device,
        l1_loss: Any,
        perceptual_loss: Any,
        normalization_stats: Optional[Dict[str, float]] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            device: Torch device (cuda/cpu)
            l1_loss: L1 loss function for reconstruction
            perceptual_loss: Perceptual loss function
            normalization_stats: Optional dict with windowed intensity statistics (mean, std, window_min, window_max)
        """
        self.device = device
        self.l1_loss = l1_loss
        self.perceptual_loss = perceptual_loss
        self.norm_stats = normalization_stats or {}
        self.metrics_calc = MetricsCalculator(normalization_stats=normalization_stats)
    
    def kl_loss(self, z_mu, z_sigma):
        """Compute KL divergence matching trainer.py implementation."""
        klloss = 0.5 * torch.sum(
            z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1,
            dim=[1, 2, 3, 4]
        )
        return torch.sum(klloss) / klloss.shape[0]
    
    def evaluate_loader(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        split_name: str = "eval",
        kl_weight: float = 1.0,
        perceptual_weight: float = 1.0
    ) -> Dict[str, float]:
        """
        Evaluate model on a single dataloader.
        
        Args:
            model: PyTorch model to evaluate
            dataloader: DataLoader for evaluation
            split_name: Name of split for logging (train/val/test)
            kl_weight: Weight for KL divergence
            perceptual_weight: Weight for perceptual loss
            
        Returns:
            Dictionary of metric name -> mean value
        """
        model.eval()
        results = {
            'recon_loss': [],
            'kl_loss': [],
            'perceptual_loss': [],
            'total_loss': [],
            'PSNR': [],
            'SSIM': []
        }
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {split_name}", ncols=100, ascii=True):
                try:
                    # Get input data
                    inputs = batch['ct']['data'].to(self.device)
                    
                    # Forward pass
                    reconstruction, z_mu, z_sigma = model(inputs)
                    
                    # Compute loss components matching trainer.py
                    recon_loss = self.l1_loss(reconstruction.float(), inputs.float())
                    p_loss = self.perceptual_loss(reconstruction.float(), inputs.float())
                    kl_loss = self.kl_loss(z_mu, z_sigma)
                    
                    # Total loss (matching trainer.py computation)
                    total_loss = recon_loss + kl_weight * kl_loss + perceptual_weight * p_loss
                    
                    # Store loss values
                    results['recon_loss'].append(recon_loss.item())
                    results['kl_loss'].append((kl_weight * kl_loss).item())
                    results['perceptual_loss'].append(p_loss.item())
                    results['total_loss'].append(total_loss.item())
                    
                    # Compute additional metrics
                    batch_metrics = self.metrics_calc.compute_batch_metrics(
                        reconstruction, inputs, data_range=1.0
                    )
                    
                    for metric_name, metric_value in batch_metrics.items():
                        results[metric_name].append(metric_value)
                        
                except Exception as e:
                    logger.error(f"Error evaluating batch in {split_name}: {e}")
                    continue
        
        # Aggregate results
        aggregated = {k: float(np.mean(v)) for k, v in results.items() if len(v) > 0}
        
        logger.info(f"{split_name.upper()} - Loss: {aggregated.get('total_loss', float('nan')):.4f}, "
                   f"PSNR: {aggregated.get('PSNR', float('nan')):.2f}, "
                   f"SSIM: {aggregated.get('SSIM', float('nan')):.4f}")
        
        return aggregated
    
    def evaluate_fold(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        fold_id: int = 0,
        kl_weight: float = 1.0,
        perceptual_weight: float = 1.0
    ) -> Dict[str, Any]:
        """
        Evaluate a single fold with train, val, and optionally test sets.
        
        Args:
            model: PyTorch model to evaluate
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Optional test data loader (created lazily)
            fold_id: Fold identifier
            kl_weight: Weight for KL divergence
            perceptual_weight: Weight for perceptual loss
            
        Returns:
            Dictionary with fold results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating Fold {fold_id}")
        logger.info(f"{'='*60}")
        
        # Evaluate train set
        train_metrics = self.evaluate_loader(model, train_loader, "train", kl_weight, perceptual_weight)
        
        # Evaluate validation set
        val_metrics = self.evaluate_loader(model, val_loader, "val", kl_weight, perceptual_weight)
        
        fold_result = {
            "fold": fold_id,
            "train": train_metrics,
            "val": val_metrics
        }
        
        # Evaluate test set if provided (lazy loading)
        if test_loader is not None:
            test_metrics = self.evaluate_loader(model, test_loader, "test", kl_weight, perceptual_weight)
            fold_result["test"] = test_metrics
            
            # Clean up test loader to free memory
            del test_loader
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return fold_result
    
    def aggregate_folds(self, fold_results: list) -> Dict[str, Any]:
        """
        Aggregate results across all folds.
        
        Args:
            fold_results: List of fold result dictionaries
            
        Returns:
            Aggregated statistics (mean, std) for each split
        """
        if not fold_results:
            logger.warning("No fold results to aggregate")
            return {}
        
        aggregated: Dict[str, Any] = {"folds": fold_results}
        
        # Determine which splits are available
        splits = ['train', 'val']
        if 'test' in fold_results[0]:
            splits.append('test')
        
        # Aggregate each split
        for split in splits:
            all_metrics = {}
            for fold in fold_results:
                if split in fold:
                    for metric_name, value in fold[split].items():
                        if metric_name not in all_metrics:
                            all_metrics[metric_name] = []
                        all_metrics[metric_name].append(value)
            
            # Compute mean and std
            aggregated[f"mean_{split}"] = {
                k: float(np.mean(v)) for k, v in all_metrics.items()
            }
            aggregated[f"std_{split}"] = {
                k: float(np.std(v)) for k, v in all_metrics.items()
            }
        
        return aggregated
