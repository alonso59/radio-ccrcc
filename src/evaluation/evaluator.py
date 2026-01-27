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
    
    def __init__(self, device: torch.device, criterion: Any):
        """
        Initialize evaluator.
        
        Args:
            device: Torch device (cuda/cpu)
            criterion: Loss function (e.g., ELBOLoss)
        """
        self.device = device
        self.criterion = criterion
        self.metrics_calc = MetricsCalculator()
        
    def evaluate_loader(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        split_name: str = "eval",
        beta: float = 1.0
    ) -> Dict[str, float]:
        """
        Evaluate model on a single dataloader.
        
        Args:
            model: PyTorch model to evaluate
            dataloader: DataLoader for evaluation
            split_name: Name of split for logging (train/val/test)
            beta: Beta weight for KL divergence in loss
            
        Returns:
            Dictionary of metric name -> mean value
        """
        model.eval()
        results = {
            'recon_loss': [],
            'kl_loss': [],
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
                    
                    # Compute loss components
                    recon_loss, kl_loss = self.criterion(
                        reconstruction, inputs, 
                        mu=z_mu, 
                        sigma=z_sigma, 
                        beta=beta
                    )
                    
                    total_loss = recon_loss + kl_loss
                    
                    # Store loss values
                    results['recon_loss'].append(recon_loss.item())
                    results['kl_loss'].append(kl_loss.item())
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
        beta: float = 1.0
    ) -> Dict[str, Any]:
        """
        Evaluate a single fold with train, val, and optionally test sets.
        
        Args:
            model: PyTorch model to evaluate
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Optional test data loader (created lazily)
            fold_id: Fold identifier
            beta: Beta weight for KL divergence
            
        Returns:
            Dictionary with fold results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating Fold {fold_id}")
        logger.info(f"{'='*60}")
        
        # Evaluate train set
        train_metrics = self.evaluate_loader(model, train_loader, "train", beta)
        
        # Evaluate validation set
        val_metrics = self.evaluate_loader(model, val_loader, "val", beta)
        
        fold_result = {
            "fold": fold_id,
            "train": train_metrics,
            "val": val_metrics
        }
        
        # Evaluate test set if provided (lazy loading)
        if test_loader is not None:
            test_metrics = self.evaluate_loader(model, test_loader, "test", beta)
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
