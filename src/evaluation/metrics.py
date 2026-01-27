"""
Metrics computation for model evaluation.
Includes PSNR, SSIM, and loss calculations.
"""

import torch
import numpy as np
from pytorch_msssim import ssim
from typing import Dict, Callable, Any
import logging

logger = logging.getLogger(__name__)


def psnr_torch(x_hat: torch.Tensor, x: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between reconstructed and original images.
    
    Args:
        x_hat: Reconstructed image tensor
        x: Original image tensor
        data_range: Maximum possible pixel value range
        
    Returns:
        PSNR value in dB
    """
    mse = torch.mean((x_hat - x) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'), device=x_hat.device)
    psnr = 20 * torch.log10(torch.tensor(data_range, device=x_hat.device)) - 10 * torch.log10(mse)
    return psnr


def ssim_torch(x_hat: torch.Tensor, x: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """
    Compute Structural Similarity Index (SSIM) between reconstructed and original images.
    
    Args:
        x_hat: Reconstructed image tensor
        x: Original image tensor  
        data_range: Maximum possible pixel value range
        
    Returns:
        SSIM value (0-1, higher is better)
    """
    return ssim(x_hat, x, data_range=data_range)


class MetricsCalculator:
    """
    Centralized metrics calculation for evaluation.
    """
    
    def __init__(self):
        self.metrics_functions: Dict[str, Callable] = {
            "PSNR": psnr_torch,
            "SSIM": ssim_torch
        }
        
    def compute_batch_metrics(
        self,
        reconstruction: torch.Tensor,
        original: torch.Tensor,
        data_range: float = 1.0
    ) -> Dict[str, float]:
        """
        Compute all metrics for a single batch.
        
        Args:
            reconstruction: Reconstructed tensor
            original: Original tensor
            data_range: Data range for metrics
            
        Returns:
            Dictionary of metric name -> value
        """
        results = {}
        for metric_name, metric_fn in self.metrics_functions.items():
            try:
                if metric_name in ["PSNR", "SSIM"]:
                    value = metric_fn(reconstruction, original, data_range)
                else:
                    value = metric_fn(reconstruction, original)
                    
                if hasattr(value, 'item'):
                    value = value.item()
                results[metric_name] = float(value)
            except Exception as e:
                logger.warning(f"Failed to compute {metric_name}: {e}")
                results[metric_name] = float('nan')
                
        return results
    
    def aggregate_results(self, all_results: Dict[str, list]) -> Dict[str, float]:
        """
        Aggregate batch-wise results into mean values.
        
        Args:
            all_results: Dictionary of metric name -> list of batch values
            
        Returns:
            Dictionary of metric name -> mean value
        """
        return {
            metric_name: float(np.mean(values))
            for metric_name, values in all_results.items()
        }
