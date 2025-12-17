# loss.py
from __future__ import annotations
from typing import Callable, Optional, Tuple, Union, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses.perceptual import PerceptualLoss

TensorOrList = Union[torch.Tensor, Sequence[torch.Tensor]]

# ---------------------------
# Autoencoder (recon + KL)
# ---------------------------

class ELBOLoss(nn.Module):
    """
    MSE reconstruction + optional KL (beta scaled).
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        beta: float = 1.0
    ):

        recon = F.mse_loss(recon_x, x, reduction="mean")

        logvar = 2.0 * torch.log(sigma.clamp(min=1e-8))
        
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon, beta * kl

class Perceptual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.perceptual = PerceptualLoss(
            spatial_dims=3, 
            network_type='medicalnet_resnet50_23datasets',
            is_fake_3d=False,
            pretrained=True,
        )
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        max_val = max(x.abs().max().item(), y.abs().max().item())
        clip_val = max(max_val, 3.0)
        
        x = x.clamp(-clip_val, clip_val) / clip_val
        y = y.clamp(-clip_val, clip_val) / clip_val
        return self.perceptual(x, y)
