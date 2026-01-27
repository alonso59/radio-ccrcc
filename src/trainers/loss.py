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
        beta: float = 0.1 
    ):

        recon = F.mse_loss(recon_x, x, reduction="mean")

        logvar = 2.0 * torch.log(sigma.clamp(min=1e-8))
        
        # KL divergence: sum over all latent dimensions [C,H,W,D], then mean over batch
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1,2,3,4])
        kl = torch.mean(kl)

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


class DirichletELBOLoss(nn.Module):
    """
    DirVAE ELBO:
    - MSE reconstruction loss
    - Beta-scaled Dirichlet KL
    """

    def __init__(self, alpha_prior: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.alpha_prior = alpha_prior
        self.eps = eps

    def kl_dirichlet(self, alpha_q: torch.Tensor):
        """
        KL( Dir(alpha_q) || Dir(alpha_prior) )
        alpha_q: (B, K)
        """
        alpha_q = alpha_q.clamp(min=self.eps)
        alpha_p = torch.full_like(alpha_q, self.alpha_prior)

        sum_q = alpha_q.sum(dim=1)
        sum_p = alpha_p.sum(dim=1)

        kl = (
            torch.lgamma(sum_q) - torch.lgamma(sum_p)
            - torch.lgamma(alpha_q).sum(dim=1)
            + torch.lgamma(alpha_p).sum(dim=1)
            + ((alpha_q - alpha_p)
               * (torch.digamma(alpha_q)
                  - torch.digamma(sum_q).unsqueeze(1))
              ).sum(dim=1)
        )
        return kl.mean()

    def forward(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        beta: float = 1.0
    ):
        """
        recon_x: reconstructed image
        x: input image
        alpha_q: Dirichlet concentration parameters (B, K)
        """

        # reconstruction term
        recon = F.mse_loss(recon_x, x, reduction="mean")

        # Dirichlet KL
        kl = self.kl_dirichlet(sigma)

        return recon, beta * kl
    
