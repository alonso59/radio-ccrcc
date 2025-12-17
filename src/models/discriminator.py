import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import PatchDiscriminator, MultiScalePatchDiscriminator # type: ignore
from omegaconf import DictConfig

class Discriminator(nn.Module):
    """
    PatchGAN Discriminator (single scale).
    Input: [B, 1, 64, 64, 64]
    """
    def __init__(
        self,
        cfg: DictConfig
    ):
        super().__init__()
        self.discriminator = PatchDiscriminator(
            spatial_dims=cfg.model.discriminator.spatial_dims,
            in_channels=cfg.model.discriminator.in_channels,
            out_channels=cfg.model.discriminator.out_channels,
            channels=cfg.model.discriminator.base_channels,
            num_layers_d=cfg.model.discriminator.num_layers_d,
            kernel_size=cfg.model.discriminator.kernel_size,
            norm=cfg.model.discriminator.norm,
            activation=(cfg.model.discriminator.activation, {"negative_slope": cfg.model.discriminator.negative_slope}),
            bias=False,
            padding=1,
            dropout=0,
            last_conv_kernel_size=None
        )

    def forward(self, x):
        return self.discriminator(x)
    

class MultiScaleDiscriminator(nn.Module):
    """
    Multi-Scale PatchGAN Discriminator.
    Input: [B, 1, 64, 64, 64]
    """
    def __init__(
        self,
        cfg: DictConfig
    ):
        super().__init__()
        self.discriminator = MultiScalePatchDiscriminator(
            num_d=2,          # scales: 64 and 32
            num_layers_d=2,   # per-scale depth → total halvings = 4 (64→32→16→8); never hits 1
            spatial_dims=3,
            channels=32,      # 16–32 for 3D
            in_channels=1,
            out_channels=1,
            kernel_size=4,
            activation=("LEAKYRELU", {"negative_slope": 0.2}),
            norm="BATCH",
            bias=False,
            dropout=0.0,
            minimum_size_im=64,
            last_conv_kernel_size=1,
        )

    def forward(self, x):
        return self.discriminator(x)
