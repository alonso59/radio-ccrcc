"""3D Variational Autoencoder wrapper around MONAI AutoencoderKL."""
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets.autoencoderkl import AutoencoderKL
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class Autoencoder(nn.Module):
    """Thin wrapper around MONAI ``AutoencoderKL`` for 3D CT volumes.

    Exposes a stable public API (``encode``, ``decode``, ``sampling``,
    ``encode_features``) so that trainers and evaluators are decoupled from
    the underlying MONAI implementation.

    Args:
        cfg: Hydra ``DictConfig`` containing a ``model.autoencoder`` node with
            architecture parameters and optional ``pretrain`` / ``freeze`` flags.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        ac = cfg.model.autoencoder
        self.autoencoder = AutoencoderKL(
            spatial_dims=ac.spatial_dims,
            in_channels=ac.in_channels,
            out_channels=ac.out_channels,
            num_res_blocks=tuple(ac.num_res_blocks),
            channels=tuple(ac.channels),
            attention_levels=tuple(ac.attention_levels),
            latent_channels=ac.latent_channels,
            norm_num_groups=ac.norm_num_groups,
            norm_eps=ac.norm_eps,
            with_encoder_nonlocal_attn=ac.with_encoder_nonlocal_attn,
            with_decoder_nonlocal_attn=ac.with_decoder_nonlocal_attn,
            use_checkpoint=ac.use_checkpoint,
            use_convtranspose=ac.use_convtranspose,
            include_fc=ac.include_fc,
            use_combined_linear=ac.use_combined_linear,
            use_flash_attention=ac.use_flash_attention,
        )

        if ac.pretrain:
            logger.info("Loading pretrained weights from: %s", ac.pretrained_path)
            state_dict = torch.load(ac.pretrained_path)['model_state_dict']
            state_dict = {k.replace("autoencoder.", ""): v for k, v in state_dict.items()}
            self.autoencoder.load_state_dict(state_dict=state_dict, strict=True)

        if ac.freeze:
            logger.info("Freezing autoencoder weights.")
            for param in self.autoencoder.parameters():
                param.requires_grad = False

    # ------------------------------------------------------------------
    # Public API (required by MR-2)
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a volume to latent distribution parameters.

        Args:
            x: Input tensor of shape ``(B, C, H, W, D)``.

        Returns:
            Tuple of ``(z_mu, z_sigma)`` each of shape ``(B, latent_channels, h, w, d)``.
        """
        return self.autoencoder.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent sample to a reconstructed volume.

        Args:
            z: Latent tensor of shape ``(B, latent_channels, h, w, d)``.

        Returns:
            Reconstructed volume of shape ``(B, C, H, W, D)``.
        """
        return self.autoencoder.decode(z)

    def sampling(self, z_mu: torch.Tensor, z_sigma: torch.Tensor) -> torch.Tensor:
        """Sample a latent vector using the reparameterisation trick.

        Args:
            z_mu: Mean of the latent distribution.
            z_sigma: Standard deviation of the latent distribution.

        Returns:
            Sampled latent tensor of the same shape as inputs.
        """
        return self.autoencoder.sampling(z_mu, z_sigma)

    def encode_features(
        self,
        x: torch.Tensor,
        pooled: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Encode a volume and optionally return a pooled feature vector.

        Args:
            x: Input tensor of shape ``(B, C, H, W, D)``.
            pooled: If ``True``, also return a globally pooled flat feature
                vector suitable for classification heads.

        Returns:
            ``(z_mu, z_sigma)`` when ``pooled=False``, or
            ``(pooled_feat, z_mu, z_sigma)`` when ``pooled=True``.
        """
        z_mu, z_sigma = self.autoencoder.encode(x)
        if pooled:
            feat = F.adaptive_avg_pool3d(z_mu, 1).flatten(1)
            return feat, z_mu, z_sigma
        return z_mu, z_sigma

    def forward(
        self,
        x: torch.Tensor,
        phase: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full encode → sample → decode pass.

        Args:
            x: Input volume of shape ``(B, C, H, W, D)``.
            phase: Unused placeholder for future phase-conditioning support.

        Returns:
            Tuple of ``(reconstruction, z_mu, z_sigma)``.
        """
        z_mu, z_sigma = self.autoencoder.encode(x)
        z = self.autoencoder.sampling(z_mu, z_sigma)
        reconstruction = self.autoencoder.decode(z)
        return reconstruction, z_mu, z_sigma
