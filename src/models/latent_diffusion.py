# Latent Diffusion Model: AutoencoderKL + DRUNet in latent space
import torch
import torch.nn as nn

from .autoencoder import Autoencoder
from .drunet import DRUNet

class LatentDiffusionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.autoencoder = Autoencoder(cfg)  # wrapper or AutoencoderKL
        self.drunet = DRUNet(
            dim=3,
            nc=[16, 32, 64],
            nb=2,
            bias=True,
            num_cond_features=128,
            in_channels=cfg.model.autoencoder.latent_channels,
            out_channels=cfg.model.autoencoder.latent_channels,
            label_dim=0,
            label_dropout=0.0,
            embedding={"in_features": 1, "hidden_features": 256, "out_features": 128},
            norm_type="instance",
        )

    def _ae(self):
        # return underlying AE implementation (either wrapper.autoencoder or the object itself)
        return getattr(self.autoencoder, "autoencoder", self.autoencoder)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Simple forward:
          - encode -> (z_mu, z_sigma)
          - sample z (training) or use mean (eval)
          - run DRUNet on z
          - optionally sampling is done only before decode to produce diverse outputs
          - decode and return (reconstruction, z_mu, z_denoised)
        """
        ae = self._ae()
        z_mu, z_sigma = ae.encode(x)            # (B, C, H, W, D)
        # training: sample; eval: use mean
        if self.training:
            z_in = ae.sampling(z_mu, z_sigma)   # uses MONAI sampling (mu + eps * sigma)
        else:
            z_in = z_mu

        # run diffusion UNet in latent space
        z_denoised = self.drunet(z_in, t)       # same shape as z_in

        # decode (you may want to sample around z_denoised for diversity; here deterministic)
        reconstruction = ae.decode(z_denoised)

        return reconstruction, z_mu, z_denoised



# Example usage	
if __name__ == "__main__":
	main()