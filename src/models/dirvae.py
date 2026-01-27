import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets.autoencoderkl import AutoencoderKL
from omegaconf import DictConfig
import logging
from torchinfo import summary

class DirVAE(nn.Module):
    """
    Pure DirVAE:
    - Encoder output is a deterministic spatial feature map (no Gaussian).
    - A single global Dirichlet latent z_dir models subtype / phenotype mixture.
    - z_dir is injected into the decoder.
    - Reconstruction + Dirichlet KL loss.
    """

    def __init__(self, cfg: DictConfig, K: int = 8, alpha_prior: float = 1.0, eps: float = 1e-6):
        super().__init__()

        # -------------------------------
        # Base AutoencoderKL (encoder + decoder only)
        # -------------------------------
        self.autoencoder = AutoencoderKL(
            spatial_dims=cfg.model.autoencoder.spatial_dims,
            in_channels=cfg.model.autoencoder.in_channels,
            out_channels=cfg.model.autoencoder.out_channels,
            num_res_blocks=tuple(cfg.model.autoencoder.num_res_blocks),
            channels=tuple(cfg.model.autoencoder.channels),
            attention_levels=tuple(cfg.model.autoencoder.attention_levels),
            latent_channels=cfg.model.autoencoder.latent_channels,
            norm_num_groups=cfg.model.autoencoder.norm_num_groups,
            norm_eps=cfg.model.autoencoder.norm_eps,
            with_encoder_nonlocal_attn=cfg.model.autoencoder.with_encoder_nonlocal_attn,
            with_decoder_nonlocal_attn=cfg.model.autoencoder.with_decoder_nonlocal_attn,
            use_checkpoint=cfg.model.autoencoder.use_checkpoint,
            use_convtranspose=cfg.model.autoencoder.use_convtranspose,
            include_fc=cfg.model.autoencoder.include_fc,
            use_combined_linear=cfg.model.autoencoder.use_combined_linear,
            use_flash_attention=cfg.model.autoencoder.use_flash_attention
        )

        self.encoder = self.autoencoder.encoder
        self.decoder = self.autoencoder.decoder

        # -------------------------------
        # Dirichlet latent
        # -------------------------------
        self.C = cfg.model.autoencoder.latent_channels
        self.K = K
        self.alpha_prior = alpha_prior
        self.eps = eps

        # global pooling of spatial features
        self.pool = nn.AdaptiveAvgPool3d(1)

        # map pooled features -> Dirichlet parameters
        self.fc_alpha = nn.Linear(self.C, self.K)
        self.softplus = nn.Softplus()

        # inject z_dir into spatial features
        self.dir_to_spatial = nn.Linear(self.K, self.C)
        self.fuse = nn.Conv3d(self.C * 2, self.C, kernel_size=1)

        # -------------------------------
        # Pretrain / freeze logic
        # -------------------------------
        if cfg.model.autoencoder.pretrain:
            logging.info("Loading pretrained weights from:", cfg.model.autoencoder.pretrained_path)
            state_dict = torch.load(cfg.model.autoencoder.pretrained_path)["model_state_dict"]
            state_dict = {k.replace("autoencoder.", ""): v for k, v in state_dict.items()}
            self.autoencoder.load_state_dict(state_dict=state_dict, strict=True)

        if cfg.model.autoencoder.freeze:
            logging.info("Freezing autoencoder weights.")
            for p in self.autoencoder.parameters():
                p.requires_grad = False

    # ------------------------------------------------
    # Dirichlet sampling (Gamma composition, DirVAE)
    # ------------------------------------------------
    def sample_dirichlet(self, alpha_raw):
        alpha_q = self.softplus(alpha_raw) + self.eps  # (B, K)

        u = torch.rand_like(alpha_q).clamp(self.eps, 1.0 - self.eps)
        log_v = (torch.log(u) + torch.log(alpha_q) + torch.lgamma(alpha_q)) / alpha_q
        v = torch.exp(log_v)

        z_dir = v / (v.sum(dim=1, keepdim=True) + self.eps)
        return z_dir, alpha_q

    # ------------------------------------------------
    # Forward
    # ------------------------------------------------
    def forward(self, x):
        # deterministic spatial features
        z_spatial = self.encoder(x)  # (B, C, D', H', W')

        # global pooling -> Dirichlet
        pooled = self.pool(z_spatial).flatten(1)  # (B, C)
        alpha_raw = self.fc_alpha(pooled)
        z_dir, alpha_q = self.sample_dirichlet(alpha_raw)

        # inject Dirichlet into spatial features
        dir_feat = self.dir_to_spatial(z_dir).view(-1, self.C, 1, 1, 1)
        dir_feat = dir_feat.expand_as(z_spatial)

        z_fused = self.fuse(torch.cat([z_spatial, dir_feat], dim=1))

        # decode
        reconstruction = self.decoder(z_fused)

        return reconstruction, z_dir, alpha_q

if __name__ == "__main__":
    # simple test
    cfg = DictConfig(
        {
            "model": {
                "autoencoder": {
                    "spatial_dims": 3,
                    "in_channels": 1,
                    "out_channels": 1,

                    # One ResBlock per scale → weaker, more interpretable decoder
                    "num_res_blocks": [1, 1, 1, 1, 1],

                    # Conservative channel growth, capped early
                    "channels": [16, 32, 64, 64, 64],

                    # No attention: data too small, phenotypes are global
                    "attention_levels": [0, 0, 0, 0, 0],

                    # Latent channels = encoder bottleneck width
                    # This feeds your GAP → Dirichlet head
                    "latent_channels": 64,

                    # GroupNorm is stable for small batch sizes
                    "norm_num_groups": 8,
                    "norm_eps": 1e-6,

                    # Explicitly disable attention
                    "with_encoder_nonlocal_attn": False,
                    "with_decoder_nonlocal_attn": False,

                    # Memory is manageable at this size; checkpointing unnecessary
                    "use_checkpoint": False,

                    # ConvTranspose is fine at this scale
                    "use_convtranspose": True,

                    # No FC inside AutoencoderKL — Dirichlet head is external
                    "include_fc": False,
                    "use_combined_linear": False,
                    "use_flash_attention": False,

                    # Training from scratch is safer here
                    "pretrain": False,
                    "freeze": False,
                    "pretrained_path": ""
                }
            }
        }
    )

    model = DirVAE(cfg, K=100).to("cpu")
    summary(model, input_size=(1, 1, 128, 128, 128), device="cpu")

    # print("Input shape:", x.shape)
    # print("Reconstruction shape:", recon.shape)
    # print("Dirichlet latent shape:", z_dir.shape)
    # print("Alpha_q shape:", alpha_q.shape)