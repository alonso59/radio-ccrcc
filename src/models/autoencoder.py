import torch
import torch.nn as nn
from monai.networks.nets.autoencoderkl import AutoencoderKL
# import resnet18 from monai
from monai.networks.nets.resnet import resnet18
from omegaconf import DictConfig
from torchinfo import summary
import logging
from omegaconf import OmegaConf, DictConfig

BACKBONE_NORM = "frozen_batch"  # choose: "group" or "frozen_batch"

class Autoencoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
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
        self.phase_embedding = nn.Embedding(num_embeddings=4, embedding_dim=cfg.model.autoencoder.latent_channels)
        if cfg.model.autoencoder.pretrain:
            logging.info("Loading pretrained weights from:", cfg.model.autoencoder.pretrained_path)
            state_dict = torch.load(cfg.model.autoencoder.pretrained_path)['model_state_dict']
            # remove autoencoder. prefix
            state_dict = {k.replace("autoencoder.", ""): v for k, v in state_dict.items()}
            
            self.autoencoder.load_state_dict(state_dict=state_dict, strict=True)
        if cfg.model.autoencoder.freeze:
            logging.info("Freezing autoencoder weights.")
            for param in self.autoencoder.parameters():
                param.requires_grad = False
                
    def forward(self, x, phase=None):
        z_mu, z_sigma = self.autoencoder.encode(x)
        z = self.autoencoder.sampling(z_mu, z_sigma)
        
        # if phase is not None:
        #     phase_embedding = self.phase_embedding(phase)  # (B, latent_channels)
        #     # Expand to match spatial dimensions of z
        #     # z shape: (B, latent_channels, H, W, D)
        #     _, _, h, w, d = z.shape
        #     phase_embedding = phase_embedding.view(phase_embedding.shape[0], phase_embedding.shape[1], 1, 1, 1)
        #     phase_embedding = phase_embedding.expand(-1, -1, h, w, d)  # (B, latent_channels, H, W, D)
            
        #     # Simple addition to modulate latent space (non-destructive)
        #     z = z + phase_embedding
        
        reconstruction = self.autoencoder.decode(z)
        return reconstruction, z_mu, z_sigma




def _valid_num_groups(num_channels: int, preferred_groups: int) -> int:
    g = min(preferred_groups, num_channels)
    while g > 1 and num_channels % g != 0:
        g -= 1
    return g


def replace_bn3d_with_gn_(module: nn.Module, preferred_groups: int) -> nn.Module:
    """
    In-place: BatchNorm3d -> GroupNorm
    Preserves conv weights, copies affine gamma/beta when possible.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm3d):
            c = child.num_features
            g = _valid_num_groups(c, preferred_groups)

            gn = nn.GroupNorm(
                num_groups=g,
                num_channels=c,
                eps=child.eps,
                affine=True,
            )

            with torch.no_grad():
                if child.affine:
                    gn.weight.copy_(child.weight)
                    gn.bias.copy_(child.bias)

            setattr(module, name, gn)
        else:
            replace_bn3d_with_gn_(child, preferred_groups)

    return module


def freeze_bn_stats_(module: nn.Module) -> None:
    """
    Keep BatchNorm layers in eval mode and freeze affine params.
    Useful when batch size is tiny.
    """
    for m in module.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eval()
            if m.affine:
                m.weight.requires_grad = False
                m.bias.requires_grad = False


def freeze_module_(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


class Resnet18Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        ae_cfg = cfg.model.autoencoder

        self.resnet18_backbone = resnet18(
            pretrained=ae_cfg.pretrain,
            n_input_channels=ae_cfg.in_channels,
            feed_forward=False,
            shortcut_type="A",
            bias_downsample=True,
        )

        if hasattr(self.resnet18_backbone, "fc"):
            self.resnet18_backbone.fc = None

        self.out_channels = 512
        self.norm_mode = BACKBONE_NORM

        if self.norm_mode == "group":
            replace_bn3d_with_gn_(
                self.resnet18_backbone,
                preferred_groups=ae_cfg.norm_num_groups,
            )
        elif self.norm_mode not in {"batch", "frozen_batch"}:
            raise ValueError(f"Unknown BACKBONE_NORM={self.norm_mode}")

        if ae_cfg.freeze:
            freeze_module_(self.resnet18_backbone)

    def train(self, mode: bool = True):
        super().train(mode)

        # if you keep BN, freeze its stats when requested
        if self.norm_mode == "frozen_batch":
            freeze_bn_stats_(self.resnet18_backbone)

        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet18_backbone.conv1(x)
        x = self.resnet18_backbone.bn1(x)
        x = self.resnet18_backbone.act(x)

        if not self.resnet18_backbone.no_max_pool:
            x = self.resnet18_backbone.maxpool(x)

        x = self.resnet18_backbone.layer1(x)
        x = self.resnet18_backbone.layer2(x)
        x = self.resnet18_backbone.layer3(x)
        x = self.resnet18_backbone.layer4(x)

        return x


class AutoencoderR18(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        ae_cfg = cfg.model.autoencoder

        # donor MONAI model: we reuse decoder-side VAE pieces
        donor = AutoencoderKL(
            spatial_dims=ae_cfg.spatial_dims,
            in_channels=ae_cfg.in_channels,
            out_channels=ae_cfg.out_channels,
            num_res_blocks=tuple(ae_cfg.num_res_blocks),
            channels=tuple(ae_cfg.channels),
            attention_levels=tuple(ae_cfg.attention_levels),
            latent_channels=ae_cfg.latent_channels,
            norm_num_groups=ae_cfg.norm_num_groups,
            norm_eps=ae_cfg.norm_eps,
            with_encoder_nonlocal_attn=ae_cfg.with_encoder_nonlocal_attn,
            with_decoder_nonlocal_attn=ae_cfg.with_decoder_nonlocal_attn,
            use_checkpoint=ae_cfg.use_checkpoint,
            use_convtranspose=ae_cfg.use_convtranspose,
            include_fc=ae_cfg.include_fc,
            use_combined_linear=ae_cfg.use_combined_linear,
            use_flash_attention=ae_cfg.use_flash_attention,
        )

        # keep only what we actually need
        self.decoder = donor.decoder
        self.post_quant_conv = donor.post_quant_conv
        self.quant_conv_mu = donor.quant_conv_mu
        self.quant_conv_log_sigma = donor.quant_conv_log_sigma
        self.sampling = donor.sampling

        self.encoder = Resnet18Backbone(cfg)

        # internal adapter width derived from existing config only
        # latent=16 -> hidden=64, latent=32 -> hidden=128
        hidden_channels = min(128, max(32, ae_cfg.latent_channels * 4))
        gn_groups = _valid_num_groups(hidden_channels, ae_cfg.norm_num_groups)

        self.to_latent = nn.Sequential(
            nn.Conv3d(self.encoder.out_channels, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(
                num_groups=gn_groups,
                num_channels=hidden_channels,
                eps=ae_cfg.norm_eps,
                affine=True,
            ),
            nn.SiLU(inplace=True),
            nn.Conv3d(
                hidden_channels,
                ae_cfg.latent_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
        )

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        h = self.to_latent(h)

        mu = self.quant_conv_mu(h)
        logvar = self.quant_conv_log_sigma(h).clamp(-30.0, 20.0)
        sigma = torch.exp(0.5 * logvar)

        return mu, logvar, sigma

    def reparameterize(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return self.sampling(mu, sigma)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv(z)
        x_hat = self.decoder(z)
        return x_hat

    def forward(self, x: torch.Tensor, phase=None):
        # phase kept for API compatibility, but not used
        mu, logvar, sigma = self.encode(x)
        z = self.reparameterize(mu, sigma)
        reconstruction = self.decode(z)

        # keep tuple output to minimize downstream changes
        return reconstruction, mu, logvar

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

        if reduction == "mean":
            return kl.mean()
        if reduction == "sum":
            return kl.sum()
        if reduction == "none":
            return kl
        raise ValueError(f"Unknown reduction={reduction}")
    
def main():
    # prototype to test the resnet18 archiecture in summary and check the output shapes
    cfg = OmegaConf.create(
        {
            "model": {
                "autoencoder": {
                    "spatial_dims": 3,
                    "in_channels": 1,
                    "out_channels": 1,
                    "num_res_blocks": [2, 2, 2, 2, 2],
                    "channels": [16, 32, 64, 64, 128],
                    "attention_levels": [False, False, False, False, True],
                    "latent_channels": 16,
                    "norm_num_groups": 16,
                    "norm_eps": 1e-6,
                    "with_encoder_nonlocal_attn": False,
                    "with_decoder_nonlocal_attn": False,
                    "use_checkpoint": False,
                    "use_convtranspose": True,
                    "include_fc": False,
                    "use_combined_linear": False,
                    "use_flash_attention": False,
                    "pretrain": True,
                    "freeze": False,
                }
            }
        }
    )
    model = Autoencoder(cfg)
    summary(model, input_size=(1, 1, 96, 96, 64))
    
if __name__ == "__main__":
    main()