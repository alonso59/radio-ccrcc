import torch
import torch.nn as nn
from monai.networks.nets.autoencoderkl import AutoencoderKL
from omegaconf import DictConfig
from torchinfo import summary
import logging

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
                
    def forward(self, x):
        reconstruction, z_mu, z_sigma = self.autoencoder(x)
        return reconstruction, z_mu, z_sigma