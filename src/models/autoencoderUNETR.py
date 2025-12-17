import torch
import torch.nn as nn
from monai.networks.nets.autoencoderkl import Decoder
from monai.networks.nets.swin_unetr import SwinUNETR
from monai.networks.blocks.convolutions import Convolution
from omegaconf import DictConfig
from torchinfo import summary

class AutoencoderUNETR(nn.Module):
    """
    Autoencoder using pretrained SwinUNETR encoder (Swin Transformer) and custom decoder.
    Extracts features from Swin Transformer and decodes.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        
        # Create SwinUNETR to match pretrained weights (feature_size=48)
        self.swin_unetr = SwinUNETR(
            img_size=(128, 128, 128),
            in_channels=cfg.model.autoencoder.in_channels,
            out_channels=cfg.model.autoencoder.out_channels,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            feature_size=48,
            norm_name="instance",
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            normalize=True,
            use_checkpoint=False,
            spatial_dims=3,
            downsample="merging",
            use_v2=False,
        )
        summary(
            self.swin_unetr, 
            input_size=(1, cfg.model.autoencoder.in_channels, 128, 128, 128),
            col_names=["input_size", "output_size", "num_params", "trainable"]
        )
        # Load pretrained weights
        try:
            path = '/home/alonso/Documents/radio-ccrcc/models/pretrained/model_swinvit.pt'
            print(f"Loading pretrained SwinUNETR weights from: {path}")
            loaded = torch.load(path)
            
            if isinstance(loaded, dict):
                state_dict = loaded.get('model_state_dict', loaded.get('state_dict', loaded))
            else:
                state_dict = loaded
            
            # Clean and remap keys for SwinUNETR structure
            swin_state_dict = {}
            for k, v in state_dict.items():
                # Remove common prefixes
                k = k.replace("module.", "")
                
                # Only keep Swin Transformer related keys (skip decoder, encoder blocks, heads, etc.)
                if any(skip in k for skip in ['encoder', 'decoder', 'out.', 'convTrans', '_head']):
                    continue
                
                # Remap MLP layer names: fc1 -> linear1, fc2 -> linear2
                k = k.replace("mlp.fc1", "mlp.linear1")
                k = k.replace("mlp.fc2", "mlp.linear2")
                
                # Add swinViT prefix if not present
                if not k.startswith("swinViT."):
                    k = "swinViT." + k
                
                swin_state_dict[k] = v
            
            # Load weights into SwinUNETR (only swinViT part will match)
            missing, unexpected = self.swin_unetr.load_state_dict(swin_state_dict, strict=False)
            
            # Count only swinViT related missing keys
            swin_missing = [k for k in missing if k.startswith('swinViT.')]
            print(f"Loaded pretrained Swin Transformer weights:")
            print(f"  Loaded: {len(swin_state_dict)} keys")
            print(f"  Missing swinViT keys: {len(swin_missing)}")
            print(f"  Total missing (including decoder): {len(missing)}")
            
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            print("Training from scratch...")

        # Extract encoder from SwinUNETR
        self.swinViT = self.swin_unetr.swinViT

        encoder_out_channels = 48 * 4  # feature_size * 2^2 = 192

        # Create a 3d compressor from 192, 8, 8, 8 to latent_channels, 256, 4, 4, 4
        self.compressor = Convolution(
            spatial_dims=cfg.model.autoencoder.spatial_dims,
            in_channels=encoder_out_channels,
            out_channels=cfg.model.autoencoder.latent_channels,
            kernel_size=3,
            strides=2,
            padding=1,
        )
        self.quant_conv_mu = Convolution(
            spatial_dims=cfg.model.autoencoder.spatial_dims,
            in_channels=cfg.model.autoencoder.latent_channels,
            out_channels=cfg.model.autoencoder.latent_channels,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )
        self.quant_conv_log_sigma = Convolution(
            spatial_dims=cfg.model.autoencoder.spatial_dims,
            in_channels=cfg.model.autoencoder.latent_channels,
            out_channels=cfg.model.autoencoder.latent_channels,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )
        self.post_quant_conv = Convolution(
            spatial_dims=cfg.model.autoencoder.spatial_dims,
            in_channels=cfg.model.autoencoder.latent_channels,
            out_channels=cfg.model.autoencoder.latent_channels,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )
        # Decoder
        self.decoder = Decoder(
            spatial_dims=3,
            channels=cfg.model.autoencoder.channels,
            in_channels=cfg.model.autoencoder.latent_channels,
            out_channels=cfg.model.autoencoder.out_channels,
            num_res_blocks=tuple(cfg.model.autoencoder.num_res_blocks),
            norm_num_groups=cfg.model.autoencoder.norm_num_groups,
            norm_eps=cfg.model.autoencoder.norm_eps,
            attention_levels=cfg.model.autoencoder.attention_levels,
            with_nonlocal_attn=cfg.model.autoencoder.with_decoder_nonlocal_attn,
            use_convtranspose=cfg.model.autoencoder.use_convtranspose,
            include_fc=cfg.model.autoencoder.include_fc,
            use_combined_linear=cfg.model.autoencoder.use_combined_linear,
            use_flash_attention=cfg.model.autoencoder.use_flash_attention,
        )
        if cfg.model.autoencoder.freeze:
            # freeze swinViT parameters
            for param in self.swinViT.parameters():
                param.requires_grad = False

    def forward(self, x):
        z_mu, z_sigma = self.encode(x)
        z = self.sampling(z_mu, z_sigma)
        reconstruction = self.decode(z)
        
        return reconstruction, z_mu, z_sigma

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Extract features using Swin Transformer encoder
        hidden_states_out = self.swinViT(x, normalize=True) 
        h = hidden_states_out[2]
        h = self.compressor(h)
        z_mu = self.quant_conv_mu(h)
        z_log_var = self.quant_conv_log_sigma(h)
        z_log_var = torch.clamp(z_log_var, -30.0, 20.0)
        z_sigma = torch.exp(z_log_var / 2)

        return z_mu, z_sigma
    
    def sampling(self, z_mu: torch.Tensor, z_sigma: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(z_sigma)
        z_vae = z_mu + eps * z_sigma
        return z_vae
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv(z)
        reconstruction = self.decoder(z)
        return reconstruction
    