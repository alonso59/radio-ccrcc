"""Lightweight 3D classification head for latent space evaluation."""
import torch
import torch.nn as nn


class Classifier3D(nn.Module):
    """Convolutional classification head for 3D latent volumes.

    Designed to receive latent feature maps of shape ``(B, C, 16, 16, 16)``
    and produce class logits. Typically used with a frozen autoencoder encoder
    for downstream representation evaluation.

    Args:
        num_classes: Number of output classes.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv_compressor = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a batch of latent feature maps.

        Args:
            x: Latent tensor of shape ``(B, C, H, W, D)``.

        Returns:
            Class logits of shape ``(B, num_classes)``.
        """
        x = self.conv_compressor(x)
        return self.cls_head(x)
