# asume input 3,8,8,8 to create a 3D classifier nn.Module

import torch
import torch.nn as nn
from typing import List, Dict, Callable, Optional, Tuple

class Classifier3D(nn.Module):
    def __init__(self, num_classes: int):
        super(Classifier3D, self).__init__()
        self.conv_compressor = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1),  #[B, 8, 16, 16, 16] -> [B, 16, 16, 16, 16]
            nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1), #[B, 16, 16, 16, 16]
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),                #[B, 16, 8, 8, 8]
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1), #[B, 32, 8, 8, 8]
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1), #[B, 32, 8, 8, 8]
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),                # [B, 32, 4, 4, 4] 
        )
        self.cls_head = nn.Sequential(
            # nn.AdaptiveAvgPool3d((1, 1, 1)),  # Output: [B, 128, 1, 1, 1]
            nn.Flatten(),                      # Output: [B, 128]
            nn.Linear(32*4*4*4, 128),                # Fully connected layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)         # Output layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_compressor(x)
        x = self.cls_head(x)
        return x