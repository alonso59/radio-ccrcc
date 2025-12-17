import torch
import torch.nn.functional as F
import torchio as tio
from typing import Sequence, Tuple

import logging
logger = logging.getLogger(__name__)

P_LOW  = -200
P_HIGH =  300

def train_augmentations(data_stats) -> tio.Compose:

    mean, std, median, p25, p75 = data_stats
    transforms = [
        tio.RandomAffine(
            scales=(0.95, 1.05),
            degrees=15,                       # up to ±15° rotations (small)
            translation=(5, 5, 5),            # voxels — adjust if using mm
            default_pad_value=P_LOW,
            p=1.0
        ),

        # Local non-rigid deformation
        tio.RandomElasticDeformation(
            num_control_points=7,
            max_displacement=(4.0, 4.0, 4.0),
            locked_borders=2,
            p=0.25
        ),

        # # Simulate acquisition / scanner variability (intensity)
        # tio.RandomGamma(log_gamma=(-0.15, 0.15), p=0.2),
        # tio.RandomBiasField(coefficients=0.3, p=0.1),   # mild; useful for cross-scanner variation
        # tio.RandomNoise(mean=0.0, std=(0.0, 0.01), p=0.25),

        # # Blur / low-res simulation (occasionally)
        # tio.RandomBlur(std=(0.0, 1.0), p=0.05),

        tio.Clamp(out_min=P_LOW, out_max=P_HIGH),
        tio.CropOrPad(target_shape=(128, 128, 128), padding_mode=P_LOW),

        # Z-Normalization
        # tio.Lambda(lambda x: (x - MEAN) / (STD + 1e-8)),
        
        # IQR normalization:
        tio.Lambda(lambda x: (x - median) / (p75 - p25 + 1e-8))
    ]
    
    return tio.Compose(transforms, p=1.0)

def val_augmentations(data_stats) -> tio.Compose:
    mean, std, median, p25, p75 = data_stats
    transforms = [
        tio.Clamp(out_min=P_LOW, out_max=P_HIGH),
        tio.CropOrPad(target_shape=(128, 128, 128), padding_mode=P_LOW),
        
        # Z-Normalization
        # tio.Lambda(lambda x: (x - mean) / (std + 1e-8)),
        
        # IQR normalization:
        tio.Lambda(lambda x: (x - median) / (p75 - p25 + 1e-8))
    ]

    return tio.Compose(transforms, p=1.0)