import torch
import torch.nn.functional as F
import torchio as tio
from typing import Sequence, Tuple, Optional
from functools import partial

import logging
logger = logging.getLogger(__name__)

from torchio.transforms import RandomAffine, RandomBiasField, RandomElasticDeformation, RandomGamma, RandomFlip, RandomNoise, RandomBlur, Clamp, Lambda, Compose


P_LOW  = -200.0
P_HIGH =  300.0

def window_and_scale(x, p_low=P_LOW, p_high=P_HIGH):
    x = torch.clamp(x, min=p_low, max=p_high)
    x = 2.0 * (x - p_low) / (p_high - p_low) - 1.0
    return x
    # equivalente: (x - 50.0) / 250.0

def train_augmentations() -> Compose:
    transforms = [
        RandomAffine(
            scales=(0.95, 1.05),
            degrees=5,
            translation=(2, 2, 2),
            default_pad_value=P_LOW,
            image_interpolation='bspline',
            label_interpolation='nearest',
            include=['ct', 'mask'],
            p=1.0
        ),

        RandomElasticDeformation(
            num_control_points=7,
            max_displacement=(2, 2, 2),
            locked_borders=2,
            image_interpolation='bspline',
            label_interpolation='nearest',
            include=['ct', 'mask'],
            p=0.1
        ),
        
        # Laterality generalization 
        RandomFlip(axes=(0,), include=['ct', 'mask'], p=0.5),
        
        # Intensity augmentations
        # RandomGamma(log_gamma=(-0.1, 0.1), p=0.1, include=['ct']),
        # RandomBiasField(coefficients=0.3, p=0.1, include=['ct']),
        RandomNoise(mean=0.0, std=(0.0, 0.01), p=0.1, include=['ct']),
        RandomBlur(std=(0.0, 1.0), p=0.05, include=['ct']),
        
        Lambda(window_and_scale, include=['ct']),
    ]
    return Compose(transforms, p=1.0)

def val_augmentations() -> Compose:
    transforms = [
        Lambda(window_and_scale, include=['ct']),
        ]
    return Compose(transforms, p=1.0)