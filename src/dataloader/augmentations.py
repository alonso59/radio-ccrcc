import torch
import torch.nn.functional as F
import torchio as tio
from typing import Sequence, Tuple, Optional
from functools import partial

import logging
logger = logging.getLogger(__name__)

from torchio.transforms import RandomAffine, RandomElasticDeformation, RandomGamma, RandomBiasField, RandomNoise, RandomBlur, Clamp, Lambda, Compose


P_LOW  = -200
P_HIGH =  300

def iqr_normalize(x, median, p25, p75):
    return (x - median) / (p75 - p25 + 1e-8)

def train_augmentations(data_stats) -> Compose:

    mean, std, median, p25, p75 = data_stats
    norm_fn = partial(iqr_normalize, median=median, p25=p25, p75=p75)
    
    transforms = [
        RandomAffine(
            scales=(0.95, 1.15),
            degrees=5,
            translation=(4, 4, 4),
            default_pad_value=-200,
            image_interpolation='bspline',
            label_interpolation='nearest',
            include=['ct', 'mask'],
            p=1.0
        ),

        # Local non-rigid deformation
        RandomElasticDeformation(
            num_control_points=7,
            max_displacement=(4.0, 4.0, 4.0),
            locked_borders=2,
            image_interpolation='bspline',
            label_interpolation='nearest',
            include=['ct', 'mask'],
            p=0.3
        ),
        
        # # Intensity augmentations
        # RandomGamma(log_gamma=(-0.15, 0.15), p=0.1, include=['ct']),
        # RandomBiasField(coefficients=0.3, p=0.1, include=['ct']),
        # RandomNoise(mean=0.0, std=(0.0, 0.05), p=0.1, include=['ct']),
        # RandomBlur(std=(0.0, 1.0), p=0.05, include=['ct']),

        Clamp(out_min=P_LOW, out_max=P_HIGH, include=['ct']),
        
        # IQR normalization:
        Lambda(norm_fn, include=['ct']),
    ]
    
    return Compose(transforms, p=1.0)

def val_augmentations(data_stats) -> Compose:
    mean, std, median, p25, p75 = data_stats
    norm_fn = partial(iqr_normalize, median=median, p25=p25, p75=p75)
    
    transforms = [
        Clamp(out_min=P_LOW, out_max=P_HIGH, include=['ct']),

        # IQR normalization:
        Lambda(norm_fn, include=['ct'])
    ]

    return Compose(transforms, p=1.0)