import torch
import torch.nn.functional as F
import torchio as tio
from typing import Sequence, Tuple, Optional

import logging
logger = logging.getLogger(__name__)

P_LOW  = -200
P_HIGH =  300


class HomogenizeToCube(tio.Transform):
    """
    Transforms a volume to a cube by padding smaller dimensions 
    to match the largest dimension.
    """
    def __init__(self, padding_value: float = P_LOW, **kwargs):
        super().__init__(**kwargs)
        self.padding_value = padding_value
    
    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        for image in subject.get_images(intensity_only=False):
            # Get spatial dimensions (D, H, W)
            shape = image.spatial_shape
            max_dim = max(shape)
            
            # If already a cube, skip
            if shape[0] == shape[1] == shape[2]:
                continue
            
            # Calculate padding needed for each dimension
            target_shape = (max_dim, max_dim, max_dim)
            
            # Apply CropOrPad to make it cubic
            crop_pad = tio.CropOrPad(
                target_shape=target_shape, 
                padding_mode=self.padding_value
            )
            image.set_data(crop_pad(image).data)
        
        return subject


class ConditionalResize(tio.Transform):
    """
    Resizes volume to target shape using b-spline interpolation 
    only if current shape differs from target.
    """
    def __init__(self, target_shape: Tuple[int, int, int], **kwargs):
        super().__init__(**kwargs)
        self.target_shape = target_shape
    
    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        for image in subject.get_images(intensity_only=False):
            current_shape = image.spatial_shape
            
            # Only resize if shapes don't match
            if current_shape != self.target_shape:
                resize = tio.Resize(
                    target_shape=self.target_shape, 
                    image_interpolation='bspline'
                )
                image.set_data(resize(image).data)
        
        return subject

def train_augmentations(data_stats) -> tio.Compose:

    mean, std, median, p25, p75 = data_stats
    transforms = [
        tio.RandomAffine(
            scales=(0.95, 1.05),
            degrees=10,                       # up to ±10° rotations (small)
            translation=(5, 5, 5),            # voxels — adjust if using mm
            default_pad_value=P_LOW,
            p=0.75
        ),

        # Local non-rigid deformation
        tio.RandomElasticDeformation(
            num_control_points=7,
            max_displacement=(4.0, 4.0, 4.0),
            locked_borders=2,
            p=0.25
        ),

        # Simulate acquisition / scanner variability (intensity)
        tio.RandomGamma(log_gamma=(-0.15, 0.15), p=0.2),
        tio.RandomBiasField(coefficients=0.3, p=0.1),   # mild; useful for cross-scanner variation
        tio.RandomNoise(mean=0.0, std=(0.0, 0.01), p=0.25),

        # Blur / low-res simulation (occasionally)
        tio.RandomBlur(std=(0.0, 1.0), p=0.05),

        tio.Clamp(out_min=P_LOW, out_max=P_HIGH),
        # tio.CropOrPad(target_shape=(128, 128, 128), padding_mode=P_LOW),

        HomogenizeToCube(padding_value=P_LOW),
        ConditionalResize(target_shape=(128, 128, 128)),
        
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
        # tio.CropOrPad(target_shape=(128, 128, 128), padding_mode=P_LOW),
        HomogenizeToCube(padding_value=P_LOW),
        ConditionalResize(target_shape=(128, 128, 128)),
        # Z-Normalization
        # tio.Lambda(lambda x: (x - mean) / (std + 1e-8)),
        
        # IQR normalization:
        tio.Lambda(lambda x: (x - median) / (p75 - p25 + 1e-8))
    ]

    return tio.Compose(transforms, p=1.0)