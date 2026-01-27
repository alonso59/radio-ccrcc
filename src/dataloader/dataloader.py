"""
Optimized DataLoader factory for CT datasets using TorchIO.
Centralizes loader creation, robust error handling, and logging.
"""

import json
import logging
import torch
import numpy as np
import nibabel as nib
import torchio as tio
from typing import List
from omegaconf import DictConfig
from src.dataloader.augmentations import train_augmentations, val_augmentations
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

def extract_iterative_statistics(dataloader: DataLoader, hu_min: float = -200.0, hu_max: float = 300.0):
    # """
    # Compute dataset-wide intensity statistics iteratively for voxels within target HU range.
    
    # Args:
    #     dataloader: Torch DataLoader yielding batches of CT volumes.
    #     hu_min: Minimum HU value to include in statistics (default: -200 HU).
    #     hu_max: Maximum HU value to include in statistics (default: 300 HU).
    # Returns:
    #     mean: Mean intensity across dataset (within HU range).
    #     std: Standard deviation of intensities (within HU range).
    #     median: Median intensity (within HU range).
    #     p25: 25th percentile intensity (within HU range).
    #     p75: 75th percentile intensity (within HU range).
    # """
    # logger.info(f"[DATALOADER] Computing dataset intensity statistics for HU range [{hu_min}, {hu_max}]...")
    # n_voxels = 0
    # sum_intensity = 0.0
    # sum_squared_intensity = 0.0
    # all_intensities = []

    # for batch in tqdm(dataloader, desc="[DATASET] Computing statistics"):
    #     images = batch['ct'][tio.DATA]
        
    #     # Only include voxels within target HU range (no clamping, just filtering)
    #     mask = (images > hu_min) & (images < hu_max)
    #     masked_images = images[mask]
        
    #     if masked_images.numel() == 0:
    #         continue
        
    #     n_voxels += masked_images.numel()
    #     sum_intensity += masked_images.sum().item()
    #     sum_squared_intensity += (masked_images ** 2).sum().item()
    #     all_intensities.append(masked_images.cpu().numpy().flatten())

    # mean = sum_intensity / n_voxels
    # variance = (sum_squared_intensity / n_voxels) - (mean ** 2)
    # std = np.sqrt(variance)

    # all_intensities = np.concatenate(all_intensities)
    # median = np.median(all_intensities)
    # p25 = np.percentile(all_intensities, 25)
    # p75 = np.percentile(all_intensities, 75)

    # logger.info(f"[DATALOADER] Computed statistics ({n_voxels} voxels in range) - Mean: {mean:.2f}, Std: {std:.2f}, Median: {median:.2f}, P25: {p25:.2f}, P75: {p75:.2f}")
    
    stats = {
      "mean": 17.4747314453125,
      "std": 98.72435760498047,
      "min": -199.0,
      "max": 300.0,
      "median": 27.0,
      "percentile_00_5": -199.0,
      "percentile_05_0": -118.0,
      "percentile_25_0": -74.0,
      "percentile_75_0": 82.0,
      "percentile_95_0": 183.0,
      "percentile_99_5": 298.0,
      "iqr": 156.0
    }
    mean, std, median, p25, p75 = stats["mean"], stats["std"], stats["median"], stats["percentile_25_0"], stats["percentile_75_0"]
    return mean, std, median, p25, p75
    
    
def extract_class_label(file_path: str) -> str:
    """
    Extract and normalize class label from file path.
    
    Args:
        file_path: Path to the medical imaging file
        
    Returns:
        Normalized class label string
    """
    
    class_label = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
    # Normalize A, C, and A+C labels to 'A'
    # 0
    if class_label in {'A', 'C', 'AC', 'AB', 'AD'}:
        return 'A'
    # 1
    if class_label in {'B',  'BC', 'BD'}:
        return 'B'
    # 2
    if class_label in {'D'}:
        return 'D'
    # 3    
    if not class_label in {'A', 'B', 'C', 'D', 'AC', 'AB', 'AD', 'BC', 'BD'}:
        return 'NG'
    return class_label


def map_label_to_category(class_label: str) -> int:
    """
    Map class label to categorical integer.
    
    Args:
        class_label: String class label
        
    Returns:
        Integer category mapping, -1 if unknown
    """
    mapping = {
        'A': 0,
        'B': 1,
        'C': 0,
        'D': 2,
        'A+C': 0,
        'A+B': 1,
        'A+D': 0,
        'B+C': 1,
        'B+D': 1
    }
    return mapping.get(class_label, -1)

def create_ct_subjects(file_paths: List[str]) -> List[tio.Subject]:
    """
    Create TorchIO Subjects for CT images from NIfTI files.
    Args:
        file_paths: List of NIfTI file paths
    Returns:
        List of TorchIO Subjects
    Raises:
        ValueError: If tensor dimensions are invalid or file not found
    """
    subjects = []
    for file_path in file_paths:
        try:
            data = np.load(file_path) # type:ignore
            # from file_path with structure as example: /home/alonso/Documents/ccRCC/data/tcga_kirc_voi_f/img loag the mas of /home/alonso/Documents/ccRCC/data/tcga_kirc_voi_f/msk but must be the same file
            # data_mask = np.load(file_path.replace('img', 'msk')) # type:ignore
            # conver to binary ensure 0 or 1
            # data_mask = (data_mask > 0).astype(np.uint8)
            # data = nii.get_fdata() # type:ignore
            # mask = np.load(file_path.replace('.npy', '_mask.npy')) # type:ignore
            tensor_data = torch.tensor(data, dtype=torch.float32)
            # tensor_mask = torch.tensor(data_mask, dtype=torch.uint8)

            # assert tensor_data.shape == tensor_mask.shape, f"CT and mask shape mismatch for {file_path}"
            # Ensure tensor is 4D (C, H, W, D)
            if tensor_data.ndim == 3:
                tensor_data = tensor_data.unsqueeze(0)
                # tensor_mask = tensor_mask.unsqueeze(0)
            elif tensor_data.ndim != 4:
                raise ValueError(f"[DATALOADER] Expected 3D or 4D tensor, got {tensor_data.shape} from {file_path}")
            image = tio.ScalarImage(tensor=tensor_data, affine=np.eye(4))
            # Ensure mask is treated as label data so intensity transforms are not applied
            # tensor_mask = tensor_mask.to(torch.uint8)
            # mask = tio.LabelMap(tensor=tensor_mask, affine=np.eye(4))
            class_label = extract_class_label(file_path)
            subject = tio.Subject(ct=image, label=str(class_label))  # type: ignore
            subjects.append(subject)
        except Exception as e:
            logger.error(f"[DATALOADER] Error loading {file_path}: {e}")
            raise
    return subjects

class DataLoaderFactory:
    @staticmethod
    def create_loaders(data_cfg: DictConfig):
        """
        Create train/val loaders for CT data using config.
        data_cfg must have:
            - splits_final: path to JSON with splits
            - fold: fold name or number
            - train_files: key for train files in fold dict
            - val_files: key for val files in fold dict
            - batch_size, num_workers
        """
        logger.info("[DATALOADER] Creating CT data loaders")
        splits_path = data_cfg.splits_path
        fold = data_cfg.fold
        fold_name = "ct_folds"

        logger.info(f"[DATALOADER] Loading splits from: {splits_path}")
        with open(splits_path, "r") as f:
            dict_splits = json.load(f)
        try:
            folds_list = dict_splits[fold_name]
            fold_data = None
            for fold_obj in folds_list:
                if fold_obj.get("fold") == fold:
                    fold_data = fold_obj
                    break
            
            if fold_data is None:
                raise ValueError(f"Fold {fold} not found in {fold_name}")
            
            train_files = fold_data['train_ct_files']
            val_files = fold_data['val_ct_files']
        except KeyError as e:
            raise ValueError(f"[DATALOADER] Invalid fold structure in splits file. Missing key: {e}")
        
        logger.info(f"[DATALOADER] #train={len(train_files)}, #val={len(val_files)}")
        train_subjects = create_ct_subjects(train_files)
        val_subjects = create_ct_subjects(val_files)
        dummy_dataset = tio.SubjectsDataset(train_subjects)
        dummy_loader = DataLoader(
            dummy_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=False,
        )
        
        # Compute dataset statistics
        data_stats = extract_iterative_statistics(dummy_loader)
        
        train_transform = train_augmentations(data_stats)
        val_transform = val_augmentations(data_stats)
        
        train_dataset = tio.SubjectsDataset(train_subjects, transform=train_transform)
        val_dataset = tio.SubjectsDataset(val_subjects, transform=val_transform)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=data_cfg.batch_size,
            shuffle=True,
            num_workers=data_cfg.num_workers,
            pin_memory=True,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=data_cfg.batch_size,
            shuffle=False,
            num_workers=data_cfg.num_workers,
            pin_memory=True,
        )
        
        # Package normalization statistics for metrics computation
        normalization_stats = {
            'mean': data_stats[0],
            'std': data_stats[1],
            'median': data_stats[2],
            'p25': data_stats[3],
            'p75': data_stats[4],
            'iqr': data_stats[4] - data_stats[3],  # IQR = p75 - p25
        }
        
        logger.info(f"[DATALOADER] Created CT loaders: train={len(train_dataset)}, val={len(val_dataset)}")
        logger.info(f"[DATALOADER] Normalization stats: median={normalization_stats['median']:.2f}, "
                   f"IQR={normalization_stats['iqr']:.2f}")
        
        return train_loader, val_loader, normalization_stats

