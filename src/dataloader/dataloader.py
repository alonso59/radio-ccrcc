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
logger = logging.getLogger(__name__)

def extract_iterative_statistics(dataloader: DataLoader):
    """
    Compute dataset-wide intensity statistics iteratively.
    
    Args:
        dataloader: Torch DataLoader yielding batches of CT volumes.
    Returns:
        mean: Mean intensity across dataset.
        std: Standard deviation of intensities across dataset.
        median: Median intensity across dataset.
        p25: 25th percentile intensity.
        p75: 75th percentile intensity.
    """
    logger.info("[DATALOADER] Computing dataset intensity statistics...")
    n_voxels = 0
    sum_intensity = 0.0
    sum_squared_intensity = 0.0
    all_intensities = []

    for batch in dataloader:
        images = batch['ct'][tio.DATA]
        batch_size = images.size(0)
        n_voxels += images.numel()
        sum_intensity += images.sum().item()
        sum_squared_intensity += (images ** 2).sum().item()
        all_intensities.append(images.cpu().numpy().flatten())

    mean = sum_intensity / n_voxels
    variance = (sum_squared_intensity / n_voxels) - (mean ** 2)
    std = np.sqrt(variance)

    all_intensities = np.concatenate(all_intensities)
    median = np.median(all_intensities)
    p25 = np.percentile(all_intensities, 25)
    p75 = np.percentile(all_intensities, 75)

    logger.info(f"[DATALOADER] Computed statistics - Mean: {mean}, Std: {std}, Median: {median}, P25: {p25}, P75: {p75}")
    return mean, std, median, p25, p75
    
    
def extract_class_label(file_path: str) -> str:
    """
    Extract and normalize class label from file path.
    
    Args:
        file_path: Path to the medical imaging file
        
    Returns:
        Normalized class label string
    """
    class_label = os.path.basename(os.path.dirname(file_path))
    # Normalize A, C, and A+C labels to 'A'
    # 0
    if class_label in {'A', 'C', 'A+C', 'A+B', 'A+D', 'A+D'}:
        return 'A'
    # 1
    if class_label in {'B',  'B+C', 'B+D'}:
        return 'B'
    # 2
    if class_label in {'D'}:
        return 'D'
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

