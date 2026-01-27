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
from typing import List, Tuple
from omegaconf import DictConfig
from src.dataloader.augmentations import train_augmentations, val_augmentations
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

from hydra.utils import to_absolute_path

logger = logging.getLogger(__name__)

def extract_iterative_statistics(fingerprint_path: str) -> Tuple[float, float, float, float, float]:
    """
    Load dataset statistics from fingerprint JSON file.
    
    Args:
        fingerprint_path: Path to dataset_fingerprint_images.json or dataset_fingerprint_segmentation.json
        
    Returns:
        Tuple of (mean, std, median, p25, p75)
    """
    logger.info(f"[DATALOADER] Loading dataset statistics from: {fingerprint_path}")
    
    if not os.path.exists(fingerprint_path):
        raise FileNotFoundError(f"[DATALOADER] Fingerprint file not found: {fingerprint_path}")
    
    with open(fingerprint_path, "r") as f:
        stats = json.load(f)
    
    def _pick_stats_dict(payload: dict) -> dict:
        """Return the dict that actually contains the intensity stats."""
        # Most robust: nnUNet-style (or similar) per-channel properties
        per_channel = payload.get("foreground_intensity_properties_per_channel")
        if isinstance(per_channel, dict) and per_channel:
            # Common key is "0" but accept any single/first channel.
            if "0" in per_channel and isinstance(per_channel["0"], dict):
                return per_channel["0"]
            for _, v in per_channel.items():
                if isinstance(v, dict):
                    return v

        # Fallbacks if project writes a flatter schema
        for k in ("dataset_statistics", "intensity_properties", "foreground_intensity_properties"):
            v = payload.get(k)
            if isinstance(v, dict) and v:
                return v

        return payload

    stats_dict = _pick_stats_dict(stats)

    # Extract required statistics (support a few common key variants)
    mean = stats_dict.get("mean")
    std = stats_dict.get("std")
    median = stats_dict.get("median")
    p25 = stats_dict.get("percentile_25_0", stats_dict.get("percentile_25", stats_dict.get("p25")))
    p75 = stats_dict.get("percentile_75_0", stats_dict.get("percentile_75", stats_dict.get("p75")))
    
    # Validate all required fields are present
    if any(v is None for v in [mean, std, median, p25, p75]):
        available = sorted(list(stats_dict.keys())) if isinstance(stats_dict, dict) else []
        raise ValueError(
            f"[DATALOADER] Missing required statistics in fingerprint file: {fingerprint_path}. "
            f"Looked in keys: {available}"
        )

    assert mean is not None
    assert std is not None
    assert median is not None
    assert p25 is not None
    assert p75 is not None
    
    mean_f = float(mean)
    std_f = float(std)
    median_f = float(median)
    p25_f = float(p25)
    p75_f = float(p75)

    logger.info(
        f"[DATALOADER] Loaded statistics - mean: {mean_f:.2f}, std: {std_f:.2f}, median: {median_f:.2f}"
    )
    return mean_f, std_f, median_f, p25_f, p75_f
    
    
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
            - dataset_id: Dataset ID (e.g., Dataset320)
            - dataset_type: images or segmentation
            - base_path: Base path to dataset folder (default: data/dataset)
            - fold: fold name or number
            - batch_size, num_workers
        """
        logger.info("[DATALOADER] Creating CT data loaders")
        
        # Construct paths from dataset_id and dataset_type
        dataset_id = data_cfg.dataset_id
        dataset_type = data_cfg.get('dataset_type', 'images')
        base_path = data_cfg.get('base_path', 'data/dataset')

        # Hydra changes the process working directory to the run folder.
        # Resolve relative dataset paths against the original project cwd.
        base_path = base_path if os.path.isabs(base_path) else to_absolute_path(base_path)
        
        splits_filename = f"splits_{dataset_type}.json"
        splits_path = os.path.join(base_path, dataset_id, 'voi', splits_filename)
        
        logger.info(f"[DATALOADER] Dataset: {dataset_id}, Type: {dataset_type}")
        logger.info(f"[DATALOADER] Loading splits from: {splits_path}")
        
        fold = data_cfg.fold
        fold_name = "ct_folds"
        if not os.path.exists(splits_path):
            raise FileNotFoundError(
                f"[DATALOADER] Splits file not found: {splits_path} (cwd={os.getcwd()})"
            )

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
        
        # Construct fingerprint file path
        splits_dir = os.path.dirname(splits_path)
        fingerprint_filename = f"dataset_fingerprint_{dataset_type}.json"
        fingerprint_path = os.path.join(splits_dir, fingerprint_filename)
        
        # Load dataset statistics from fingerprint file
        data_stats = extract_iterative_statistics(fingerprint_path)
        
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

