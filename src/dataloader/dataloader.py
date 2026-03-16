import json
import logging
import os

import numpy as np
import torch
import torchio as tio
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from typing import List, Tuple

from hydra.utils import to_absolute_path
from .augmentations import train_augmentations, val_augmentations
from .sampler import AdaptiveTumorSamplerWrapper, AdaptiveSamplerConfig

logger = logging.getLogger(__name__)

def extract_iterative_statistics(fingerprint_path: str) -> Tuple[float, float, float, float, float]:
    """Load dataset statistics from fingerprint JSON file."""
    logger.info(f"[DATALOADER] Loading dataset statistics from: {fingerprint_path}")
    
    if not os.path.exists(fingerprint_path):
        raise FileNotFoundError(f"[DATALOADER] Fingerprint file not found: {fingerprint_path}")
    
    with open(fingerprint_path, "r") as f:
        stats = json.load(f)
    
    def _pick_stats_dict(payload: dict) -> dict:
        per_channel = payload.get("foreground_intensity_properties_per_channel")
        if isinstance(per_channel, dict) and per_channel:
            return per_channel.get("0", next(iter(per_channel.values())))
        # Handle {"foreground_intensity": {"channel_0": {...stats...}}}
        foreground_intensity = payload.get("foreground_intensity")
        if isinstance(foreground_intensity, dict) and foreground_intensity:
            first_channel = foreground_intensity.get("channel_0", next(iter(foreground_intensity.values())))
            if isinstance(first_channel, dict):
                return first_channel
        for key in ("dataset_statistics", "intensity_properties", "foreground_intensity_properties"):
            if isinstance(payload.get(key), dict):
                return payload[key]
        return payload
    
    stats_dict = _pick_stats_dict(stats)
    
    mean = stats_dict.get("mean")
    std = stats_dict.get("std")
    median = stats_dict.get("median")
    p25 = stats_dict.get("percentile_25_0") or stats_dict.get("percentile_25") or stats_dict.get("p25")
    p75 = stats_dict.get("percentile_75_0") or stats_dict.get("percentile_75") or stats_dict.get("p75")
    
    if any(v is None for v in [mean, std, median, p25, p75]):
        raise ValueError(
            f"[DATALOADER] Missing statistics in {fingerprint_path}. "
            f"Available keys: {sorted(stats_dict.keys())}"
        )
    
    logger.info(f"[DATALOADER] Stats - mean: {mean:.2f}, std: {std:.2f}, median: {median:.2f}")
    return mean, std, median, p25, p75  # type: ignore
    
    
def extract_class_label(file_path):
    """Extract and normalize class label from file path."""
    # based on dir FROM path data/dataset/voi/dataset_id/images/<group_id>/<subject_id>/<c_phase>/**.npy
    class_label = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))
    assert class_label in {'A', 'B', 'C', 'D', 'AB', 'AC', 'AD', 'BC', 'BD', 'NG'}, \
        f"Unexpected class label '{class_label}' extracted from {file_path}"
    if class_label in {'A', 'C', 'AC', 'AB', 'AD'}:
        return 'A'
    elif class_label in {'B', 'BC', 'BD'}:
        return 'B'
    elif class_label == 'D':
        return 'D'
    else:
        return 'NG'
def extract_contrast_phase(file_path):
    """
    Extract contrast type based on phase directory in file path.
        FROM path data/dataset/voi/dataset_id/images/<group_id>/<subject_id>/<c_phase>/**.npy
        <c_phase>: nc, art, ven, delay
    """
    phase_mapping = {
        'NC': 0,
        'ART': 1,
        'VEN': 2,
    }
    # based on FROM path data/dataset/voi/dataset_id/images/<group_id>/<subject_id>/<c_phase>/**.npy
    phase_dir = os.path.basename(os.path.dirname(file_path))
    contrast_type = phase_mapping.get(phase_dir, None)
    
    # Exclude if phase not found (only group or subject in path)
    if contrast_type is None:
        logger.warning(f"[DATALOADER] Phase not found in {file_path}. Path: {phase_dir}")
    
    return contrast_type
    
def map_label_to_category(class_label: str):
    """Map class label to categorical integer."""
    mapping = {
        'A': 0,
        'B': 1,
        'D': 2,
        'NG': 3
    }
    return mapping.get(class_label, -1)

def create_ct_subjects(file_paths: List[str]) -> List[tio.Subject]:
    """Create TorchIO Subjects for CT images and multilabel masks."""
    
    subjects = []
    for file_path in file_paths:
        try:
            data = np.load(file_path)
            mask_path = file_path.replace('images', 'mask')
            multilabel_mask = np.load(mask_path)
            
            tensor_data = torch.tensor(data, dtype=torch.float32)
            tensor_mask = torch.tensor(multilabel_mask, dtype=torch.uint8)
            
            if tensor_data.ndim == 3:
                tensor_data = tensor_data.unsqueeze(0)
            if tensor_mask.ndim == 3:
                tensor_mask = tensor_mask.unsqueeze(0)
            
            image = tio.ScalarImage(tensor=tensor_data, affine=np.eye(4))
            mask = tio.LabelMap(tensor=tensor_mask, affine=np.eye(4))
            assert image.shape == mask.shape, \
                f"CT and mask shape mismatch: {image.shape} vs {mask.shape}"
            
            class_label = extract_class_label(file_path)
            phase_label = extract_contrast_phase(file_path)
            
            subject = tio.Subject(ct=image, mask=mask, label=class_label, phase=phase_label)  # type: ignore
            subjects.append(subject)
            
        except Exception as e:
            logger.error(f"[DATALOADER] Error loading {file_path}: {e}")
            raise
    
    logger.info(f"[DATALOADER] Created {len(subjects)} subjects with CT + multilabel mask pairs")
    return subjects


class DataLoaderFactory:
    @staticmethod
    def create_loaders(data_cfg: DictConfig):
        """Create train/val loaders for CT data using config."""
        logger.info("[DATALOADER] Creating CT data loaders")
        
        dataset_id = data_cfg.dataset_id
        base_path = data_cfg.get('base_path', 'data/dataset')

        base_path = base_path if os.path.isabs(base_path) else to_absolute_path(base_path)
        
        splits_filename = f"splits.json"
        splits_path = os.path.join(base_path, dataset_id, 'voi', splits_filename)
        
        logger.info(f"[DATALOADER] Dataset: {dataset_id}")
        logger.info(f"[DATALOADER] Loading splits from: {splits_path}")
        
        fold = data_cfg.fold
        fold_name = "folds"
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
            
            train_files = fold_data['train_files']
            val_files = fold_data['val_files']
        except KeyError as e:
            raise ValueError(f"[DATALOADER] Invalid fold structure in splits file. Missing key: {e}")
        
        logger.info(f"[DATALOADER] #train={len(train_files)}, #val={len(val_files)}")
        train_subjects = create_ct_subjects(train_files)
        val_subjects = create_ct_subjects(val_files)
        
        # Construct fingerprint file path
        splits_dir = os.path.dirname(splits_path)
        fingerprint_filename = f"dataset_fingerprint.json"
        fingerprint_path = os.path.join(splits_dir, fingerprint_filename)
        
        # Load dataset statistics from fingerprint file
        data_stats = extract_iterative_statistics(fingerprint_path)
        
        train_transform = train_augmentations(data_stats)
        val_transform = val_augmentations(data_stats)
        
        train_dataset = tio.SubjectsDataset(train_subjects, transform=train_transform)
        val_dataset = tio.SubjectsDataset(val_subjects, transform=val_transform)
        
        patch_size = (96, 96, 64)
        patch_overlap = (48, 48, 32)
        samples_per_volume = data_cfg.get('samples_per_volume', 4)  
        max_queue_length = data_cfg.get('max_queue_length', 200) 
        num_workers = data_cfg.get('num_workers', 8)
        sampler_config = AdaptiveSamplerConfig(
                            min_tumor_voxels=50,
                            voxels_per_patch=300,
                            max_patches_cap=8,
                            tumor_label=2,
                            top_pool_factor=1,
                            weighted_pool_sampling=True,
                            ring_dilate_vox=1,
                            ring_weight=0.1,
                            kidney_label=1,
                            kidney_weight=0.05,
                            border_weight=0.2,
                            selection_pool_factor=4,
                            diversity_weight=0.35,
                            iou_redundancy_weight=0.65,
                            distance_redundancy_weight=0.35,
                            audit_enabled=True,
                        )
        
        logger.info(
            "[DATALOADER] Sampler: patch=%s overlap=%s min_voxels=%s voxels_per_patch=%s max_cap=%s samples=%s queue=%s workers=%s",
            patch_size,
            patch_overlap,
            sampler_config.min_tumor_voxels,
            sampler_config.voxels_per_patch,
            sampler_config.max_patches_cap,
            samples_per_volume,
            max_queue_length,
            num_workers,
        )
        
        tumor_sampler = AdaptiveTumorSamplerWrapper(
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            mask_name='mask',
            config=sampler_config,
        )
        train_queue = tio.Queue(
            train_dataset,
            max_length=max_queue_length,
            samples_per_volume=samples_per_volume,
            sampler=tumor_sampler,
            num_workers=num_workers,
            shuffle_subjects=True,
            shuffle_patches=True,
        )
        
        val_queue = tio.Queue(
            val_dataset,
            max_length=max_queue_length // 2,
            samples_per_volume=samples_per_volume,
            sampler=tumor_sampler,
            num_workers=num_workers,
            shuffle_subjects=False,
            shuffle_patches=False,
        )
        
        train_loader = DataLoader(
            train_queue,
            batch_size=data_cfg.batch_size,
            num_workers=0,  # Queue handles workers
            pin_memory=False,  # TorchIO images are not pin_memory compatible
        )
        
        val_loader = DataLoader(
            val_queue,
            batch_size=data_cfg.batch_size,
            num_workers=0,
            pin_memory=False,  # TorchIO images are not pin_memory compatible
        )
        
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

