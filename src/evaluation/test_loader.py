"""
Test loader factory for lazy loading of test data.
Memory-efficient test set evaluation.
"""

import json
import logging
import os
from typing import List, Tuple

import numpy as np
import torch
import torchio as tio
from omegaconf import DictConfig
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class TestLoaderFactory:
    """
    Factory for creating test data loaders on-demand.
    Avoids keeping test data in memory during training evaluation.
    """
    
    @staticmethod
    def extract_class_label(file_path: str) -> str:
        """
        Extract and normalize class label from file path.
        
        Args:
            file_path: Path to the medical imaging file
            
        Returns:
            Normalized class label string
        """
        class_label = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        
        # Normalize labels
        if class_label in {'A', 'C', 'AC', 'AB', 'AD'}:
            return 'A'
        if class_label in {'B', 'BC', 'BD'}:
            return 'B'
        if class_label in {'D'}:
            return 'D'
        if class_label not in {'A', 'B', 'C', 'D', 'AC', 'AB', 'AD', 'BC', 'BD'}:
            return 'NG'
        return class_label
    
    @staticmethod
    def create_test_subjects(file_paths: List[str]) -> List[tio.Subject]:
        """
        Create TorchIO Subjects for test CT images.
        
        Args:
            file_paths: List of test file paths
            
        Returns:
            List of TorchIO Subjects
        """
        subjects = []
        for file_path in file_paths:
            try:
                data = np.load(file_path)
                tensor_data = torch.tensor(data, dtype=torch.float32)
                
                # Ensure 4D tensor (C, H, W, D)
                if tensor_data.ndim == 3:
                    tensor_data = tensor_data.unsqueeze(0)
                elif tensor_data.ndim != 4:
                    raise ValueError(f"Expected 3D or 4D tensor, got {tensor_data.shape}")
                
                image = tio.ScalarImage(tensor=tensor_data, affine=np.eye(4))
                class_label = TestLoaderFactory.extract_class_label(file_path)
                subject = tio.Subject(ct=image, label=class_label)  # type: ignore
                subjects.append(subject)
                
            except Exception as e:
                logger.error(f"Error loading test file {file_path}: {e}")
                raise
                
        return subjects
    
    @staticmethod
    def create_test_loader(
        splits_path: str,
        val_transform: tio.Compose,
        batch_size: int = 16,
        num_workers: int = 0
    ) -> Tuple[DataLoader, int]:
        """
        Create test data loader from splits file.
        
        Args:
            splits_path: Path to splits JSON file
            val_transform: TorchIO transform pipeline (same as validation)
            batch_size: Batch size for test loader (smaller to save memory)
            num_workers: Number of workers (0 recommended for test to save memory)
            
        Returns:
            Tuple of (test_loader, num_test_samples)
        """
        logger.info(f"[TEST LOADER] Loading test split from: {splits_path}")
        
        with open(splits_path, 'r') as f:
            splits_data = json.load(f)
        
        # Get test files from top-level key
        if 'test_ct_files' not in splits_data:
            raise KeyError("'test_ct_files' not found in splits file")
        
        test_files = splits_data['test_ct_files']
        logger.info(f"[TEST LOADER] Found {len(test_files)} test samples")
        
        # Create subjects
        test_subjects = TestLoaderFactory.create_test_subjects(test_files)
        
        # Create dataset with validation transform
        test_dataset = tio.SubjectsDataset(test_subjects, transform=val_transform)
        
        # Create loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,  # Disable to save memory
        )
        
        logger.info(f"[TEST LOADER] Created test loader with batch_size={batch_size}")
        
        return test_loader, len(test_files)
    
    @staticmethod
    def create_test_loader_with_stats(
        splits_path: str,
        normalization_stats: Tuple[float, float, float, float, float],
        batch_size: int = 16,
        num_workers: int = 0
    ) -> Tuple[DataLoader, int]:
        """
        Create test loader using pre-computed normalization statistics.
        
        Args:
            splits_path: Path to splits JSON file
            normalization_stats: Tuple of (mean, std, median, p25, p75)
            batch_size: Batch size for test loader
            num_workers: Number of workers
            
        Returns:
            Tuple of (test_loader, num_test_samples)
        """
        # Import augmentation factory (validation transform)
        from src.dataloader.augmentations import val_augmentations
        
        # Create validation transform with same stats
        val_transform = val_augmentations(normalization_stats)
        
        return TestLoaderFactory.create_test_loader(
            splits_path=splits_path,
            val_transform=val_transform,
            batch_size=batch_size,
            num_workers=num_workers
        )
    
    @staticmethod
    def create_test_loader_from_config(
        data_cfg: DictConfig,
        normalization_stats: Tuple[float, float, float, float, float],
        batch_size: int = 16,
        num_workers: int = 0
    ) -> Tuple[DataLoader, int]:
        """
        Create test loader using dataset config to construct paths.
        
        Args:
            data_cfg: Dataset configuration with dataset_id
            normalization_stats: Tuple of (mean, std, median, p25, p75)
            batch_size: Batch size for test loader
            num_workers: Number of workers
            
        Returns:
            Tuple of (test_loader, num_test_samples)
        """
        import os
        
        # Construct splits path from config
        dataset_id = data_cfg.dataset_id
        base_path = data_cfg.get('base_path', 'data/dataset')
        
        splits_filename = f"splits_final.json"
        splits_path = os.path.join(base_path, dataset_id, 'voi', splits_filename)
        
        logger.info(f"[TEST LOADER] Using dataset {dataset_id}")
        
        return TestLoaderFactory.create_test_loader_with_stats(
            splits_path=splits_path,
            normalization_stats=normalization_stats,
            batch_size=batch_size,
            num_workers=num_workers
        )
