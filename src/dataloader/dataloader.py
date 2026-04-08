from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torchio as tio
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .augmentations import train_augmentations, val_augmentations
from .sampler import (
    AdaptiveSamplerConfig,
    AdaptiveTumorSamplerWrapper,
    load_sampling_artifacts,
)

logger = logging.getLogger(__name__)


def extract_iterative_statistics(fingerprint_path: str) -> Tuple[float, float, float, float, float]:
    """Load normalization statistics from dataset fingerprint JSON."""
    logger.info("[DATALOADER] Loading dataset statistics from: %s", fingerprint_path)

    if not os.path.exists(fingerprint_path):
        raise FileNotFoundError(f"[DATALOADER] Fingerprint file not found: {fingerprint_path}")

    with open(fingerprint_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    def _pick_stats_dict(payload: Dict[str, Any]) -> Dict[str, Any]:
        # Prefer windowed_intensity (v3+ fingerprints) — stats over the HU window used for training
        windowed = payload.get("windowed_intensity")
        if isinstance(windowed, dict) and windowed:
            first_channel = windowed.get("channel_0", next(iter(windowed.values())))
            if isinstance(first_channel, dict):
                return first_channel

        # Fallback: foreground_intensity (v2 fingerprints and nnU-Net format)
        per_channel = payload.get("foreground_intensity_properties_per_channel")
        if isinstance(per_channel, dict) and per_channel:
            return per_channel.get("0", next(iter(per_channel.values())))

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

    logger.info("[DATALOADER] Stats - mean: %.2f, std: %.2f, median: %.2f", mean, std, median)
    return float(mean), float(std), float(median), float(p25), float(p75)


def extract_class_label(file_path: str) -> str:
    """Extract and normalize class label from file path."""
    class_label = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))
    valid_labels = {"A", "B", "C", "D", "AB", "AC", "AD", "BC", "BD", "NG"}
    if class_label not in valid_labels:
        raise ValueError(f"Unexpected class label '{class_label}' extracted from {file_path}")
    if class_label in {"A", "C", "AC", "AB", "AD"}:
        return "A"
    if class_label in {"B", "BC", "BD"}:
        return "B"
    if class_label == "D":
        return "D"
    return "NG"


def extract_contrast_phase(file_path: str) -> int | None:
    """Extract contrast phase from the phase directory in the file path."""
    phase_mapping = {"NC": 0, "ART": 1, "VEN": 2}
    phase_dir = os.path.basename(os.path.dirname(file_path)).upper()
    contrast_type = phase_mapping.get(phase_dir)
    if contrast_type is None:
        logger.warning("[DATALOADER] Phase not found in %s. Phase dir: %s", file_path, phase_dir)
    return contrast_type


def map_label_to_category(class_label: str) -> int:
    mapping = {"A": 0, "B": 1, "D": 2, "NG": 3}
    return mapping.get(class_label, -1)


def _compute_catalog_key(mask_path: str, voi_root: str) -> str:
    """Match the mask path format used in lesion_catalog.jsonl."""
    mask_root = (Path(voi_root) / "mask").resolve()
    try:
        return str(Path(mask_path).resolve().relative_to(mask_root))
    except Exception:
        # Fallback keeps training running even if paths were serialized differently.
        return Path(mask_path).name


def create_ct_subjects(file_paths: List[str], voi_root: str) -> List[tio.Subject]:
    """Create TorchIO Subjects and attach a stable key for lesion-catalog lookup."""
    subjects: List[tio.Subject] = []

    for file_path in file_paths:
        try:
            data = np.load(file_path)
            mask_path = file_path.replace("images", "mask")
            multilabel_mask = np.load(mask_path)

            tensor_data = torch.tensor(data, dtype=torch.float32)
            tensor_mask = torch.tensor(multilabel_mask, dtype=torch.uint8)

            if tensor_data.ndim == 3:
                tensor_data = tensor_data.unsqueeze(0)
            if tensor_mask.ndim == 3:
                tensor_mask = tensor_mask.unsqueeze(0)

            image = tio.ScalarImage(tensor=tensor_data, affine=np.eye(4))
            mask = tio.LabelMap(tensor=tensor_mask, affine=np.eye(4))
            if image.shape != mask.shape:
                raise ValueError(f"CT and mask shape mismatch: {image.shape} vs {mask.shape}")

            class_label = extract_class_label(file_path)
            phase_label = extract_contrast_phase(file_path)
            catalog_key = _compute_catalog_key(mask_path, voi_root)

            subject = tio.Subject(ct=image, mask=mask, label=class_label, phase=phase_label, image_path=str(file_path), mask_path=str(mask_path), catalog_key=catalog_key,)  # type: ignore
            subjects.append(subject)

        except Exception as e:
            logger.error("[DATALOADER] Error loading %s: %s", file_path, e)
            raise

    logger.info("[DATALOADER] Created %d subjects with CT + mask pairs", len(subjects))
    return subjects


class DataLoaderFactory:
    @staticmethod
    def create_loaders(data_cfg: DictConfig):
        """Create train/val loaders for CT data using lesion-centric offline metadata."""
        logger.info("[DATALOADER] Creating CT data loaders")

        dataset_id = data_cfg.dataset_id
        base_path = data_cfg.get("base_path", "data/dataset")
        base_path = base_path if os.path.isabs(base_path) else to_absolute_path(base_path)

        voi_root = os.path.join(base_path, dataset_id, "voi")
        splits_path = os.path.join(voi_root, "splits.json")

        logger.info("[DATALOADER] Dataset: %s", dataset_id)
        logger.info("[DATALOADER] Loading splits from: %s", splits_path)

        fold = data_cfg.fold
        if not os.path.exists(splits_path):
            raise FileNotFoundError(f"[DATALOADER] Splits file not found: {splits_path} (cwd={os.getcwd()})")

        with open(splits_path, "r", encoding="utf-8") as f:
            dict_splits = json.load(f)

        folds_list = dict_splits["folds"]
        fold_data = next((fold_obj for fold_obj in folds_list if fold_obj.get("fold") == fold), None)
        if fold_data is None:
            raise ValueError(f"Fold {fold} not found in splits.json")

        train_files = fold_data["train_files"]
        val_files = fold_data["val_files"]

        logger.info("[DATALOADER] #train=%d, #val=%d", len(train_files), len(val_files))
        train_subjects = create_ct_subjects(train_files, voi_root=voi_root)
        val_subjects = create_ct_subjects(val_files, voi_root=voi_root)

        fingerprint_path = os.path.join(voi_root, "dataset_fingerprint.json")
        lesion_catalog_path = os.path.join(voi_root, "lesion_catalog.jsonl")
        sampling_profile_path = os.path.join(voi_root, "sampling_profile.json")

        data_stats = extract_iterative_statistics(fingerprint_path)
        lesion_catalog, sampling_profile = load_sampling_artifacts(
            lesion_catalog_path=lesion_catalog_path,
            sampling_profile_path=sampling_profile_path,
        )

        logger.info(
            "[DATALOADER] Loaded offline sampling artifacts: lesions=%d, size_bins=%s",
            len(lesion_catalog),
            sorted((sampling_profile.get("size_bin_counts") or {}).keys()),
        )

        train_transform = train_augmentations()
        val_transform = val_augmentations()

        train_dataset = tio.SubjectsDataset(train_subjects, transform=train_transform)
        val_dataset = tio.SubjectsDataset(val_subjects, transform=val_transform)

        sc = data_cfg.get("sampler", {})
        patch_size = tuple(int(v) for v in sc.get("patch_size", [96, 96, 64]))
        samples_per_volume = int(data_cfg.get("samples_per_volume", 4))
        max_queue_length = int(data_cfg.get("max_queue_length", 200))
        num_workers = int(data_cfg.get("num_workers", 8))

        sampler_config = AdaptiveSamplerConfig(
            tumor_label=int(sc.get("tumor_label", 2)),
            background_sampling_prob=float(sc.get("background_sampling_prob", 0.0)),
            center_jitter_fraction=float(sc.get("center_jitter_fraction", 0.10)),
            border_shift_fraction=float(sc.get("border_shift_fraction", 0.30)),
            context_shift_fraction=float(sc.get("context_shift_fraction", 0.55)),
            min_shift_voxels=float(sc.get("min_shift_voxels", 2.0)),
            max_patches_per_lesion=sc.get("max_patches_per_lesion", None),
            subject_key_name=str(sc.get("subject_key_name", "catalog_key")),
            audit_enabled=bool(sc.get("audit_enabled", False)),
        )

        logger.info(
            "[DATALOADER] Sampler: patch=%s samples=%s queue=%s workers=%s jitter=%.3f border=%.3f context=%.3f",
            patch_size,
            samples_per_volume,
            max_queue_length,
            num_workers,
            sampler_config.center_jitter_fraction,
            sampler_config.border_shift_fraction,
            sampler_config.context_shift_fraction,
        )

        train_sampler = AdaptiveTumorSamplerWrapper(
            patch_size=patch_size, # type: ignore
            lesion_catalog=lesion_catalog,
            sampling_profile=sampling_profile,
            mask_name="mask",
            config=sampler_config,
            stochastic=True,
        )
        val_sampler = AdaptiveTumorSamplerWrapper(
            patch_size=patch_size, # type: ignore
            lesion_catalog=lesion_catalog,
            sampling_profile=sampling_profile,
            mask_name="mask",
            config=sampler_config,
            stochastic=False,
        )

        train_queue = tio.Queue(
            train_dataset,
            max_length=max_queue_length,
            samples_per_volume=samples_per_volume,
            sampler=train_sampler,
            num_workers=num_workers,
            shuffle_subjects=True,
            shuffle_patches=True,
        )

        val_queue = tio.Queue(
            val_dataset,
            max_length=max_queue_length // 2,
            samples_per_volume=samples_per_volume,
            sampler=val_sampler,
            num_workers=num_workers,
            shuffle_subjects=False,
            shuffle_patches=False,
        )

        train_loader = DataLoader(
            train_queue,
            batch_size=data_cfg.batch_size,
            num_workers=0,
            pin_memory=False,
        )
        val_loader = DataLoader(
            val_queue,
            batch_size=data_cfg.batch_size,
            num_workers=0,
            pin_memory=False,
        )

        normalization_stats = {
            "mean": data_stats[0],
            "std": data_stats[1],
            "median": data_stats[2],
            "p25": data_stats[3],
            "p75": data_stats[4],
            "window_min": -200.0,
            "window_max": 300.0,
        }

        logger.info("[DATALOADER] Created CT loaders: train=%d, val=%d", len(train_dataset), len(val_dataset))
        logger.info(
            "[DATALOADER] Normalization stats (windowed): mean=%.2f, std=%.2f",
            normalization_stats["mean"],
            normalization_stats["std"],
        )

        return train_loader, val_loader, normalization_stats