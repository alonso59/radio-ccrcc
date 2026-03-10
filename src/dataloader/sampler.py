"""
Custom patch samplers for TorchIO.
Provides adaptive, quality-aware tumor-focused sampling strategies for medical imaging.

Key Features:
- Adaptive patch count based on tumor size (avoid redundancy for small tumors)
- Quality-based selection (prioritize patches with most tumor content)
- Shuffle on iteration (diversity across epochs when used with Queue)
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torchio as tio


logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AdaptiveSamplerConfig:
    """Configuration for adaptive tumor patch sampling."""
    
    min_tumor_voxels: int = 50
    """Minimum tumor voxels required per patch (quality threshold)."""
    
    voxels_per_patch: int = 500
    """Tumor voxels that 'justify' one patch (for adaptive count calculation)."""
    
    max_patches_cap: int = 8
    """Maximum patches per volume (prevents oversampling large tumors)."""
    
    tumor_label: int = 2
    """Label value for tumor in multilabel mask."""
    # NUEVO (defaults = comportamiento actual)
    top_pool_factor: int = 1          # 1 => pool = K
    weighted_pool_sampling: bool = False 
    ring_dilate_vox: int = 1
    ring_weight: float = 0.1
# =============================================================================
# Helper Functions
# =============================================================================

def _ensure_min_size(subject: tio.Subject, patch_size: Tuple[int, int, int]) -> tio.Subject:
    """
    Pad subject to ensure minimum size matches patch_size.
    
    Args:
        subject: Input TorchIO subject
        patch_size: Required minimum size (H, W, D)
        
    Returns:
        Padded subject if needed, otherwise original subject
    """
    spatial_shape = subject.spatial_shape
    padding_needed = [max(0, p - s) for p, s in zip(patch_size, spatial_shape)]
    
    if any(pad > 0 for pad in padding_needed):
        # Distribute padding evenly: ceil for left, floor for right
        padding_tuple: Tuple[int, int, int, int, int, int] = (
            (padding_needed[0] + 1) // 2, padding_needed[0] // 2,
            (padding_needed[1] + 1) // 2, padding_needed[1] // 2,
            (padding_needed[2] + 1) // 2, padding_needed[2] // 2,
        )
        
        pad_transform = tio.Pad(padding_tuple, padding_mode=0)
        subject = pad_transform(subject)
        logger.debug(f"Padded subject from {spatial_shape} to {subject.spatial_shape}")
    
    return subject


def _compute_adaptive_max_patches(total_tumor_voxels: int, config: AdaptiveSamplerConfig) -> int:
    """
    Compute adaptive maximum patches based on tumor size.
    
    Logic:
    - Small tumors (< 300 voxels) → 1 patch (avoid redundancy)
    - Medium tumors → proportional patches
    - Large tumors → capped at max_patches_cap
    
    Args:
        total_tumor_voxels: Total tumor voxels in the volume
        config: Sampler configuration
        
    Returns:
        Adaptive maximum number of patches to extract
    """
    if total_tumor_voxels == 0:
        return 0
    
    adaptive_max = total_tumor_voxels // config.voxels_per_patch
    adaptive_max = max(1, adaptive_max)  # At least 1 patch
    adaptive_max = min(adaptive_max, config.max_patches_cap)  # Cap upper limit
    
    return adaptive_max


# =============================================================================
# Main Sampler Classes
# =============================================================================

def binary_dilation(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Simple binary dilation using a 3D structuring element."""
    from scipy.ndimage import binary_dilation as scipy_binary_dilation
    
    struct = np.ones((3, 3, 3), dtype=bool)  # 26-connected neighborhood
    dilated = scipy_binary_dilation(mask, structure=struct, iterations=iterations)
    return dilated

class AdaptiveTumorSampler(tio.data.GridSampler):
    """
    Adaptive quality-aware tumor patch sampler.
    
    Key improvements over basic GridSampler:
    1. Quality threshold: Patches must have meaningful tumor content (≥50 voxels)
    2. Adaptive quantity: Patch count scales with tumor size
    3. Score-based selection: Best patches (most tumor) are prioritized
    4. Shuffle on iteration: Different patches selected each epoch
    
    This sampler is designed for use with tio.Queue via AdaptiveTumorSamplerWrapper.
    
    Example:
        >>> config = AdaptiveSamplerConfig(min_tumor_voxels=50, voxels_per_patch=300)
        >>> sampler = AdaptiveTumorSampler(subject, patch_size=(96, 96, 64), config=config)
        >>> for patch in sampler:
        ...     process(patch)
    """
    
    def __init__(
        self,
        subject: tio.Subject,
        patch_size: Tuple[int, int, int],
        patch_overlap: Tuple[int, int, int] = (0, 0, 0),
        mask_name: str = 'mask',
        config: Optional[AdaptiveSamplerConfig] = None,
    ):
        """
        Args:
            subject: TorchIO Subject with CT image and multilabel mask
            patch_size: Size of patches to extract (H, W, D)
            patch_overlap: Overlap between adjacent patches (H, W, D)
            mask_name: Key for the mask in the subject
            config: Sampling configuration (uses defaults if None)
        """
        self.config = config or AdaptiveSamplerConfig()
        self.mask_name = mask_name
        
        # Pad subject if needed before parent init
        subject = _ensure_min_size(subject, patch_size)
        super().__init__(subject, patch_size, patch_overlap)
        
        # Compute patch scores and select best patches
        self._selected_patches = self._select_best_patches()
    
    def _score_all_patches(self) -> List[Tuple[int, int]]:
        """
        Score all grid patches by tumor content (optionally plus peritumoral ring).

        Returns:
            List of (patch_index, score) tuples for patches meeting the minimum threshold,
            sorted by score descending.
        """
        if len(self.locations) == 0:
            return []

        multilabel_mask = self.subject[self.mask_name].data.numpy()  # type: ignore
        tumor_mask = (multilabel_mask == self.config.tumor_label)    # shape: (1, X, Y, Z) typically

        # --- Optional ring scoring (safe defaults: off if config lacks attrs) ---
        ring_weight = float(getattr(self.config, "ring_weight", 0.0))        # 0.0 => behaves like your original code
        ring_dilate_vox = int(getattr(self.config, "ring_dilate_vox", 0))    # 0 => no ring

        ring_mask = None
        if ring_weight > 0.0 and ring_dilate_vox > 0:
            # Use channel 0
            t = tumor_mask[0]
            dil = binary_dilation(t, iterations=ring_dilate_vox)
            ring_mask = np.logical_and(dil, np.logical_not(t))
            # ring_mask shape: (X, Y, Z)

        scored_patches: List[Tuple[int, int]] = []

        for idx, location in enumerate(self.locations):
            index_ini = location[:3]
            index_end = location[3:]

            x0, y0, z0 = int(index_ini[0]), int(index_ini[1]), int(index_ini[2])
            x1, y1, z1 = int(index_end[0]), int(index_end[1]), int(index_end[2])

            patch_tumor = tumor_mask[:, x0:x1, y0:y1, z0:z1]
            tumor_count = int(patch_tumor.sum())

            if tumor_count < self.config.min_tumor_voxels:
                continue

            if ring_mask is not None:
                ring_count = int(ring_mask[x0:x1, y0:y1, z0:z1].sum())
                score = tumor_count + ring_weight * ring_count
            else:
                score = tumor_count

            # keep score as int for compatibility (or keep float if you prefer)
            scored_patches.append((idx, int(score)))

        scored_patches.sort(key=lambda x: x[1], reverse=True)
        return scored_patches
    
    def _select_best_patches(self) -> List[Tuple[int, int]]:
        multilabel_mask = self.subject[self.mask_name].data.numpy()  # type: ignore
        total_tumor = int((multilabel_mask == self.config.tumor_label).sum())
        adaptive_max = _compute_adaptive_max_patches(total_tumor, self.config)

        scored_patches = self._score_all_patches()
        if adaptive_max == 0 or len(scored_patches) == 0:
            return []

        # Pool más grande que K para diversidad
        pool_size = min(len(scored_patches), adaptive_max * max(1, self.config.top_pool_factor))
        pool = scored_patches[:pool_size]

        if (not self.config.weighted_pool_sampling) or pool_size <= adaptive_max:
            return pool[:adaptive_max]

        # Samplear K desde pool con prob ~ score
        scores = np.array([s for _, s in pool], dtype=np.float64)
        probs = scores / scores.sum()
        chosen_idx = np.random.choice(pool_size, size=adaptive_max, replace=False, p=probs)
        return [pool[i] for i in chosen_idx]
    
    def __len__(self) -> int:
        """Return number of selected patches."""
        return len(self._selected_patches)
    
    def __iter__(self):
        """
        Iterate over selected patches in SHUFFLED order.
        
        CRITICAL: Shuffling ensures diversity across epochs when used with
        Queue's islice, which takes the first N patches from the iterator.
        Without shuffling, the same patches would be selected every epoch.
        
        Yields:
            TorchIO Subject patches with aligned CT and mask data
        """
        if len(self._selected_patches) == 0:
            return
        
        # Shuffle indices for epoch diversity
        indices = np.random.permutation(len(self._selected_patches))
        
        for i in indices:
            patch_idx, _score = self._selected_patches[i]
            location = self.locations[patch_idx]
            index_ini = tuple(location[:3].tolist())
            
            # crop() extracts aligned patches from ALL images (ct + mask)
            patch_subject = self.crop(self.subject, index_ini, tuple(self.patch_size.tolist()))  # type: ignore
            yield patch_subject


class AdaptiveTumorSamplerWrapper(tio.data.sampler.PatchSampler):
    """
    Queue-compatible wrapper for AdaptiveTumorSampler.
    
    tio.Queue expects a callable sampler: sampler(subject) → iterator.
    This wrapper stores configuration and creates a fresh sampler for each subject.
    
    The wrapper ensures:
    - Fresh sampler instance per subject (no state leakage)
    - Shuffled iteration order (epoch diversity via Queue's islice)
    - Configurable quality and adaptive parameters
    
    Example:
        >>> wrapper = AdaptiveTumorSamplerWrapper(
        ...     patch_size=(96, 96, 64),
        ...     patch_overlap=(48, 48, 32),
        ...     config=AdaptiveSamplerConfig(min_tumor_voxels=50)
        ... )
        >>> queue = tio.Queue(dataset, max_length=200, samples_per_volume=4, sampler=wrapper)
    """
    
    def __init__(
        self,
        patch_size: Tuple[int, int, int],
        patch_overlap: Tuple[int, int, int] = (0, 0, 0),
        mask_name: str = 'mask',
        config: Optional[AdaptiveSamplerConfig] = None,
    ):
        """
        Args:
            patch_size: Size of patches to extract (H, W, D)
            patch_overlap: Overlap between adjacent patches (H, W, D)
            mask_name: Key for the mask in subject
            config: Adaptive sampling configuration
        """
        super().__init__(patch_size)
        self.patch_overlap = patch_overlap
        self.mask_name = mask_name
        self.config = config or AdaptiveSamplerConfig()
    
    def __call__(self, subject: tio.Subject):
        """
        Create an AdaptiveTumorSampler for the subject and return its iterator.
        
        This is the interface expected by tio.Queue._fill().
        
        Args:
            subject: TorchIO Subject to sample patches from
            
        Returns:
            Iterator yielding patch Subjects (shuffled for diversity)
        """
        sampler = AdaptiveTumorSampler(
            subject=subject,
            patch_size=tuple(self.patch_size.tolist()),  # type: ignore
            patch_overlap=self.patch_overlap,
            mask_name=self.mask_name,
            config=self.config,
        )
        return iter(sampler)
