"""
Custom patch samplers for TorchIO.
Provides adaptive, quality-aware tumor-focused sampling strategies for medical imaging.

Key Features:
- Adaptive patch count based on tumor size (avoid redundancy for small tumors)
- Region-aware scoring (core, border/peritumor, optional kidney context)
- Diversity-aware selection (reduce redundant tumor-core duplicates)
- Shuffle on iteration (diversity across epochs when used with Queue)
"""

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torchio as tio


logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
@dataclass
class AdaptiveSamplerConfig:
    """Configuration for adaptive tumor patch sampling."""
    min_tumor_voxels: int = 50  # Minimum tumor voxels required per patch (quality threshold).
    voxels_per_patch: int = 500  # Tumor voxels that 'justify' one patch (for adaptive count calculation).
    max_patches_cap: int = 8  # Maximum patches per volume (prevents oversampling large tumors).
    tumor_label: int = 2  # Label value for tumor in multilabel mask.
    top_pool_factor: int = 1
    weighted_pool_sampling: bool = True
    ring_dilate_vox: int = 1
    ring_weight: float = 0.1
    kidney_label: int = 1
    kidney_weight: float = 0.10
    border_weight: float = 0.20
    selection_pool_factor: int = 4
    diversity_weight: float = 0.35
    iou_redundancy_weight: float = 0.65
    distance_redundancy_weight: float = 0.35
    audit_enabled: bool = False


@dataclass(frozen=True)
class _RegionMasks:
    tumor: np.ndarray
    tumor_core: np.ndarray
    tumor_border: np.ndarray
    ring: np.ndarray
    kidney: np.ndarray
    total_tumor_voxels: int
    kidney_present: bool


@dataclass(frozen=True)
class _PatchCandidate:
    index: int
    location: Tuple[int, int, int, int, int, int]
    start: Tuple[int, int, int]
    end: Tuple[int, int, int]
    center: Tuple[float, float, float]
    patch_volume: int
    tumor_count: int
    core_count: int
    border_count: int
    ring_count: int
    kidney_count: int
    tumor_fraction: float
    core_fraction: float
    border_fraction: float
    ring_fraction: float
    kidney_fraction: float
    base_score: float = 0.0
    region_tag: str = "core"


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
        padding_tuple: Tuple[int, int, int, int, int, int] = (
            (padding_needed[0] + 1) // 2,
            padding_needed[0] // 2,
            (padding_needed[1] + 1) // 2,
            padding_needed[1] // 2,
            (padding_needed[2] + 1) // 2,
            padding_needed[2] // 2,
        )

        pad_transform = tio.Pad(padding_tuple, padding_mode=-200)
        subject = pad_transform(subject)
        logger.debug("Padded subject from %s to %s", spatial_shape, subject.spatial_shape)

    return subject


def _compute_adaptive_max_patches(total_tumor_voxels: int, config: AdaptiveSamplerConfig) -> int:
    """
    Compute adaptive maximum patches based on tumor size.

    Logic:
    - Small tumors → 1 patch (avoid redundancy)
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
    adaptive_max = max(1, adaptive_max)
    adaptive_max = min(adaptive_max, config.max_patches_cap)
    return adaptive_max


def binary_dilation(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Simple binary dilation using a 3D structuring element."""
    from scipy.ndimage import binary_dilation as scipy_binary_dilation

    if iterations <= 0:
        return mask.astype(bool, copy=True)

    struct = np.ones((3, 3, 3), dtype=bool)
    return scipy_binary_dilation(mask, structure=struct, iterations=iterations)


def binary_erosion(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Simple binary erosion using a 3D structuring element."""
    from scipy.ndimage import binary_erosion as scipy_binary_erosion

    if iterations <= 0:
        return mask.astype(bool, copy=True)

    struct = np.ones((3, 3, 3), dtype=bool)
    return scipy_binary_erosion(mask, structure=struct, iterations=iterations)


def _safe_fraction(count: int, total: int) -> float:
    """Return a non-negative fraction with robust zero handling."""
    if total <= 0:
        return 0.0
    return max(0.0, float(count) / float(total))


def _sanitize_score(score: float) -> float:
    """Clamp invalid or negative scores to zero."""
    if not np.isfinite(score):
        return 0.0
    return max(0.0, float(score))


# =============================================================================
# Main Sampler Classes
# =============================================================================


class AdaptiveTumorSampler(tio.data.GridSampler):
    """
    Adaptive quality-aware tumor patch sampler.

    Key improvements over basic GridSampler:
    1. Quality threshold: Patches must have meaningful tumor content
    2. Adaptive quantity: Patch count scales with tumor size
    3. Region-aware scoring: Core, border, ring, and optional kidney context
    4. Diversity-aware selection: Reduce redundant near-duplicate patches
    5. Shuffle on iteration: Different patches selected each epoch

    This sampler is designed for use with tio.Queue via AdaptiveTumorSamplerWrapper.
    """

    _REGION_PRIORITY: Tuple[str, ...] = ("core", "border", "context")
    _SOFT_TARGET_FILL_ORDER: Tuple[str, ...] = ("border", "context", "core")

    def __init__(
        self,
        subject: tio.Subject,
        patch_size: Tuple[int, int, int],
        patch_overlap: Tuple[int, int, int] = (0, 0, 0),
        mask_name: str = "mask",
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
        self._selection_audit: Optional[Dict[str, object]] = None

        subject = _ensure_min_size(subject, patch_size)
        super().__init__(subject, patch_size, patch_overlap)

        self._region_masks = self._prepare_region_masks()
        self._selected_patches = self._select_best_patches()

    def _prepare_region_masks(self) -> _RegionMasks:
        """Create tumor/core/border/ring/kidney masks used for candidate scoring."""
        multilabel_mask = self.subject[self.mask_name].data.numpy()  # type: ignore
        if multilabel_mask.ndim == 4:
            label_volume = multilabel_mask[0]
        else:
            label_volume = multilabel_mask

        tumor_mask = label_volume == int(self.config.tumor_label)
        total_tumor_voxels = int(tumor_mask.sum())

        tumor_core_mask = np.zeros_like(tumor_mask, dtype=bool)
        tumor_border_mask = np.zeros_like(tumor_mask, dtype=bool)
        ring_mask = np.zeros_like(tumor_mask, dtype=bool)

        if total_tumor_voxels > 0:
            eroded = binary_erosion(tumor_mask, iterations=1)
            if eroded.any():
                tumor_core_mask = eroded
                tumor_border_mask = np.logical_and(tumor_mask, np.logical_not(eroded))
            else:
                tumor_core_mask = tumor_mask.copy()
                tumor_border_mask = np.zeros_like(tumor_mask, dtype=bool)

            ring_dilate_vox = max(0, int(getattr(self.config, "ring_dilate_vox", 0)))
            if ring_dilate_vox > 0:
                dilated = binary_dilation(tumor_mask, iterations=ring_dilate_vox)
                ring_mask = np.logical_and(dilated, np.logical_not(tumor_mask))

        kidney_mask = label_volume == int(getattr(self.config, "kidney_label", 1))
        kidney_present = bool(kidney_mask.any())
        if not kidney_present:
            kidney_mask = np.zeros_like(tumor_mask, dtype=bool)

        return _RegionMasks(
            tumor=tumor_mask,
            tumor_core=tumor_core_mask,
            tumor_border=tumor_border_mask,
            ring=ring_mask,
            kidney=kidney_mask,
            total_tumor_voxels=total_tumor_voxels,
            kidney_present=kidney_present,
        )

    def _build_candidate_metadata(self) -> List[_PatchCandidate]:
        """Extract metadata for every grid candidate before filtering and scoring."""
        if len(self.locations) == 0:
            return []

        candidates: List[_PatchCandidate] = []
        for idx, location in enumerate(self.locations):
            x0, y0, z0, x1, y1, z1 = [int(value) for value in location.tolist()]
            patch_volume = max(1, (x1 - x0) * (y1 - y0) * (z1 - z0))

            tumor_count = int(self._region_masks.tumor[x0:x1, y0:y1, z0:z1].sum())
            core_count = int(self._region_masks.tumor_core[x0:x1, y0:y1, z0:z1].sum())
            border_count = int(self._region_masks.tumor_border[x0:x1, y0:y1, z0:z1].sum())
            ring_count = int(self._region_masks.ring[x0:x1, y0:y1, z0:z1].sum())
            kidney_count = int(self._region_masks.kidney[x0:x1, y0:y1, z0:z1].sum())

            candidates.append(
                _PatchCandidate(
                    index=idx,
                    location=(x0, y0, z0, x1, y1, z1),
                    start=(x0, y0, z0),
                    end=(x1, y1, z1),
                    center=(
                        0.5 * (x0 + x1),
                        0.5 * (y0 + y1),
                        0.5 * (z0 + z1),
                    ),
                    patch_volume=patch_volume,
                    tumor_count=tumor_count,
                    core_count=core_count,
                    border_count=border_count,
                    ring_count=ring_count,
                    kidney_count=kidney_count,
                    tumor_fraction=_safe_fraction(tumor_count, patch_volume),
                    core_fraction=_safe_fraction(core_count, patch_volume),
                    border_fraction=_safe_fraction(border_count, patch_volume),
                    ring_fraction=_safe_fraction(ring_count, patch_volume),
                    kidney_fraction=_safe_fraction(kidney_count, patch_volume),
                )
            )

        return candidates

    def _score_candidate(self, candidate: _PatchCandidate) -> float:
        """Compute a float score that balances tumor density with morphological context."""
        score = np.sqrt(max(candidate.tumor_fraction, 0.0))
        score += max(0.0, float(getattr(self.config, "border_weight", 0.0))) * candidate.border_fraction
        score += max(0.0, float(getattr(self.config, "ring_weight", 0.0))) * candidate.ring_fraction

        if self._region_masks.kidney_present:
            score += max(0.0, float(getattr(self.config, "kidney_weight", 0.0))) * candidate.kidney_fraction

        return _sanitize_score(score)

    def _tag_candidate_region(self, candidate: _PatchCandidate) -> str:
        """Assign a coarse region tag used for soft balancing during selection."""
        core_signal = candidate.core_fraction
        border_signal = candidate.border_fraction + candidate.ring_fraction
        context_signal = candidate.kidney_fraction if self._region_masks.kidney_present else 0.0

        # Context is useful for representation learning, but if it dominates
        # every tumor-containing patch we lose core/border diversity. Reserve
        # the context tag for patches where kidney context clearly outweighs
        # tumor morphology and the tumor occupies a relatively small fraction.
        if (
            context_signal > max(core_signal, border_signal)
            and candidate.tumor_fraction < 0.25 * context_signal
        ):
            return "context"

        if border_signal > core_signal:
            return "border"
        if core_signal > 0.0:
            return "core"
        if context_signal > 0.0:
            return "context"
        if candidate.ring_fraction > 0.0:
            return "border"
        return "core"

    def _score_all_patches(self) -> List[Tuple[int, float]]:
        """
        Score all grid patches by region-aware tumor content.

        Returns:
            List of (patch_index, float_score) tuples for patches meeting the
            minimum tumor threshold, sorted by score descending.
        """
        candidates = self._build_candidate_metadata()
        scored_candidates: List[_PatchCandidate] = []

        for candidate in candidates:
            if candidate.tumor_count < self.config.min_tumor_voxels:
                continue

            base_score = self._score_candidate(candidate)
            if base_score <= 0.0:
                continue

            scored_candidates.append(
                _PatchCandidate(
                    **{
                        **candidate.__dict__,
                        "base_score": base_score,
                        "region_tag": self._tag_candidate_region(candidate),
                    }
                )
            )

        scored_candidates.sort(
            key=lambda candidate: (
                candidate.base_score,
                candidate.tumor_fraction,
                candidate.ring_fraction + candidate.border_fraction,
                candidate.kidney_fraction,
            ),
            reverse=True,
        )
        self._candidate_metadata_by_index = {candidate.index: candidate for candidate in scored_candidates}
        return [(candidate.index, candidate.base_score) for candidate in scored_candidates]

    def _get_scored_candidates(self) -> List[_PatchCandidate]:
        """Return scored candidate objects, keeping `_score_all_patches` as the source of truth."""
        self._score_all_patches()
        return list(getattr(self, "_candidate_metadata_by_index", {}).values())

    def _build_selection_pool(self, candidates: Sequence[_PatchCandidate], adaptive_max: int) -> List[_PatchCandidate]:
        """Create a larger working pool before diversity-aware selection."""
        if adaptive_max <= 0 or not candidates:
            return []

        effective_pool_factor = max(
            1,
            int(getattr(self.config, "top_pool_factor", 1)),
            int(getattr(self.config, "selection_pool_factor", 1)),
        )
        pool_size = min(len(candidates), adaptive_max * effective_pool_factor)
        pool = list(candidates[:pool_size])

        if not getattr(self.config, "weighted_pool_sampling", False) or pool_size <= adaptive_max:
            return pool

        scores = np.asarray([candidate.base_score for candidate in pool], dtype=np.float64)
        scores = np.clip(scores, a_min=0.0, a_max=None)
        if scores.sum() <= 0.0:
            return pool

        jitter = np.random.uniform(0.95, 1.05, size=pool_size)
        weighted_scores = scores * jitter
        if weighted_scores.sum() <= 0.0:
            weighted_scores = scores

        probabilities = weighted_scores / weighted_scores.sum()
        working_size = min(pool_size, max(adaptive_max, 2 * adaptive_max))
        sampled_indices = np.random.choice(pool_size, size=working_size, replace=False, p=probabilities)
        return [pool[index] for index in sampled_indices]

    def _compute_region_targets(self, candidates: Sequence[_PatchCandidate], target_count: int) -> Counter:
        """Compute soft region targets so border/context patches remain represented."""
        available_counts = Counter(candidate.region_tag for candidate in candidates)
        targets: Counter = Counter()

        if target_count <= 0 or not available_counts:
            return targets

        slots_remaining = target_count
        for region in self._REGION_PRIORITY:
            if slots_remaining <= 0:
                break
            if available_counts.get(region, 0) > 0:
                targets[region] += 1
                slots_remaining -= 1

        fill_cycle = [region for region in self._SOFT_TARGET_FILL_ORDER if available_counts.get(region, 0) > 0]
        fill_index = 0
        while slots_remaining > 0 and fill_cycle:
            region = fill_cycle[fill_index % len(fill_cycle)]
            targets[region] += 1
            slots_remaining -= 1
            fill_index += 1

        return targets

    def _compute_region_balance_bonus(
        self,
        candidate: _PatchCandidate,
        selected_regions: Counter,
        target_regions: Counter,
    ) -> float:
        """Reward candidates from underrepresented regions without enforcing hard quotas."""
        target = int(target_regions.get(candidate.region_tag, 0))
        if target <= 0:
            return 0.0

        current = int(selected_regions.get(candidate.region_tag, 0))
        shortfall = max(0, target - current)
        if shortfall <= 0:
            return 0.0

        return 0.10 * float(shortfall) / float(target)

    def _patch_iou(self, candidate: _PatchCandidate, other: _PatchCandidate) -> float:
        """Compute IoU between two patch boxes."""
        ax0, ay0, az0, ax1, ay1, az1 = candidate.location
        bx0, by0, bz0, bx1, by1, bz1 = other.location

        ix0, iy0, iz0 = max(ax0, bx0), max(ay0, by0), max(az0, bz0)
        ix1, iy1, iz1 = min(ax1, bx1), min(ay1, by1), min(az1, bz1)

        inter_x = max(0, ix1 - ix0)
        inter_y = max(0, iy1 - iy0)
        inter_z = max(0, iz1 - iz0)
        intersection = inter_x * inter_y * inter_z
        if intersection <= 0:
            return 0.0

        union = candidate.patch_volume + other.patch_volume - intersection
        return _safe_fraction(intersection, union)

    def _center_distance_similarity(self, candidate: _PatchCandidate, other: _PatchCandidate, patch_diagonal: float) -> float:
        """Distance-based similarity term used in the redundancy penalty."""
        center_a = np.asarray(candidate.center, dtype=np.float64)
        center_b = np.asarray(other.center, dtype=np.float64)
        center_distance = float(np.linalg.norm(center_a - center_b))
        if patch_diagonal <= 0.0:
            return 0.0
        return float(np.exp(-((center_distance / patch_diagonal) ** 2)))

    def _redundancy_penalty(
        self,
        candidate: _PatchCandidate,
        selected_candidates: Sequence[_PatchCandidate],
        patch_diagonal: float,
    ) -> float:
        """Compute maximal similarity to already selected patches."""
        if not selected_candidates:
            return 0.0

        iou_weight = max(0.0, float(getattr(self.config, "iou_redundancy_weight", 0.0)))
        distance_weight = max(0.0, float(getattr(self.config, "distance_redundancy_weight", 0.0)))
        weight_sum = iou_weight + distance_weight
        if weight_sum <= 0.0:
            return 0.0

        similarities = []
        for selected in selected_candidates:
            similarity = 0.0
            if iou_weight > 0.0:
                similarity += iou_weight * self._patch_iou(candidate, selected)
            if distance_weight > 0.0:
                similarity += distance_weight * self._center_distance_similarity(candidate, selected, patch_diagonal)
            similarities.append(similarity / weight_sum)

        return max(similarities, default=0.0)

    def _greedy_select_diverse_patches(
        self,
        candidates: Sequence[_PatchCandidate],
        target_count: int,
    ) -> List[_PatchCandidate]:
        """Select diverse patches using soft region balancing and redundancy penalties."""
        if target_count <= 0 or not candidates:
            return []

        target_count = min(target_count, len(candidates))
        target_regions = self._compute_region_targets(candidates, target_count)
        selected_candidates: List[_PatchCandidate] = []
        selected_regions: Counter = Counter()

        patch_diagonal = float(np.linalg.norm(np.asarray(self.patch_size, dtype=np.float64)))
        diversity_weight = max(0.0, float(getattr(self.config, "diversity_weight", 0.0)))

        remaining_candidates = list(candidates)
        while remaining_candidates and len(selected_candidates) < target_count:
            best_candidate: Optional[_PatchCandidate] = None
            best_utility = -np.inf
            best_base_score = -np.inf

            for candidate in remaining_candidates:
                redundancy = self._redundancy_penalty(candidate, selected_candidates, patch_diagonal)
                balance_bonus = self._compute_region_balance_bonus(candidate, selected_regions, target_regions)
                jitter = float(np.random.uniform(0.0, 1e-6))
                utility = candidate.base_score + balance_bonus - (diversity_weight * redundancy) + jitter

                if utility > best_utility:
                    best_candidate = candidate
                    best_utility = utility
                    best_base_score = candidate.base_score
                elif utility == best_utility and candidate.base_score > best_base_score:
                    best_candidate = candidate
                    best_base_score = candidate.base_score

            if best_candidate is None:
                break

            if best_utility <= 0.0:
                best_candidate = max(remaining_candidates, key=lambda candidate: candidate.base_score)

            selected_candidates.append(best_candidate)
            selected_regions[best_candidate.region_tag] += 1
            remaining_candidates = [candidate for candidate in remaining_candidates if candidate.index != best_candidate.index]

        return selected_candidates

    def _build_audit_summary(self, selected_candidates: Sequence[_PatchCandidate]) -> Optional[Dict[str, object]]:
        """Optionally summarize selection quality for debugging/auditing."""
        if not getattr(self.config, "audit_enabled", False):
            return None

        if not selected_candidates:
            return {
                "selected_count": 0,
                "region_counts": {},
                "tumor_coverage": 0.0,
                "mean_tumor_fraction": 0.0,
                "mean_pairwise_iou": 0.0,
                "mean_pairwise_center_distance": 0.0,
            }

        tumor_union = np.zeros_like(self._region_masks.tumor, dtype=bool)
        ious: List[float] = []
        distances: List[float] = []

        for candidate in selected_candidates:
            x0, y0, z0, x1, y1, z1 = candidate.location
            tumor_union[x0:x1, y0:y1, z0:z1] |= self._region_masks.tumor[x0:x1, y0:y1, z0:z1]

        for i, candidate in enumerate(selected_candidates):
            for other in selected_candidates[i + 1 :]:
                ious.append(self._patch_iou(candidate, other))
                distances.append(float(np.linalg.norm(np.asarray(candidate.center) - np.asarray(other.center))))

        selected_tumor_voxels = int(np.logical_and(tumor_union, self._region_masks.tumor).sum())
        total_tumor_voxels = max(1, self._region_masks.total_tumor_voxels)

        return {
            "selected_count": len(selected_candidates),
            "region_counts": dict(Counter(candidate.region_tag for candidate in selected_candidates)),
            "tumor_coverage": float(selected_tumor_voxels) / float(total_tumor_voxels),
            "mean_tumor_fraction": float(np.mean([candidate.tumor_fraction for candidate in selected_candidates])),
            "mean_pairwise_iou": float(np.mean(ious)) if ious else 0.0,
            "mean_pairwise_center_distance": float(np.mean(distances)) if distances else 0.0,
        }

    def _select_best_patches(self) -> List[Tuple[int, float]]:
        """Select diverse, informative patches while preserving current external API."""
        adaptive_max = _compute_adaptive_max_patches(self._region_masks.total_tumor_voxels, self.config)
        scored_candidates = self._get_scored_candidates()

        if adaptive_max == 0 or not scored_candidates:
            self._selection_audit = self._build_audit_summary([])
            return []

        pool = self._build_selection_pool(scored_candidates, adaptive_max)
        selected_candidates = self._greedy_select_diverse_patches(pool, adaptive_max)

        if len(selected_candidates) < adaptive_max:
            selected_indices = {candidate.index for candidate in selected_candidates}
            for candidate in scored_candidates:
                if candidate.index not in selected_indices:
                    selected_candidates.append(candidate)
                    selected_indices.add(candidate.index)
                if len(selected_candidates) >= adaptive_max:
                    break

        self._selection_audit = self._build_audit_summary(selected_candidates)
        if self._selection_audit is not None:
            logger.debug("Sampler audit: %s", self._selection_audit)

        return [(candidate.index, candidate.base_score) for candidate in selected_candidates]

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

        indices = np.random.permutation(len(self._selected_patches))
        for i in indices:
            patch_idx, _score = self._selected_patches[i]
            location = self.locations[patch_idx]
            index_ini = tuple(location[:3].tolist())

            patch_subject = self.crop(self.subject, index_ini, tuple(self.patch_size.tolist()))  # type: ignore
            yield patch_subject


class AdaptiveTumorSamplerWrapper(tio.data.sampler.PatchSampler):
    """
    Queue-compatible wrapper for AdaptiveTumorSampler.

    tio.Queue expects a callable sampler: sampler(subject) → iterator.
    This wrapper stores configuration and creates a fresh sampler for each subject.
    """

    def __init__(
        self,
        patch_size: Tuple[int, int, int],
        patch_overlap: Tuple[int, int, int] = (0, 0, 0),
        mask_name: str = "mask",
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
