"""Shared medical imaging utilities for visualization and training metrics."""

from typing import Mapping, Optional, Tuple, Union

import numpy as np
import torch
from scipy.ndimage import binary_erosion

DEFAULT_CT_WINDOW: Tuple[float, float] = (-200.0, 300.0)

TensorLike = Union[torch.Tensor, np.ndarray]


def squeeze_volume(volume: torch.Tensor) -> torch.Tensor:
    """Normalize a volume tensor to `[D, H, W]`."""
    if volume.ndim == 3:
        return volume
    if volume.ndim == 4 and volume.shape[0] == 1:
        return volume[0]
    if volume.ndim == 5 and volume.shape[0] == 1 and volume.shape[1] == 1:
        return volume[0, 0]
    raise ValueError(
        "Input volume must be 3D [D,H,W] or 4D/5D with singleton batch/channel "
        f"dimensions. Got shape: {tuple(volume.shape)}"
    )


def window_ct_volume(
    volume: torch.Tensor,
    norm_stats: Mapping[str, float],
    window: Tuple[float, float] = DEFAULT_CT_WINDOW,
) -> torch.Tensor:
    """Map an IQR-normalized CT volume into a windowed `[0, 1]` display range."""
    median, iqr = _get_norm_stats(norm_stats)
    window_min, window_max = window

    hu_volume = volume * iqr + median
    return torch.clamp((hu_volume - window_min) / (window_max - window_min), 0.0, 1.0)


def window_ct_pair(
    input_volume: torch.Tensor,
    reconstruction_volume: torch.Tensor,
    norm_stats: Mapping[str, float],
    window: Tuple[float, float] = DEFAULT_CT_WINDOW,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Window an input/reconstruction pair into the same CT display range."""
    return (
        window_ct_volume(input_volume, norm_stats, window=window),
        window_ct_volume(reconstruction_volume, norm_stats, window=window),
    )


def extract_tumor_boundary(mask: torch.Tensor, tumor_label: int = 2) -> np.ndarray:
    """Return a thin 2D in-plane tumor boundary mask."""
    mask_3d = squeeze_volume(mask)
    tumor_mask = mask_3d.cpu().numpy() == tumor_label

    structure = np.zeros((3, 3, 3), dtype=bool)
    structure[1, :, :] = True
    eroded = binary_erosion(tumor_mask, structure=structure)
    boundary = tumor_mask & np.logical_not(eroded)
    return boundary.astype(np.uint8)


def select_slice_indices(
    depth: int,
    num_slices: int,
    center_based: bool = True,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Select slice indices for a volume mosaic."""
    if depth <= 0:
        raise ValueError(f"Volume depth must be positive. Got depth={depth}")

    num_slices = max(1, min(num_slices, depth))

    if center_based:
        center_idx = depth // 2
        half_range = num_slices // 2
        start_idx = max(0, center_idx - half_range)
        end_idx = min(depth, start_idx + num_slices)
        if end_idx - start_idx < num_slices:
            start_idx = max(0, end_idx - num_slices)
        return torch.arange(start_idx, end_idx, device=device, dtype=torch.long)

    if num_slices == 1:
        return torch.zeros(1, device=device, dtype=torch.long)

    return (
        torch.linspace(0, depth - 1, steps=num_slices, device=device)
        .round()
        .long()
        .clamp(0, depth - 1)
    )


def build_mosaic(
    volume: TensorLike,
    grid_size: int = 6,
    max_slices: int = 64,
    center_based: bool = True,
    indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a `[grid_size * H, grid_size * W]` mosaic and return the used indices."""
    volume_3d = squeeze_volume(_as_tensor(volume))
    total_slots = grid_size * grid_size

    if indices is None:
        num_slices = min(volume_3d.shape[0], max_slices, total_slots)
        indices = select_slice_indices(
            depth=volume_3d.shape[0],
            num_slices=num_slices,
            center_based=center_based,
            device=volume_3d.device,
        )
    else:
        indices = indices.to(device=volume_3d.device, dtype=torch.long)
        if indices.ndim != 1:
            raise ValueError(f"Slice indices must be 1D. Got shape: {tuple(indices.shape)}")
        if indices.numel() > total_slots:
            indices = indices[:total_slots]

    slices = volume_3d.index_select(0, indices)
    return _compose_mosaic(slices, grid_size=grid_size, total_slots=total_slots), indices


def _as_tensor(volume: TensorLike) -> torch.Tensor:
    if isinstance(volume, np.ndarray):
        return torch.from_numpy(volume)
    if torch.is_tensor(volume):
        return volume
    raise TypeError(f"Expected a torch.Tensor or np.ndarray. Got {type(volume)!r}")


def _compose_mosaic(slices: torch.Tensor, grid_size: int, total_slots: int) -> torch.Tensor:
    num_slices, height, width = slices.shape

    if num_slices < total_slots:
        padding = torch.zeros(
            total_slots - num_slices,
            height,
            width,
            device=slices.device,
            dtype=slices.dtype,
        )
        slices = torch.cat([slices, padding], dim=0)
    elif num_slices > total_slots:
        slices = slices[:total_slots]

    return (
        slices.view(grid_size, grid_size, height, width)
        .permute(0, 2, 1, 3)
        .reshape(grid_size * height, grid_size * width)
    )


def _get_norm_stats(norm_stats: Mapping[str, float]) -> Tuple[float, float]:
    if norm_stats is None:
        raise ValueError(
            "CT windowing requires normalization stats with 'median' and 'iqr'. Got None."
        )

    if "median" not in norm_stats or "iqr" not in norm_stats:
        raise ValueError(
            "CT windowing requires normalization stats with 'median' and 'iqr'. "
            f"Got keys: {sorted(norm_stats.keys())}"
        )

    median = float(norm_stats["median"])
    iqr = float(norm_stats["iqr"])
    if iqr == 0:
        raise ValueError("CT windowing requires a non-zero 'iqr' normalization statistic.")
    return median, iqr
