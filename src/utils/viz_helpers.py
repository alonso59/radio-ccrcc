"""Visualization helper functions for medical imaging and ML metrics."""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from .classification_metrics import (
    compute_binary_roc,
    compute_confusion_matrix,
    compute_multiclass_ovr_roc,
)
from .imaging import (
    build_mosaic as _build_mosaic,
    extract_tumor_boundary,
    squeeze_volume as _squeeze_volume,
    window_ct_pair as _window_ct_pair,
)


@dataclass(frozen=True)
class _ReconstructionPanels:
    input_mosaic: np.ndarray
    reconstruction_mosaic: np.ndarray
    diff_mosaic: np.ndarray
    input_mid: np.ndarray
    reconstruction_mid: np.ndarray
    diff_mid: np.ndarray
    boundary_mosaic: Optional[np.ndarray]
    boundary_mid: Optional[np.ndarray]

def create_volume_mosaic(
    volume: torch.Tensor,
    grid_size: int = 6,
    max_slices: int = 64,
    center_based: bool = True,
    boundary_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Create a slice mosaic; returns (mosaic, boundary_mosaic|None)."""
    volume_3d = _squeeze_volume(volume)
    mosaic, indices = _build_mosaic(
        volume_3d,
        grid_size=grid_size,
        max_slices=max_slices,
        center_based=center_based,
    )

    if boundary_mask is None:
        return mosaic, None

    boundary_3d = _squeeze_volume(
        torch.from_numpy(boundary_mask) if isinstance(boundary_mask, np.ndarray) else boundary_mask
    )
    if boundary_3d.shape != volume_3d.shape:
        raise ValueError(
            "Boundary mask shape must match the volume shape after squeezing. "
            f"Got volume {tuple(volume_3d.shape)} and boundary {tuple(boundary_3d.shape)}"
        )

    boundary_mosaic, _ = _build_mosaic(
        boundary_3d.to(volume_3d.device),
        grid_size=grid_size,
        max_slices=max_slices,
        center_based=center_based,
        indices=indices,
    )

    return mosaic, boundary_mosaic


def create_reconstruction_figure(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Figure:
    """Create reconstruction mosaic + mid-slice comparison figure."""
    panels = _prepare_reconstruction_panels(x, x_hat, mask)

    fig = plt.figure(figsize=(18, 10), dpi=120, constrained_layout=True, frameon=False)
    grid = fig.add_gridspec(2, 3, height_ratios=[2, 1])
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[0, 1])
    ax2 = fig.add_subplot(grid[0, 2])
    ax3 = fig.add_subplot(grid[1, 0])
    ax4 = fig.add_subplot(grid[1, 1])
    ax5 = fig.add_subplot(grid[1, 2])

    _draw_image_panel(ax0, panels.input_mosaic, "Input", panels.boundary_mosaic, 1.2)
    _draw_image_panel(ax1, panels.reconstruction_mosaic, "Reconstruction", panels.boundary_mosaic, 1.2)

    diff_max = max(float(panels.diff_mosaic.max()), 1e-6)
    im_mosaic = ax2.imshow(panels.diff_mosaic, cmap="jet", vmin=0, vmax=diff_max)
    ax2.axis("off")
    fig.colorbar(im_mosaic, ax=ax2, fraction=0.046, pad=0.04)

    _draw_image_panel(ax3, panels.input_mid, boundary=panels.boundary_mid, linewidth=1.8)
    _draw_image_panel(ax4, panels.reconstruction_mid, boundary=panels.boundary_mid, linewidth=1.8)

    im_mid = ax5.imshow(panels.diff_mid, cmap="jet", vmin=0, vmax=diff_max)
    ax5.axis("off")
    fig.colorbar(im_mid, ax=ax5, fraction=0.046, pad=0.04)
    return fig


def create_confusion_matrix_figure(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    class_names: List[str]
) -> Figure:
    """Create a normalized confusion matrix figure."""
    cm, cm_norm = compute_confusion_matrix(y_true, y_pred, num_classes=len(class_names))
    fig, ax = plt.subplots(figsize=(7, 6), dpi=120, constrained_layout=True)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Confusion Matrix (Normalized)"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j, i,
                f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if cm_norm[i, j] > 0.5 else "black"
            )

    return fig


def create_binary_roc_figure(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[Figure, dict]:
    """Create ROC curve for binary classification."""
    roc = compute_binary_roc(y_true, y_scores)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=120, constrained_layout=True)

    ax.plot(roc["fpr"], roc["tpr"], lw=2, label=f"AUC = {roc['auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.scatter(
        [1 - roc["specificity"]],
        [roc["sensitivity"]],
        c="red",
        s=50,
        zorder=5,
        edgecolors="black",
    )

    ax.annotate(
        (
            f"Sens: {roc['sensitivity']:.3f}\n"
            f"Spec: {roc['specificity']:.3f}\n"
            f"Thresh: {roc['threshold']:.3f}"
        ),
        xy=(1 - roc["specificity"], roc["sensitivity"]),
        xytext=(0.6, 0.2),
        arrowprops=dict(arrowstyle="->", lw=1.5),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8)
    )

    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC Curve",
        xlim=[-0.02, 1.02],
        ylim=[-0.02, 1.02]
    )
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    metrics = {
        "auc": roc["auc"],
        "sensitivity": roc["sensitivity"],
        "specificity": roc["specificity"],
        "threshold": roc["threshold"],
    }

    return fig, metrics


def create_multiclass_roc_figure(
    y_true: np.ndarray, 
    y_scores: np.ndarray, 
    class_names: Optional[List[str]] = None
) -> Tuple[Figure, float]:
    """Create One-vs-Rest ROC curves."""
    curves, macro_auc = compute_multiclass_ovr_roc(y_true, y_scores, class_names)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=120, constrained_layout=True)

    for curve in curves:
        ax.plot(curve["fpr"], curve["tpr"], lw=1.5, label=f"{curve['label']} (AUC={curve['auc']:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC Curves (One-vs-Rest)",
        xlim=[-0.02, 1.02],
        ylim=[-0.02, 1.02]
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    return fig, macro_auc


def create_umap_figure(
    latents: List[np.ndarray],
    labels: Optional[List] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean"
) -> Figure:
    """Create a 2D UMAP embedding figure."""
    from umap import UMAP

    all_latents = np.concatenate(latents, axis=0)
    num_samples = all_latents.shape[0]
    latents_flat = all_latents.reshape(num_samples, -1)

    logging.info(
        f"Running UMAP on {num_samples} samples with {latents_flat.shape[1]} dimensions..."
    )

    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric=metric
    )
    embedding = reducer.fit_transform(latents_flat)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150, constrained_layout=True)

    if labels is not None:
        labels_arr = np.concatenate(labels) if isinstance(labels, list) else np.asarray(labels)
        unique_classes = np.unique(labels_arr)
        n_classes = len(unique_classes)

        cmap = plt.cm.get_cmap("jet", n_classes)
        scatter = ax.scatter(
            embedding[:, 0],  # type: ignore
            embedding[:, 1],  # type: ignore
            c=labels_arr,
            cmap=cmap,
            alpha=0.7,
            s=40,
            edgecolors="black",
            linewidths=0.5
        )

        cbar = plt.colorbar(scatter, ax=ax, ticks=unique_classes)
        cbar.set_label("Class Label", rotation=90, fontsize=11)

        for class_id in unique_classes:
            ax.scatter(
                [], [],
                c=[cmap(int(class_id))],
                s=80,
                label=f"Class {int(class_id)}",
                edgecolors="black",
                linewidths=0.5
            )
        ax.legend(loc="best", framealpha=0.9, fontsize=10)
    else:
        ax.scatter(
            embedding[:, 0],  # type: ignore
            embedding[:, 1],  # type: ignore
            alpha=0.6,
            s=40,
            c="steelblue",
            edgecolors="black",
            linewidths=0.5
        )

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(f"Latent Space UMAP (n={num_samples})", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")

    return fig


def _prepare_reconstruction_panels(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> _ReconstructionPanels:
    input_volume = _squeeze_volume(x)
    reconstruction_volume = _squeeze_volume(x_hat)
    input_volume, reconstruction_volume = _window_ct_pair(
        input_volume, reconstruction_volume
    )

    input_mosaic, indices = _build_mosaic(input_volume)
    reconstruction_mosaic, _ = _build_mosaic(reconstruction_volume, indices=indices)

    boundary_mosaic = None
    boundary_mid = None
    if mask is not None:
        boundary_volume = extract_tumor_boundary(mask, tumor_label=2)
        boundary_mosaic_tensor, _ = _build_mosaic(boundary_volume, indices=indices)
        boundary_mosaic = boundary_mosaic_tensor.cpu().numpy()
        boundary_mid = boundary_volume[input_volume.shape[0] // 2]

    mid_idx = input_volume.shape[0] // 2
    input_mid = input_volume[mid_idx].cpu().numpy()
    reconstruction_mid = reconstruction_volume[mid_idx].cpu().numpy()

    return _ReconstructionPanels(
        input_mosaic=input_mosaic.cpu().numpy(),
        reconstruction_mosaic=reconstruction_mosaic.cpu().numpy(),
        diff_mosaic=torch.abs(input_mosaic - reconstruction_mosaic).cpu().numpy(),
        input_mid=input_mid,
        reconstruction_mid=reconstruction_mid,
        diff_mid=np.abs(input_mid - reconstruction_mid),
        boundary_mosaic=boundary_mosaic,
        boundary_mid=boundary_mid,
    )


def _draw_image_panel(
    ax,
    image: np.ndarray,
    title: Optional[str] = None,
    boundary: Optional[np.ndarray] = None,
    linewidth: float = 1.2,
) -> None:
    ax.imshow(image, cmap="gray")
    if boundary is not None:
        ax.contour(boundary, levels=[0.5], colors="yellow", linewidths=linewidth)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
