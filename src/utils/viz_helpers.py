"""Visualization helper functions for medical imaging and ML metrics."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
from matplotlib.figure import Figure


# ========================== Volume Visualization ==========================

def create_volume_mosaic(
    volume: torch.Tensor, 
    grid_size: int = 4, 
    max_slices: int = 64, 
    center_based: bool = True
) -> torch.Tensor:
    """
    Create a mosaic grid from volume slices.
    
    Args:
        volume: Volume tensor [D, H, W]
        grid_size: Number of rows/cols in the grid
        max_slices: Maximum number of slices to display
        center_based: If True, sample around center; else sample uniformly
        
    Returns:
        Mosaic image tensor [H*grid_size, W*grid_size]
    """
    D, H, W = volume.shape
    n_slices = min(max_slices, D)
    total_slots = grid_size * grid_size
    
    # Sample slices
    if center_based:
        center_idx = D // 2
        half_range = n_slices // 2
        start_idx = max(0, center_idx - half_range)
        end_idx = min(D, start_idx + n_slices)
        if end_idx - start_idx < n_slices:
            start_idx = max(0, end_idx - n_slices)
        indices = torch.arange(start_idx, end_idx, device=volume.device)[:n_slices]
    else:
        indices = torch.linspace(0, D - 1, steps=n_slices, device=volume.device)
        indices = indices.round().long().clamp(0, D - 1)
    
    slices = volume[indices]
    
    # Pad or truncate to fill grid
    if n_slices < total_slots:
        padding = torch.zeros(
            total_slots - n_slices, H, W, 
            device=volume.device, 
            dtype=volume.dtype
        )
        slices = torch.cat([slices, padding], dim=0)
    elif n_slices > total_slots:
        slices = slices[:total_slots]
    
    # Reshape to grid [grid_size*H, grid_size*W]
    return slices.view(grid_size, grid_size, H, W) \
                 .permute(0, 2, 1, 3) \
                 .reshape(grid_size * H, grid_size * W)


def create_reconstruction_figure(
    x: torch.Tensor, 
    x_hat: torch.Tensor,
    norm_stats: Optional[dict] = None
) -> Figure:
    """
    Create volume reconstruction comparison figure.
    
    Args:
        x: Input volume [D, H, W] (IQR-normalized)
        x_hat: Reconstructed volume [D, H, W] (IQR-normalized)
        norm_stats: Normalization statistics dict with keys: median, iqr
                    If provided, denormalizes to HU space for visualization
        
    Returns:
        Matplotlib figure with mosaics (top row) and middle slice comparison (bottom row)
    """
    # Denormalize to HU space if stats available
    if norm_stats is not None:
        iqr = norm_stats['iqr']
        median = norm_stats['median']
        x = x * iqr + median
        x_hat = x_hat * iqr + median
        vmin, vmax = -200, 300  # HU range
    else:
        # Fallback: auto-detect range from data
        vmin = min(x.min().item(), x_hat.min().item())
        vmax = max(x.max().item(), x_hat.max().item())
    
    # Create mosaics
    x_mosaic = create_volume_mosaic(x)
    xhat_mosaic = create_volume_mosaic(x_hat)
    diff_mosaic = (x_mosaic - xhat_mosaic).abs()
    
    # Get middle slice
    D = x.shape[0]
    mid_idx = D // 2
    x_mid = x[mid_idx].cpu().numpy()
    xhat_mid = x_hat[mid_idx].cpu().numpy()
    diff_mid = np.abs(x_mid - xhat_mid)
    
    # Create figure with 2 rows: mosaics (top), middle slices (bottom)
    fig = plt.figure(figsize=(18, 10), dpi=120, constrained_layout=True, frameon=False)
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1])
    
    # Top row - Mosaics
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(x_mosaic.cpu().numpy(), cmap="gray", vmin=vmin, vmax=vmax)
    ax0.set_title("Input", fontsize=14, fontweight='bold')
    ax0.axis("off")
    
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(xhat_mosaic.cpu().numpy(), cmap="gray", vmin=vmin, vmax=vmax)
    ax1.set_title("Reconstruction", fontsize=14, fontweight='bold')
    ax1.axis("off")
    
    ax2 = fig.add_subplot(gs[0, 2])
    # Auto-scale difference range using 95th percentile (robust to outliers)
    diff_vmax = torch.quantile(diff_mosaic, 0.95).item()
    im_mosaic = ax2.imshow(diff_mosaic.cpu().numpy(), cmap="jet", vmin=0, vmax=diff_vmax)
    ax2.set_title("Difference", fontsize=14, fontweight='bold')
    ax2.axis("off")
    fig.colorbar(im_mosaic, ax=ax2, fraction=0.046, pad=0.04)
    
    # Bottom row - Middle slices
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(x_mid, cmap="gray", vmin=vmin, vmax=vmax)
    ax3.axis("off")
    
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(xhat_mid, cmap="gray", vmin=vmin, vmax=vmax)
    ax4.axis("off")
    
    ax5 = fig.add_subplot(gs[1, 2])
    # Auto-scale using 95th percentile
    diff_mid_vmax = np.percentile(diff_mid, 95)
    im_mid = ax5.imshow(diff_mid, cmap="jet", vmin=0, vmax=diff_mid_vmax)
    ax5.axis("off")
    fig.colorbar(im_mid, ax=ax5, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


# ========================== Classification Metrics ==========================

def create_confusion_matrix_figure(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    class_names: List[str]
) -> Figure:
    """
    Create confusion matrix visualization.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        Matplotlib figure
    """
    num_classes = len(class_names)
    
    # Build confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true.astype(int), y_pred.astype(int)), 1)
    
    # Normalize by row
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(7, 6), dpi=120, constrained_layout=True)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Confusion Matrix (Normalized)"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    
    # Add text annotations
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j, i, 
                f"{cm[i, j]}\n({cm_norm[i, j]:.2f})", 
                ha="center", 
                va="center", 
                fontsize=9,
                color="white" if cm_norm[i, j] > 0.5 else "black"
            )
    
    return fig


# ========================== ROC Curve Analysis ==========================

def _compute_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC curve for binary classification."""
    order = np.argsort(-y_score)
    y_sorted = y_true[order].astype(np.float64)
    scores_sorted = y_score[order]
    
    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Cannot compute ROC curve with only one class present")
    
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1.0 - y_sorted)
    tpr = tp / n_pos
    fpr = fp / n_neg
    
    # Add origin
    fpr = np.concatenate([[0.0], fpr])
    tpr = np.concatenate([[0.0], tpr])
    thresholds = np.concatenate([[np.inf], scores_sorted])
    
    return fpr, tpr, thresholds


def _compute_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Compute area under ROC curve using trapezoidal rule."""
    return float(np.trapz(tpr, fpr))


def _find_optimal_threshold(
    fpr: np.ndarray, 
    tpr: np.ndarray, 
    thresholds: np.ndarray
) -> Tuple[float, float, float]:
    """Find optimal threshold using Youden's J statistic."""
    j_scores = tpr - fpr
    optimal_idx = int(np.argmax(j_scores))
    return (
        float(thresholds[optimal_idx]),
        float(tpr[optimal_idx]),
        float(1.0 - fpr[optimal_idx])
    )


def create_binary_roc_figure(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[Figure, dict]:
    """
    Create binary classification ROC curve figure.
    
    Args:
        y_true: True binary labels
        y_scores: Prediction scores
        
    Returns:
        Tuple of (figure, metrics_dict)
    """
    scores = y_scores.reshape(-1) if y_scores.ndim == 1 else y_scores[:, -1]
    
    fpr, tpr, thresholds = _compute_roc_curve(y_true, scores)
    auc = _compute_auc(fpr, tpr)
    threshold, sensitivity, specificity = _find_optimal_threshold(fpr, tpr, thresholds)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120, constrained_layout=True)
    
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.scatter([1 - specificity], [sensitivity], c="red", s=50, zorder=5, edgecolors='black')
    
    # Optimal point annotation
    ax.annotate(
        f"Sens: {sensitivity:.3f}\nSpec: {specificity:.3f}\nThresh: {threshold:.3f}",
        xy=(1 - specificity, sensitivity),
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
        "auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "threshold": threshold
    }
    
    return fig, metrics


def create_multiclass_roc_figure(
    y_true: np.ndarray, 
    y_scores: np.ndarray, 
    class_names: Optional[List[str]] = None
) -> Tuple[Figure, float]:
    """
    Create multiclass One-vs-Rest ROC curves figure.
    
    Args:
        y_true: True labels
        y_scores: Prediction scores [n_samples, n_classes]
        class_names: Optional list of class names
        
    Returns:
        Tuple of (figure, macro_auc)
    """
    num_classes = y_scores.shape[1]
    aucs = []
    
    fig, ax = plt.subplots(figsize=(7, 6), dpi=120, constrained_layout=True)
    
    for c in range(num_classes):
        y_binary = (y_true == c).astype(int)
        
        try:
            fpr, tpr, _ = _compute_roc_curve(y_binary, y_scores[:, c])
            auc_c = _compute_auc(fpr, tpr)
        except ValueError:
            continue
        
        aucs.append(auc_c)
        label = class_names[c] if class_names and c < len(class_names) else f"Class {c}"
        ax.plot(fpr, tpr, lw=1.5, label=f"{label} (AUC={auc_c:.3f})")
    
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
    
    macro_auc = float(np.mean(aucs)) if aucs else 0.0
    
    return fig, macro_auc


# ========================== Latent Space Visualization ==========================

def create_umap_figure(
    latents: List[np.ndarray],
    labels: Optional[List] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean"
) -> Figure:
    """
    Create UMAP visualization of latent space.
    
    Args:
        latents: List of latent vectors [B, CH, D, H, W] from batches
        labels: Optional labels for coloring points
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        metric: Distance metric for UMAP
        
    Returns:
        Matplotlib figure with 2D UMAP embedding
    """
    from umap import UMAP

    # Concatenate and flatten
    all_latents = np.concatenate(latents, axis=0)
    B = all_latents.shape[0]
    latents_flat = all_latents.reshape(B, -1)
    
    print(f"Running UMAP on {B} samples with {latents_flat.shape[1]} dimensions...")
    
    # Fit UMAP
    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric=metric,
        random_state=42
    )
    embedding = reducer.fit_transform(latents_flat)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150, constrained_layout=True)
    
    if labels is not None:
        labels_arr = np.concatenate(labels) if isinstance(labels, list) else np.array(labels)
        unique_classes = np.unique(labels_arr)
        n_classes = len(unique_classes)
        
        cmap = plt.cm.get_cmap('tab10', n_classes)
        
        scatter = ax.scatter(
            embedding[:, 0],  # type: ignore 
            embedding[:, 1],  # type: ignore 
            c=labels_arr,
            cmap=cmap,
            alpha=0.7,
            s=40,
            edgecolors='black',
            linewidths=0.5
        )
        
        cbar = plt.colorbar(scatter, ax=ax, ticks=unique_classes)
        cbar.set_label("Class Label", rotation=90, fontsize=11)
        
        # Legend
        for class_id in unique_classes:
            ax.scatter(
                [], [], 
                c=[cmap(int(class_id))],
                s=80,
                label=f'Class {int(class_id)}',
                edgecolors='black',
                linewidths=0.5
            )
        ax.legend(loc='best', framealpha=0.9, fontsize=10)
    else:
        ax.scatter(
            embedding[:, 0],  # type: ignore 
            embedding[:, 1],  # type: ignore 
            alpha=0.6,
            s=40,
            c='steelblue',
            edgecolors='black',
            linewidths=0.5
        )
    
    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(f"Latent Space UMAP (n={B})", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    return fig
