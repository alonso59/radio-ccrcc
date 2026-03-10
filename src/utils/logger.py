"""Simplified TensorBoard logger for training workflows."""

import os
from datetime import datetime
from typing import Optional, List

import torch
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np

from . import viz_helpers


class TensorBoardLogger:
    """Lightweight TensorBoard logger with visualization utilities."""
    
    def __init__(
        self, 
        root: str = "outputs", 
        experiment_name: Optional[str] = None,
        parent_dir: Optional[str] = None,
        run_name: Optional[str] = None,
        add_timestamp: bool = True,
        use_date_structure: bool = True
    ):
        """
        Initialize TensorBoard logger.
        
        Args:
            root: Root directory for logs
            experiment_name: Name of the experiment
            parent_dir: Optional parent directory
            run_name: Alternative to experiment_name
            add_timestamp: Whether to add timestamp to log dir
            use_date_structure: Whether to use date-based folder structure
        """
        self.experiment_name = experiment_name or run_name or "experiment"
        self.log_dir = self._build_log_directory(root, parent_dir, add_timestamp, use_date_structure)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        if experiment_name:
            self.writer.add_text("experiment/name", experiment_name, 0)
    
    def _build_log_directory(self, root: str, parent_dir: Optional[str], 
                            add_timestamp: bool, use_date_structure: bool) -> str:
        """Construct log directory path."""
        now = datetime.now()
        parts = [root]
        if parent_dir:
            parts.append(parent_dir)
        
        if use_date_structure:
            parts.extend([now.strftime("%Y-%m-%d"), now.strftime("%H-%M-%S")])
        else:
            name = f"{self.experiment_name}_{now.strftime('%Y%m%d-%H%M%S')}" if add_timestamp else self.experiment_name
            parts.append(name)
        
        return os.path.join(*parts)

    # ========================== Scalar Logging ==========================
    
    def add_scalar(self, tag: str, value: float, step: int, split: Optional[str] = None) -> None:
        """Log a scalar metric to TensorBoard."""
        tag_name = f"{tag}/{split}" if split else tag
        self.writer.add_scalar(tag_name, value, step)

    def log_scalars(self, group: str, scalars: dict, step: int) -> None:
        """Log multiple related scalars efficiently."""
        for name, value in scalars.items():
            self.writer.add_scalar(f"{group}/{name}", value, step)
    
    # ==================== Volume Visualization ==========================
    
    def log_axial_figure(
        self, 
        x: torch.Tensor, 
        x_hat: torch.Tensor, 
        step: int, 
        tag: str = "val",
        norm_stats: Optional[dict] = None,
        mask: Optional[torch.Tensor] = None
    ) -> None:
        """
        Log volume reconstruction comparison with optional tumor boundary overlay.
        
        Args:
            x: Input volume
            x_hat: Reconstructed volume
            step: Current training step/epoch
            tag: Tag prefix for the figure
            norm_stats: Normalization statistics for HU space visualization
            mask: Optional segmentation mask for tumor boundary overlay
        """
        fig = viz_helpers.create_reconstruction_figure(x, x_hat, norm_stats=norm_stats, mask=mask)
        self.writer.add_figure(f"{tag}/AxialGrid", fig, global_step=step, close=True)
    
    # ==================== Classification Metrics =======================
    
    def log_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        class_names: List[str], 
        step: int, 
        tag: str = "val"
    ) -> None:
        """
        Log confusion matrix visualization.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            step: Current training step/epoch
            tag: Tag prefix for the figure
        """
        fig = viz_helpers.create_confusion_matrix_figure(y_true, y_pred, class_names)
        self.writer.add_figure(f"{tag}/ConfusionMatrix", fig, global_step=step, close=True)
    
    def log_roc_auc(
        self, 
        y_true: np.ndarray, 
        y_scores: np.ndarray, 
        step: int, 
        tag: str = "val", 
        class_names: Optional[List[str]] = None
    ) -> None:
        """
        Log ROC curve and AUC metrics.
        
        Args:
            y_true: True labels
            y_scores: Prediction scores (1D for binary, 2D for multiclass)
            step: Current training step/epoch
            tag: Tag prefix for the figure
            class_names: List of class names for multiclass
        """
        is_binary = y_scores.ndim == 1 or (y_scores.ndim == 2 and y_scores.shape[1] <= 2)
        
        try:
            if is_binary:
                fig, metrics = viz_helpers.create_binary_roc_figure(y_true, y_scores)
                self.writer.add_figure(f"{tag}/ROC_AUC", fig, global_step=step, close=True)
                
                # Log scalar metrics
                self.add_scalar(f"Metrics/AUC/{tag}", metrics["auc"], step)
                self.add_scalar(f"Metrics/Sensitivity/{tag}", metrics["sensitivity"], step)
                self.add_scalar(f"Metrics/Specificity/{tag}", metrics["specificity"], step)
            else:
                fig, macro_auc = viz_helpers.create_multiclass_roc_figure(y_true, y_scores, class_names)
                self.writer.add_figure(f"{tag}/ROC_AUC_OVR", fig, global_step=step, close=True)
                self.add_scalar(f"Metrics/AUC_macro/{tag}", macro_auc, step)
        except ValueError as e:
            print(f"Warning: Could not compute ROC curve - {e}")
    
    # ===================== Latent Space Visualization ====================
    
    def log_latent_umap(
        self, 
        latents: List[np.ndarray],
        labels: Optional[List] = None,
        step: int = 0,
        tag: str = "latent",
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean"
    ) -> None:
        """
        Log UMAP visualization of latent space.
        
        Args:
            latents: List of latent vectors [B, CH, D, H, W] from each batch
            labels: Optional list of labels for coloring points
            step: Current training step/epoch
            tag: Tag prefix for the figure
            n_neighbors: UMAP n_neighbors parameter
            min_dist: UMAP min_dist parameter
            metric: UMAP distance metric
        """
        try:
            fig = viz_helpers.create_umap_figure(latents, labels, n_neighbors, min_dist, metric)
            self.writer.add_figure(f"{tag}/UMAP", fig, global_step=step, close=True)
            print(f"✓ UMAP visualization logged to TensorBoard")
        except ImportError:
            print("UMAP not installed. Install with: pip install umap-learn")
        except Exception as e:
            print(f"Warning: UMAP visualization failed - {e}")

    # ======================== Logger Management =========================
    
    def flush(self) -> None:
        """Flush pending events to disk."""
        self.writer.flush()

    def close(self) -> None:
        """Close the TensorBoard writer."""
        self.writer.close()

    def get_log_dir(self) -> str:
        """Get the full path to the log directory."""
        return self.log_dir
