"""
Classifier Trainer using frozen pretrained autoencoder.
Trains a classifier on latent representations from a frozen VAE encoder.
"""
from typing import Dict, Any, Optional, Tuple, Sequence, List
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig

from .base_trainer import BaseTrainer
from utils.scheduler import get_scheduler


class ClassifierTrainer(BaseTrainer):
    """
    Trainer for classification using frozen VAE latent representations.
    Freezes the autoencoder and trains only the classifier head.
    """
    
    def __init__(
        self,
        cfg: DictConfig,
        model_auto: torch.nn.Module,
        model_class: torch.nn.Module,
        dataloaders: Dict[str, Any],
        device: torch.device,
        logger: Any,
        callbacks: Optional[list] = None,
        max_epochs: Optional[int] = None,
        class_names: Optional[Sequence[str]] = None,
        allow_resize: bool = False,
    ):
        super().__init__(cfg, dataloaders, device, logger, callbacks, max_epochs)
        self.model_auto = model_auto
        self.model_class = model_class
        self.allow_resize = allow_resize
        self.image_key = getattr(cfg.dataset, "image_key", "ct")
        
        # Class configuration
        self.num_classes = int(getattr(cfg.model, "num_classes", 2))
        self.class_names = list(class_names) if class_names else [f"Class_{i}" for i in range(self.num_classes)]
        
        if len(self.class_names) != self.num_classes:
            raise ValueError(f"Number of class names ({len(self.class_names)}) != num_classes ({self.num_classes})")
        
        # Expected latent shape
        self.expected_channels = 3
        self.expected_spatial = (8, 8, 8)
        
        # Initialize components
        self.setup_models()
        self.setup_optimizers()
        self.setup_schedulers()
        self.setup_criteria()
    
    def setup_models(self) -> None:
        """Setup and freeze autoencoder, setup classifier."""
        # Move to device
        self.model_auto = self.model_auto.to(self.device)
        self.model_class = self.model_class.to(self.device)
        
        # Freeze autoencoder
        for p in self.model_auto.parameters():
            p.requires_grad = False
        self.model_auto.eval()
    
    def setup_optimizers(self) -> None:
        """Initialize optimizer for classifier only."""
        OptimCls = getattr(optim, self.cfg.optimizer.name)
        self.optimizer = OptimCls(
            self.model_class.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=getattr(self.cfg.optimizer, "weight_decay", 0.0)
        )
    
    def setup_schedulers(self) -> None:
        """Initialize learning rate scheduler."""
        if hasattr(self.cfg, 'scheduler'):
            self.scheduler = get_scheduler(self.optimizer, self.cfg.scheduler)
        else:
            self.scheduler = None
    
    def setup_criteria(self) -> None:
        """Initialize loss function with optional class weights."""
        class_weights = getattr(self.cfg, "class_weights", None)
        if class_weights is not None:
            class_weights = torch.as_tensor(class_weights, dtype=torch.float32, device=self.device)
        self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    def set_train_mode(self, train: bool) -> None:
        """Set classifier to train/eval mode. Autoencoder stays in eval."""
        self.model_class.train(train)
        self.model_auto.eval()  # Always in eval mode
    
    def train_step(self, batch: Any) -> Dict[str, float]:
        """Execute one training step."""
        x, y = self._get_inputs_and_labels(batch)

        # Extract latent representation (no gradients)
        with torch.no_grad():
            mu_vol = self._encode_mu_volume(x)
        
        # Forward pass through classifier
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model_class(mu_vol)
        
        if logits.ndim != 2:
            raise RuntimeError(f"Classifier must output [B,C]; got {tuple(logits.shape)}")
        
        # Compute loss
        loss = self.criterion(logits, y)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)
            accuracy = (pred == y).float().mean().item()
        
        return {
            "loss": loss.item(),
            "accuracy": accuracy,
            "predictions": pred.cpu().numpy(),
            "labels": y.cpu().numpy(),
            "probabilities": probs.cpu().numpy()
        }
    
    def validation_step(self, batch: Any) -> Dict[str, Any]:
        """Execute one validation step."""
        x, y = self._get_inputs_and_labels(batch)
        
        with torch.no_grad():
            # Extract latent representation
            mu_vol = self._encode_mu_volume(x)
            
            # Forward pass
            logits = self.model_class(mu_vol)
            
            if logits.ndim != 2:
                raise RuntimeError(f"Classifier must output [B,C]; got {tuple(logits.shape)}")
            
            # Compute loss
            loss = self.criterion(logits, y)
            
            # Compute accuracy
            probs = F.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)
            accuracy = (pred == y).float().mean().item()
        
        return {
            "loss": loss.item(),
            "accuracy": accuracy,
            "predictions": pred.cpu().numpy(),
            "labels": y.cpu().numpy(),
            "probabilities": probs.cpu().numpy()
        }
    
    def _initialize_epoch_metrics(self) -> Dict[str, Any]:
        """Initialize metrics dictionary for epoch."""
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "predictions": [],
            "labels": [],
            "probabilities": []
        }
    
    def _average_epoch_metrics(
        self, 
        epoch_metrics: Dict[str, Any], 
        n_batches: int
    ) -> Dict[str, Any]:
        """Average accumulated metrics, handle arrays separately."""
        averaged = {}
        for k, v in epoch_metrics.items():
            if k in ["predictions", "labels", "probabilities"]:
                # Concatenate arrays (only present in validation metrics)
                if isinstance(v, list) and len(v) > 0:
                    averaged[k] = np.concatenate(v, axis=0)
                else:
                    # Empty or no data
                    averaged[k] = np.array([])
            else:
                # Average scalars
                averaged[k] = v / max(1, n_batches)
        return averaged
    
    def on_epoch_end(
        self, 
        epoch: int, 
        train_metrics: Dict[str, Any], 
        val_metrics: Dict[str, Any]
    ) -> None:
        """Compute additional metrics and visualizations."""
        # Compute precision, recall, F1 from validation predictions
        if "predictions" in val_metrics and "labels" in val_metrics:
            y_pred = val_metrics["predictions"]
            y_true = val_metrics["labels"]
            y_probs = val_metrics.get("probabilities", None)
            
            if y_true.size > 0:
                # Compute metrics
                prec, rec, f1 = self._precision_recall_f1(y_pred, y_true)
                
                # Log additional metrics directly (don't modify val_metrics after logging)
                self.logger.add_scalar("Precision_macro/val", prec, epoch)
                self.logger.add_scalar("Recall_macro/val", rec, epoch)
                self.logger.add_scalar("F1_macro/val", f1, epoch)
                
                # Log confusion matrix
                try:
                    self.logger.log_confusion_matrix(
                        y_true, y_pred, self.class_names, step=epoch, tag="val"
                    )
                except Exception as e:
                    print(f"[WARN] CM logging failed: {e}")
                
                # Log ROC curve
                try:
                    self.logger.log_roc_auc(y_true, y_probs, step=epoch, tag="val", class_names=self.class_names)
                except Exception as e:
                    print(f"[WARN] ROC logging failed: {e}")

        # Call parent for callbacks
        super().on_epoch_end(epoch, train_metrics, val_metrics)
    
    def _log_epoch_metrics(
        self, 
        epoch: int, 
        train_metrics: Dict[str, Any], 
        val_metrics: Dict[str, Any]
    ) -> None:
        """Override to filter out array metrics before logging."""
        # Filter out array metrics (predictions, labels, probabilities)
        scalar_train = {k: v for k, v in train_metrics.items() 
                       if k not in ["predictions", "labels", "probabilities"]}
        scalar_val = {k: v for k, v in val_metrics.items() 
                     if k not in ["predictions", "labels", "probabilities"]}
        
        # Debug: print what we're logging
        print(f"[DEBUG] Logging epoch {epoch} - Train keys: {list(scalar_train.keys())}, Val keys: {list(scalar_val.keys())}")
        
        # Call parent to log only scalar metrics
        super()._log_epoch_metrics(epoch, scalar_train, scalar_val)
    
    def _print_epoch_summary(
        self, 
        epoch: int, 
        train_metrics: Dict[str, Any], 
        val_metrics: Dict[str, Any], 
        epoch_time: float
    ) -> None:
        """Override to filter out array metrics before printing."""
        # Filter out array metrics (predictions, labels, probabilities)
        scalar_train = {k: v for k, v in train_metrics.items() 
                       if k not in ["predictions", "labels", "probabilities"]}
        scalar_val = {k: v for k, v in val_metrics.items() 
                     if k not in ["predictions", "labels", "probabilities"]}
        
        # Call parent to print only scalar metrics
        super()._print_epoch_summary(epoch, scalar_train, scalar_val, epoch_time)
    
    def step_schedulers(self) -> None:
        """Step the learning rate scheduler."""
        if self.scheduler is not None:
            self.scheduler.step()
    
    @torch.no_grad()
    def _encode_mu_volume(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract latent mean volume from autoencoder.
        Returns μ(x) as a volume with shape [B, C, D, H, W].
        """
        _, mu, _ = self.model_auto(x)
        
        # Ensure 5D (B,C,D,H,W)
        if mu.ndim == 5:
            pass
        elif mu.ndim == 4:
            mu = mu.unsqueeze(2)
        else:
            raise RuntimeError(f"Unexpected mu shape: {tuple(mu.shape)}")

        return mu
    
    def _get_inputs_and_labels(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract inputs and labels from batch."""
        # Get images
        x = batch["ct"]["data"]
        x = x.to(self.device)
        # Get labels
        y = batch["label"]
        
        # Convert class names to indices if needed
        if isinstance(y, (list, tuple)) and isinstance(y[0], str):
            y = [self.class_names.index(c) for c in y]
        
        y = torch.tensor(y, dtype=torch.long, device=self.device) if not isinstance(y, torch.Tensor) else y.to(self.device)
        
        return x, y
    
    def _precision_recall_f1(
        self, 
        y_pred: np.ndarray, 
        y_true: np.ndarray
    ) -> Tuple[float, float, float]:
        """Compute macro-averaged precision, recall, and F1 score."""
        precs, recs, f1s = [], [], []
        for c in range(self.num_classes):
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))
            
            prec = tp / (tp + fp + 1e-12)
            rec = tp / (tp + fn + 1e-12)
            f1 = 2 * prec * rec / (prec + rec + 1e-12)
            
            precs.append(prec)
            recs.append(rec)
            f1s.append(f1)
        
        return float(np.mean(precs)), float(np.mean(recs)), float(np.mean(f1s))
    
    def _run_epoch(
        self, 
        dataloader: Any, 
        epoch: int, 
        train: bool = True
    ) -> Dict[str, Any]:
        """
        Override to handle array accumulation for predictions.
        """
        from tqdm import tqdm
        
        mode = "train" if train else "val"
        self.set_train_mode(train)
        
        # Initialize metrics using the method
        epoch_metrics = self._initialize_epoch_metrics()
        n_batches = len(dataloader)
        
        with torch.set_grad_enabled(train):
            for batch in tqdm(
                dataloader, 
                desc=f"{mode.capitalize()} Epoch {epoch}", 
                ncols=100, 
                ascii=True
            ):
                # Execute step
                if train:
                    step_metrics = self.train_step(batch)
                else:
                    step_metrics = self.validation_step(batch)
                
                # Accumulate metrics
                for key, value in step_metrics.items():
                    if key in ["predictions", "labels", "probabilities"]:
                        # Accumulate arrays for both train and val
                        epoch_metrics[key].append(value)
                    elif key in epoch_metrics:
                        epoch_metrics[key] += value
                    else:
                        epoch_metrics[key] = value
                
                # Log step metrics (scalars only)
                step_scalars = {k: v for k, v in step_metrics.items() 
                               if k not in ["predictions", "labels", "probabilities"]}
                self._log_step_metrics(mode, step_scalars)
                self.global_step += 1
        
        # Average metrics
        epoch_metrics = self._average_epoch_metrics(epoch_metrics, n_batches)
        
        return epoch_metrics
