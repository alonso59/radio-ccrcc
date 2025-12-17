import os
import torch

class ModelCheckpoint:
    """
    Callback to save the model when monitored metric improves.
    """
    def __init__(self, dirpath, monitor='val_loss', mode='min', save_best_only=True, filename='best_model.pth'):
        self.dirpath = dirpath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.filename = filename
        self.best = None
        os.makedirs(self.dirpath, exist_ok=True)

    def on_epoch_end(self, epoch, train_loss, val_loss):
        # Determine monitored value
        monitor_value = val_loss
        improved = False
        if self.best is None:
            improved = True
        elif self.mode == 'min' and monitor_value < self.best:
            improved = True
        elif self.mode == 'max' and monitor_value > self.best:
            improved = True
        if improved:
            self.best = monitor_value
            self._save_checkpoint(epoch)

    def _save_checkpoint(self, epoch):
        # Assumes model is accessible as self.model (set externally)
        if hasattr(self, 'model') and self.model is not None:
            path = os.path.join(self.dirpath, self.filename)
            
            # Handle DataParallel models - save module.state_dict()
            model_to_save = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'best': self.best
            }, path)
            print(f"[ModelCheckpoint] Saved best model at epoch {epoch} to {path}")

    def set_model(self, model):
        self.model = model
