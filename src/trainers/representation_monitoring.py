"""Monitoring helpers for latent collection, figures, and probe evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..dataloader.dataloader import map_label_to_category


@dataclass(frozen=True)
class MonitoringSettings:
    """Configuration for optional representation monitoring."""

    collect_latents: bool
    umap_log_interval: int
    image_log_interval: int
    probe_interval: int
    probe_max_batches: int
    probe_epochs: int
    probe_lr: float
    probe_l2: float


class RepresentationMonitor:
    """Handle optional image logging, UMAP, and linear-probe evaluation."""

    def __init__(
        self,
        settings: MonitoringSettings,
        logger: Any,
        model: torch.nn.Module,
        dataloaders: Dict[str, Any],
        device: torch.device,
    ):
        self.settings = settings
        self.logger = logger
        self.model = model
        self.dataloaders = dataloaders
        self.device = device
        self.samples: Dict[str, Dict[str, torch.Tensor]] = {"train": {}, "val": {}}
        self.latents: Dict[str, list[np.ndarray]] = {"train": [], "val": []}
        self.labels: Dict[str, list[np.ndarray]] = {"train": [], "val": []}
        self.collect_this_epoch = False
        self.log_images_this_epoch = False

    def on_epoch_start(self, epoch: int, is_train: bool) -> None:
        """Refresh collection flags and clear per-split buffers."""
        self.collect_this_epoch = self.settings.collect_latents and epoch % self.settings.umap_log_interval == 0
        self.log_images_this_epoch = epoch % self.settings.image_log_interval == 0
        if self.collect_this_epoch:
            split = "train" if is_train else "val"
            self.latents[split] = []
            self.labels[split] = []

    def record_batch(
        self,
        split: str,
        batch: Any,
        inputs: torch.Tensor,
        reconstruction: torch.Tensor,
        mask: torch.Tensor,
        z_mu: torch.Tensor,
    ) -> None:
        """Capture optional batch artifacts for later logging."""
        if self.log_images_this_epoch:
            self._store_sample(split, inputs, reconstruction, mask)
        if self.collect_this_epoch:
            self._collect_latent_data(batch, z_mu, split)

    def on_epoch_end(self, epoch: int) -> None:
        """Emit collected figures and probe metrics."""
        if self.log_images_this_epoch:
            for split in ("val", "train"):
                sample = self.samples.get(split)
                if sample:
                    self.logger.log_axial_figure(
                        sample["input"],
                        sample["recon"],
                        step=epoch,
                        tag=split,
                        mask=sample["mask"],
                    )

        if self.collect_this_epoch:
            self._log_latent_umap(epoch)

        if self.settings.probe_interval and epoch % self.settings.probe_interval == 0:
            train_embeddings, train_labels = self._extract_embeddings(self.dataloaders["train"])
            val_embeddings, val_labels = self._extract_embeddings(self.dataloaders["val"])
            for key, value in self._linear_probe_classification(
                train_embeddings,
                train_labels,
                val_embeddings,
                val_labels,
            ).items():
                self.logger.add_scalar(f"probe_{key}", value, epoch)

    def _store_sample(
        self,
        split: str,
        inputs: torch.Tensor,
        reconstruction: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        idx = int(torch.randint(0, inputs.shape[0], (1,)).item())
        self.samples[split] = {
            "input": inputs[idx].detach().cpu(),
            "recon": reconstruction[idx].detach().cpu(),
            "mask": mask[idx].detach().cpu(),
        }

    def _collect_latent_data(self, batch: Any, z_mu: torch.Tensor, split: str) -> None:
        self.latents[split].append(z_mu.detach().cpu().numpy())
        self.labels[split].append(np.array([map_label_to_category(label) for label in batch["label"]]))

    def _log_latent_umap(self, epoch: int) -> None:
        for split in ("train", "val"):
            if not self.latents[split]:
                continue
            labels = self.labels[split] or None
            self.logger.log_latent_umap(self.latents[split], labels=labels, step=epoch, tag=split)
            self.latents[split] = []
            self.labels[split] = []

    @torch.no_grad()
    def _extract_embeddings(self, loader: Any) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        embeddings, labels = [], []

        for index, batch in enumerate(loader):
            if index >= self.settings.probe_max_batches:
                break
            inputs = batch["ct"]["data"].to(self.device)
            targets = [map_label_to_category(label) for label in batch["label"]]
            _, z_mu, _ = self.model(inputs)
            embeddings.append(F.normalize(z_mu.mean(dim=(2, 3, 4)), dim=1).cpu())
            labels.append(torch.as_tensor(targets))

        return torch.cat(embeddings, dim=0).float(), torch.cat(labels, dim=0)

    def _linear_probe_classification(
        self,
        train_embeddings: torch.Tensor,
        train_labels: torch.Tensor,
        val_embeddings: torch.Tensor,
        val_labels: torch.Tensor,
    ) -> Dict[str, float]:
        is_binary = len(torch.unique(train_labels)) == 2
        in_dim = train_embeddings.shape[1]
        train_embeddings = train_embeddings.to(self.device)
        val_embeddings = val_embeddings.to(self.device)

        if is_binary:
            head = torch.nn.Linear(in_dim, 1).to(self.device)
            loss_fn = torch.nn.BCEWithLogitsLoss()
            target_tensor = train_labels.float().to(self.device).view(-1, 1)
        else:
            n_classes = int(torch.max(train_labels).item() + 1)
            head = torch.nn.Linear(in_dim, n_classes).to(self.device)
            loss_fn = torch.nn.CrossEntropyLoss()
            target_tensor = train_labels.long().to(self.device)

        optimizer = torch.optim.Adam(
            head.parameters(),
            lr=self.settings.probe_lr,
            weight_decay=self.settings.probe_l2,
        )
        head.train()
        for _ in range(self.settings.probe_epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(head(train_embeddings), target_tensor)
            loss.backward()
            optimizer.step()

        head.eval()
        with torch.no_grad():
            logits = head(val_embeddings)
            if is_binary:
                predictions = (torch.sigmoid(logits).squeeze(1) > 0.5).long().cpu()
            else:
                predictions = torch.argmax(logits, dim=1).cpu()

        accuracy = (predictions == val_labels.long()).float().mean().item()
        return {"probe_acc": accuracy}
