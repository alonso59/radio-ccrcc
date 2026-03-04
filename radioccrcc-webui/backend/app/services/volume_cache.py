from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock

import numpy as np

from app.config import get_settings
from app.models.dataset import SeriesInfo, VolumeInfo
from app.services.discovery import discover_series, resolve_dataset_path
from app.services.mask_loader import load_mask
from app.services.nifti_loader import load_nifti
from app.services.numpy_loader import load_numpy


@dataclass
class CachedVolume:
    key: str
    info: VolumeInfo
    volume: np.ndarray
    mask: np.ndarray | None
    spacing: tuple[float, float, float]


class VolumeCache:
    def __init__(self, max_items: int = 2):
        self.max_items = max_items
        self._cache: OrderedDict[str, CachedVolume] = OrderedDict()
        self._current_key: str | None = None
        self._lock = RLock()

    def load_series(self, dataset_id: str, patient_id: str, series_id: str) -> VolumeInfo:
        cache_key = f"{dataset_id}:{patient_id}:{series_id}"
        with self._lock:
            if cache_key in self._cache:
                cached = self._cache.pop(cache_key)
                self._cache[cache_key] = cached
                self._current_key = cache_key
                return cached.info

            dataset_path = resolve_dataset_path(get_settings().data_root, dataset_id)
            series = self._resolve_series(dataset_path, patient_id, series_id)

            if series.type == "nifti":
                volume, spacing = load_nifti(series.image_path)
                mask = load_mask(series.mask_path, is_nifti=True) if series.mask_path else None
            elif series.type == "voi":
                volume = load_numpy(series.image_path)
                spacing = (1.0, 1.0, 1.0)
                mask = load_mask(series.mask_path, is_nifti=False) if series.mask_path else None
            else:
                raise ValueError(f"Unsupported series type '{series.type}'")

            if mask is not None and volume.shape != mask.shape:
                raise ValueError(
                    f"Volume shape {volume.shape} does not match mask shape {mask.shape}"
                )

            labels = sorted(int(value) for value in np.unique(mask) if value > 0) if mask is not None else []
            info = VolumeInfo(
                series_id=series_id,
                shape=list(volume.shape),
                spacing=[float(value) for value in spacing],
                has_mask=mask is not None,
                labels=labels,
            )
            cached = CachedVolume(
                key=cache_key,
                info=info,
                volume=volume,
                mask=mask,
                spacing=spacing,
            )
            self._cache[cache_key] = cached
            self._current_key = cache_key

            while len(self._cache) > self.max_items:
                self._cache.popitem(last=False)

            return info

    def get_current(self) -> tuple[np.ndarray, np.ndarray | None, tuple[float, float, float]]:
        with self._lock:
            if self._current_key is None or self._current_key not in self._cache:
                raise RuntimeError("No series is currently loaded")
            cached = self._cache[self._current_key]
            return cached.volume, cached.mask, cached.spacing

    @staticmethod
    def _resolve_series(dataset_path, patient_id: str, series_id: str) -> SeriesInfo:
        for series in discover_series(dataset_path, patient_id):
            if series.series_id == series_id:
                return series
        raise FileNotFoundError(
            f"Series '{series_id}' for patient '{patient_id}' was not found in {dataset_path.name}"
        )


volume_cache = VolumeCache()
