from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
import time
import uuid

import numpy as np

from app.config import get_settings
from app.models.dataset import VolumeInfo
from app.services.discovery import resolve_dataset_path, resolve_series_source
from app.services.mask_loader import load_mask
from app.services.nifti_loader import load_nifti
from app.services.numpy_loader import load_numpy


@dataclass
class CachedSeries:
    key: str
    series_id: str
    volume: np.ndarray
    mask: np.ndarray | None
    spacing: tuple[float, float, float]
    labels: list[int]


@dataclass
class HandleRecord:
    cache_key: str
    updated_at: float


class VolumeCache:
    def __init__(
        self,
        max_items: int = 2,
        max_handles: int = 256,
        handle_ttl_seconds: int = 30 * 60,
    ):
        self.max_items = max_items
        self.max_handles = max_handles
        self.handle_ttl_seconds = handle_ttl_seconds
        self._series_cache: OrderedDict[str, CachedSeries] = OrderedDict()
        self._handles: OrderedDict[str, HandleRecord] = OrderedDict()
        self._lock = RLock()

    def load_series(self, dataset_id: str, patient_id: str, series_id: str) -> VolumeInfo:
        with self._lock:
            self._purge_expired_handles()
            cache_key = f"{dataset_id}:{patient_id}:{series_id}"
            cached = self._series_cache.pop(cache_key, None)
            if cached is not None:
                self._series_cache[cache_key] = cached
                load_handle = self._register_handle(cache_key)
                return VolumeInfo(
                    series_id=series_id,
                    load_handle=load_handle,
                    shape=list(cached.volume.shape),
                    spacing=[float(value) for value in cached.spacing],
                    has_mask=cached.mask is not None,
                    labels=list(cached.labels),
                )

            # Load from disk on first access for this dataset/patient/series tuple.
            dataset_path = resolve_dataset_path(get_settings().data_root, dataset_id)
            source = resolve_series_source(dataset_path, patient_id, series_id)

            try:
                if source.type == "nifti":
                    volume, spacing = load_nifti(source.image_path)
                    mask = load_mask(source.mask_path, is_nifti=True) if source.mask_path else None
                elif source.type == "voi":
                    volume = load_numpy(source.image_path)
                    spacing = (1.0, 1.0, 1.0)
                    mask = load_mask(source.mask_path, is_nifti=False) if source.mask_path else None
                else:
                    raise ValueError(f"Unsupported series type '{source.type}'")
            except (FileNotFoundError, ValueError) as exc:
                raise type(exc)(
                    f"{exc} (dataset={dataset_id}, patient={patient_id}, series={series_id})"
                ) from exc

            if mask is not None and volume.shape != mask.shape:
                raise ValueError(
                    f"Volume shape {volume.shape} does not match mask shape {mask.shape}"
                )

            labels = sorted(int(value) for value in np.unique(mask) if value > 0) if mask is not None else []
            cached = CachedSeries(
                key=cache_key,
                series_id=series_id,
                volume=volume,
                mask=mask,
                spacing=spacing,
                labels=labels,
            )
            self._series_cache[cache_key] = cached
            self._trim_series_cache()

            load_handle = self._register_handle(cache_key)
            return VolumeInfo(
                series_id=series_id,
                load_handle=load_handle,
                shape=list(volume.shape),
                spacing=[float(value) for value in spacing],
                has_mask=mask is not None,
                labels=labels,
            )

    def get_by_handle(self, load_handle: str) -> tuple[np.ndarray, np.ndarray | None, tuple[float, float, float]]:
        with self._lock:
            self._purge_expired_handles()
            if not load_handle:
                raise RuntimeError("Load handle is required")

            record = self._handles.get(load_handle)
            if record is None:
                raise RuntimeError("Load handle is invalid or expired")

            cached = self._series_cache.get(record.cache_key)
            if cached is None:
                self._handles.pop(load_handle, None)
                raise RuntimeError("Load handle is no longer available")

            # Touch both series entry and handle for LRU semantics.
            self._series_cache.pop(record.cache_key, None)
            self._series_cache[record.cache_key] = cached
            self._handles.pop(load_handle, None)
            self._handles[load_handle] = HandleRecord(
                cache_key=record.cache_key,
                updated_at=time.monotonic(),
            )
            return cached.volume, cached.mask, cached.spacing

    def _trim_series_cache(self) -> None:
        while len(self._series_cache) > self.max_items:
            evicted_key, _ = self._series_cache.popitem(last=False)
            self._drop_handles_for_cache_key(evicted_key)

    def _register_handle(self, cache_key: str) -> str:
        handle = uuid.uuid4().hex
        self._handles[handle] = HandleRecord(cache_key=cache_key, updated_at=time.monotonic())
        while len(self._handles) > self.max_handles:
            self._handles.popitem(last=False)
        return handle

    def _drop_handles_for_cache_key(self, cache_key: str) -> None:
        stale_handles = [
            handle
            for handle, record in self._handles.items()
            if record.cache_key == cache_key
        ]
        for handle in stale_handles:
            self._handles.pop(handle, None)

    def _purge_expired_handles(self) -> None:
        now = time.monotonic()
        while self._handles:
            handle, record = next(iter(self._handles.items()))
            if now - record.updated_at <= self.handle_ttl_seconds:
                break
            self._handles.pop(handle, None)


volume_cache = VolumeCache()
