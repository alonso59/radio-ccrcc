"""Helpers for resolving and comparing monitored training metrics."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple


def parse_monitor_name(monitor: str) -> Tuple[str, str]:
    """Return (split, metric_key) for monitor names like ``val_loss``."""
    monitor = str(monitor)
    if monitor.startswith("train_"):
        return "train", monitor[len("train_"):]
    if monitor.startswith("val_"):
        return "val", monitor[len("val_"):]
    return "val", monitor


def resolve_monitor_value(
    monitor: str,
    train_metrics: Optional[Mapping[str, Any]],
    val_metrics: Optional[Mapping[str, Any]],
) -> Optional[float]:
    """Resolve a configured monitor name to a scalar metric value."""
    split, metric_key = parse_monitor_name(monitor)
    metrics = train_metrics if split == "train" else val_metrics
    if metrics is None or metric_key not in metrics:
        return None

    value = metrics[metric_key]
    if hasattr(value, "item"):
        value = value.item()

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_improved(current: float, best: Optional[float], mode: str) -> bool:
    """Return whether a metric improved under ``min`` or ``max`` mode."""
    mode = str(mode).lower()
    if best is None:
        return True
    if mode == "min":
        return current < best
    if mode == "max":
        return current > best
    raise ValueError(f"Unsupported monitor mode: {mode!r}. Expected 'min' or 'max'.")
