"""Epoch-level scalar history storage and import/export helpers."""

from __future__ import annotations

import csv
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

logger = logging.getLogger(__name__)


def as_float(value: Any) -> Optional[float]:
    """Convert numeric-like values to float when possible."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


class EpochScalarRecorder:
    """Store epoch-level scalar history for export, resume, and plotting."""

    def __init__(self) -> None:
        self._series: Dict[str, Dict[int, float]] = {}

    def add_scalar(self, tag: str, value: Any, step: int) -> None:
        """Record an epoch-level scalar value."""
        if tag.startswith("step_"):
            return

        scalar = as_float(value)
        if scalar is None:
            return

        epoch = int(step)
        self._series.setdefault(tag, {})[epoch] = scalar

    def has_data(self) -> bool:
        """Return whether any scalar data has been recorded."""
        return any(points for points in self._series.values())

    def is_empty(self) -> bool:
        """Return whether the recorder has no series."""
        return not self.has_data()

    def reset(self) -> None:
        """Clear all recorded history."""
        self._series.clear()

    def merge_series(self, series: Dict[str, Dict[int, float]]) -> None:
        """Merge externally prepared series into the recorder."""
        for tag, points in series.items():
            target = self._series.setdefault(tag, {})
            for epoch, value in points.items():
                scalar = as_float(value)
                if scalar is None:
                    continue
                target[int(epoch)] = scalar

    def as_series_dict(self) -> Dict[str, Dict[int, float]]:
        """Return a deep-copied epoch-to-value mapping for each tag."""
        return {
            tag: {int(epoch): float(value) for epoch, value in points.items()}
            for tag, points in self._series.items()
        }

    def to_payload(self, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build a JSON-serializable payload."""
        payload = {
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {},
            "series": {},
        }
        for tag in sorted(self._series):
            payload["series"][tag] = [
                {"epoch": int(epoch), "value": float(self._series[tag][epoch])}
                for epoch in sorted(self._series[tag])
            ]
        return payload

    def load_payload(self, payload: Dict[str, Any]) -> None:
        """Merge a previously exported payload."""
        loaded: Dict[str, Dict[int, float]] = {}
        for tag, points in payload.get("series", {}).items():
            loaded_points: Dict[int, float] = {}
            for point in points:
                if not isinstance(point, dict):
                    continue
                epoch = point.get("epoch")
                value = as_float(point.get("value"))
                if epoch is None or value is None:
                    continue
                loaded_points[int(epoch)] = value
            if loaded_points:
                loaded[tag] = loaded_points
        self.merge_series(loaded)

    def load_json(self, path: Path) -> bool:
        """Load history from a JSON file if present."""
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to read history JSON from %s: %s", path, exc)
            return False
        self.load_payload(payload)
        return True

    def save_json(self, path: Path, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Write history to a JSON file."""
        path.write_text(
            json.dumps(self.to_payload(metadata=metadata), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def save_csv(self, path: Path) -> None:
        """Write history to a wide CSV table keyed by epoch."""
        all_epochs = sorted({epoch for points in self._series.values() for epoch in points})
        tags = sorted(self._series)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch", *tags])
            writer.writeheader()
            for epoch in all_epochs:
                row: Dict[str, Any] = {"epoch": epoch}
                for tag in tags:
                    row[tag] = self._series[tag].get(epoch, "")
                writer.writerow(row)

    def import_tensorboard(self, tb_root: Path) -> bool:
        """Seed epoch history from TensorBoard scalar event files."""
        event_parents = sorted({path.parent for path in tb_root.rglob("events.out.tfevents.*")})
        if not event_parents:
            return False

        loaded_any = False
        for directory in event_parents:
            try:
                accumulator = EventAccumulator(str(directory))
                accumulator.Reload()
            except Exception as exc:
                logger.warning("Failed to load TensorBoard events from %s: %s", directory, exc)
                continue

            for tag in accumulator.Tags().get("scalars", []):
                if tag.startswith("step_"):
                    continue
                for event in accumulator.Scalars(tag):
                    value = as_float(event.value)
                    if value is None or math.isnan(value):
                        continue
                    self.add_scalar(tag, value, event.step)
                    loaded_any = True
        return loaded_any
