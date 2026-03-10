from __future__ import annotations

from pathlib import Path

import numpy as np


def load_numpy(path: str | Path) -> np.ndarray:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"NumPy volume not found: {file_path}")

    try:
        data = np.load(file_path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"Failed to read NumPy volume '{file_path.name}': {exc}") from exc

    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(
            f"NumPy volume '{file_path.name}' must be 3D after squeeze; got shape {data.shape}"
        )

    return data.astype(np.float32)
