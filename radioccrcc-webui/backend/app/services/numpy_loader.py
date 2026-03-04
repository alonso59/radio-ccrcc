from __future__ import annotations

from pathlib import Path

import numpy as np


def load_numpy(path: str | Path) -> np.ndarray:
    data = np.load(Path(path))
    if data.ndim == 4:
        data = data[..., 0]
    return data.astype(np.float32)
