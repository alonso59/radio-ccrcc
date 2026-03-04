from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


def load_nifti(path: str | Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    image = nib.load(str(Path(path)))
    image = nib.as_closest_canonical(image)
    data = image.get_fdata()
    if data.ndim == 4:
        data = data[..., 0]

    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    return data.astype(np.float32), spacing
