from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


def load_nifti(path: str | Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"NIfTI volume not found: {file_path}")

    try:
        image = nib.load(str(file_path))
        image = nib.as_closest_canonical(image)
        data = image.get_fdata()
    except Exception as exc:
        raise ValueError(f"Failed to read NIfTI volume '{file_path.name}': {exc}") from exc

    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(
            f"NIfTI volume '{file_path.name}' must be 3D after canonicalization; got shape {data.shape}"
        )

    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    return data.astype(np.float32), spacing
