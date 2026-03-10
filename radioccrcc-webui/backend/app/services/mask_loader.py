from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


def load_mask(path: str | Path, is_nifti: bool) -> np.ndarray:
    mask_path = Path(path)
    if not mask_path.is_file():
        raise FileNotFoundError(f"Segmentation mask not found: {mask_path}")

    if is_nifti:
        try:
            image = nib.load(str(mask_path))
            image = nib.as_closest_canonical(image)
            data = image.get_fdata()
        except Exception as exc:
            raise ValueError(
                f"Failed to read NIfTI segmentation '{mask_path.name}': {exc}"
            ) from exc
    else:
        try:
            data = np.load(mask_path, allow_pickle=False)
        except Exception as exc:
            raise ValueError(
                f"Failed to read NumPy segmentation '{mask_path.name}': {exc}"
            ) from exc

    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(
            f"Segmentation mask '{mask_path.name}' must be 3D after squeeze; got shape {data.shape}"
        )

    return data.astype(np.uint8)
