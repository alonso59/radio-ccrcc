from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


def load_mask(path: str | Path, is_nifti: bool) -> np.ndarray:
    mask_path = Path(path)
    if is_nifti:
        image = nib.load(str(mask_path))
        image = nib.as_closest_canonical(image)
        data = image.get_fdata()
    else:
        data = np.load(mask_path)

    if data.ndim == 4:
        data = data[..., 0]

    return data.astype(np.uint8)
