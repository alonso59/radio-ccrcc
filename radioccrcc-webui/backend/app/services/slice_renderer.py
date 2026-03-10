from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image, ImageColor, ImageDraw
from skimage import measure


DEFAULT_LAYER_CONFIG: dict[int, dict[str, Any]] = {
    1: {"name": "Kidney", "color": "cyan", "alpha": 0.15, "linewidth": 2},
    2: {"name": "Tumor", "color": "yellow", "alpha": 0.20, "linewidth": 2},
    3: {"name": "Cyst", "color": "magenta", "alpha": 0.15, "linewidth": 2},
}

# Slices are upscaled so their longest side is at least this many pixels.
# This makes VOI crops (which can be ~64 px) fill the viewport like full
# NIfTI scans, giving a homogeneous look across series types.
_MIN_DISPLAY_PX = 512


def render_slice(
    volume: np.ndarray,
    mask: np.ndarray | None,
    axis: str,
    index: int,
    ww: float,
    wl: float,
    layers: list[int],
    layer_config: dict[int, dict[str, Any]] | None = None,
) -> bytes:
    config = layer_config or DEFAULT_LAYER_CONFIG
    image_slice = _extract_slice(volume, axis, index)
    windowed = _window_to_uint8(image_slice, ww, wl)
    rgb = np.repeat(windowed[..., None], 3, axis=2).astype(np.float32)

    mask_slice = _extract_slice(mask, axis, index) if mask is not None else None
    if mask_slice is not None:
        for label in layers:
            if label not in config:
                continue
            label_mask = mask_slice == label
            if not label_mask.any():
                continue

            color = np.asarray(ImageColor.getrgb(config[label]["color"]), dtype=np.float32)
            alpha = float(config[label]["alpha"])
            rgb[label_mask] = rgb[label_mask] * (1.0 - alpha) + color * alpha

    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")

    # ── Normalise output size ─────────────────────────────────────────────────
    # Upscale small images (e.g. VOI crops) so they fill the viewport just like
    # a full NIfTI scan would. Nearest-neighbour preserves crisp CT pixel edges.
    # We upscale BEFORE drawing contours so the lines are crisp at final size.
    w, h = image.size
    longest = max(w, h)
    if longest < _MIN_DISPLAY_PX:
        scale = _MIN_DISPLAY_PX / longest
        new_w = max(1, round(w * scale))
        new_h = max(1, round(h * scale))
        image = image.resize((new_w, new_h), Image.NEAREST)
        if mask_slice is not None:
            # Upscale mask with nearest-neighbour to keep label integrity
            mask_pil = Image.fromarray(mask_slice.astype(np.uint8), mode="L")
            mask_pil = mask_pil.resize((new_w, new_h), Image.NEAREST)
            mask_slice = np.asarray(mask_pil)

    if mask_slice is not None:
        image = _draw_contours(image, mask_slice, layers, config)

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _draw_contours(
    image: Image.Image,
    mask_slice: np.ndarray,
    layers: list[int],
    layer_config: dict[int, dict[str, Any]],
) -> Image.Image:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for label in layers:
        if label not in layer_config:
            continue
        label_mask = (mask_slice == label).astype(np.uint8)
        if label_mask.max() == 0:
            continue

        color = ImageColor.getrgb(layer_config[label]["color"])
        width = max(1, int(round(float(layer_config[label].get("linewidth", 1)))))
        contours = measure.find_contours(label_mask, level=0.5)
        for contour in contours:
            if len(contour) < 2:
                continue
            points = [(float(point[1]), float(point[0])) for point in contour]
            draw.line(points + [points[0]], fill=(*color, 255), width=width)

    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def _extract_slice(volume: np.ndarray | None, axis: str, index: int) -> np.ndarray:
    if volume is None:
        raise RuntimeError("No volume is currently loaded")

    normalized_axis = axis.strip().lower()
    if normalized_axis == "axial":
        if not 0 <= index < volume.shape[2]:
            raise IndexError(f"Axial index {index} out of bounds for depth {volume.shape[2]}")
        slice_2d = volume[:, :, index].T
    elif normalized_axis == "coronal":
        if not 0 <= index < volume.shape[1]:
            raise IndexError(f"Coronal index {index} out of bounds for height {volume.shape[1]}")
        slice_2d = volume[:, index, :].T
    elif normalized_axis == "sagittal":
        if not 0 <= index < volume.shape[0]:
            raise IndexError(f"Sagittal index {index} out of bounds for width {volume.shape[0]}")
        slice_2d = volume[index, :, :].T
    else:
        raise ValueError(f"Unsupported axis '{axis}'")

    return np.flipud(slice_2d)


def _window_to_uint8(slice_2d: np.ndarray, ww: float, wl: float) -> np.ndarray:
    if ww <= 0:
        raise ValueError("Window width must be greater than zero")

    hu_min = wl - (ww / 2.0)
    hu_max = wl + (ww / 2.0)
    clipped = np.clip(slice_2d.astype(np.float32), hu_min, hu_max)
    normalized = (clipped - hu_min) / (hu_max - hu_min)
    return np.round(normalized * 255.0).astype(np.uint8)
