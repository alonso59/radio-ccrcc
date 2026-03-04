from __future__ import annotations

import numpy as np
import trimesh
from skimage.measure import marching_cubes
from trimesh.smoothing import filter_laplacian


def generate_mesh(
    mask: np.ndarray | None,
    label: int,
    spacing: tuple[float, float, float],
    smooth: bool = True,
) -> bytes | None:
    if mask is None:
        return None
    if label <= 0:
        raise ValueError("Label must be a positive integer")

    binary = np.asarray(mask == label, dtype=np.uint8)
    if not np.any(binary):
        return None

    # Padding closes surfaces that would otherwise be clipped at the volume edge.
    padded = np.pad(binary, pad_width=1, mode="constant", constant_values=0)
    vertices, faces, normals, _ = marching_cubes(
        padded,
        level=0.5,
        spacing=spacing,
        allow_degenerate=False,
    )
    offset = np.asarray(spacing, dtype=np.float32)
    vertices = vertices - offset

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=normals,
        process=False,
    )
    if smooth:
        filter_laplacian(mesh, lamb=0.35, iterations=5)

    return mesh.export(file_type="glb")
