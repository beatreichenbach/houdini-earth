"""Build Houdini geometry from decoded GLB chunks."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import hou
import numpy as np

from ..glb import Chunk

Progress = Callable[[float, str], None]


def build_geometry(
    geo: hou.Geometry,
    chunks: Sequence[Chunk],
    progress: Progress | None = None,
) -> None:
    """Create points, polygons and attributes for the decoded chunks."""

    points: list[np.ndarray] = []
    uvs: list[np.ndarray] = []
    faces: list[list[int]] = []
    textures: list[str] = []
    offset = 0

    total = len(chunks)
    for index, chunk in enumerate(chunks):
        for primitive in chunk.primitives:
            count = primitive.points.shape[0]
            points.append(primitive.points)
            uvs.append(primitive.uvs.reshape(-1))
            faces.extend((primitive.faces + offset).tolist())
            textures.extend([primitive.texture or ''] * len(primitive.faces))
            offset += count
        if progress is not None:
            progress((index + 1) / total, f'Building mesh {index + 1}/{total}')

    if not points:
        return

    geo.createPoints(np.concatenate(points).tolist())
    geo.createPolygons(faces)

    geo.addAttrib(hou.attribType.Point, 'uv', (0.0, 0.0))
    geo.setPointFloatAttribValues('uv', np.concatenate(uvs).tolist())

    geo.addAttrib(hou.attribType.Prim, 'texture', '')
    geo.setPrimStringAttribValues('texture', textures)
