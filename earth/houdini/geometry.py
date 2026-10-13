"""Build Houdini geometry from decoded GLB chunks."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import hou
import numpy as np

from ..glb import Chunk

logger = logging.getLogger(__name__)

Progress = Callable[[float, str], None]


def build_chunk(
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
            uvs.append(primitive.uvs)
            faces.extend((primitive.faces + offset).tolist())
            textures.extend([primitive.texture or ''] * len(primitive.faces))
            offset += count
        if progress is not None:
            progress((index + 1) / total, f'Building mesh {index + 1}/{total}')

    if not points:
        logger.debug('No primitives to build')
        return

    uvs_2d = np.concatenate(uvs, axis=0)
    uvs_3d = np.insert(uvs_2d, 2, 0.0, axis=1)
    uv_values = uvs_3d.flatten().tolist()

    geo.createPoints(np.concatenate(points).tolist())
    geo.createPolygons(faces)

    geo.addAttrib(hou.attribType.Point, 'uv', (0.0, 0.0, 0.0))
    geo.setPointFloatAttribValues('uv', uv_values)

    geo.addAttrib(hou.attribType.Prim, 'texture', '')
    geo.setPrimStringAttribValues('texture', textures)

    logger.debug(
        'Built %d point(s) and %d polygon(s) from %d chunk(s)',
        offset,
        len(faces),
        len(chunks),
    )
