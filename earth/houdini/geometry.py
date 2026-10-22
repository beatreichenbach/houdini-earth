"""Build Houdini geometry from decoded GLB chunks."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import hou
import numpy as np

from ..glb import Chunk

logger = logging.getLogger(__name__)

Progress = Callable[[float, str], None]

NAME_ATTR = 'name'
TEXTURE_ATTR = 'texture'
UV_ATTR = 'uv'


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
    names: list[str] = []
    offset = 0

    # Collect attributes
    total = len(chunks)
    for index, chunk in enumerate(chunks):
        for primitive in chunk.primitives:
            points.append(primitive.points)
            uvs.append(primitive.uvs[primitive.faces])
            faces.extend((primitive.faces + offset).tolist())
            textures.extend([primitive.texture] * len(primitive.faces))
            names.extend([chunk.name] * len(primitive.faces))

            count = primitive.points.shape[0]
            offset += count

        if progress is not None:
            progress((index + 1) / total, f'Building mesh {index + 1}/{total}')

    if not points:
        logger.debug('No primitives to build')
        return

    # Create geometry
    uvs_2d = np.concatenate(uvs, axis=0).reshape(-1, 2)
    uvs_3d = np.column_stack((uvs_2d, np.zeros(len(uvs_2d))))
    uv_values = uvs_3d.flatten().tolist()

    geo.createPoints(np.concatenate(points).tolist())
    geo.createPolygons(faces)

    geo.addAttrib(hou.attribType.Vertex, UV_ATTR, (0.0, 0.0, 0.0))
    geo.setVertexFloatAttribValues('uv', uv_values)

    geo.addAttrib(hou.attribType.Prim, TEXTURE_ATTR, '')
    geo.setPrimStringAttribValues(TEXTURE_ATTR, textures)

    if any(names):
        geo.addAttrib(hou.attribType.Prim, NAME_ATTR, '')
        geo.setPrimStringAttribValues(NAME_ATTR, names)

    logger.debug(
        'Built %d point(s) and %d polygon(s) from %d chunk(s)',
        offset,
        len(faces),
        len(chunks),
    )
