"""Build Houdini geometry from decoded GLB chunks."""

from __future__ import annotations

from collections.abc import Callable

import hou
import numpy as np

from ..glb import Chunk

Progress = Callable[[float, str], None]


def copyright_string(chunks: list[tuple[Chunk, float]]) -> str:
    """Return the unique copyrights, most frequent first."""

    counts: dict[str, int] = {}
    for chunk, _ in chunks:
        for item in chunk.copyright.split(';'):
            if item:
                counts[item] = counts.get(item, 0) + 1
    ordered = sorted(counts, key=lambda key: counts[key], reverse=True)
    return ';'.join(ordered)


def build_geometry(
    geo: hou.Geometry,
    chunks: list[tuple[Chunk, float]],
    progress: Progress | None = None,
) -> None:
    """Create points, polygons and attributes for the decoded chunks."""

    points: list[np.ndarray] = []
    uvs: list[np.ndarray] = []
    faces: list[list[int]] = []
    prim_textures: list[str] = []
    prim_errors: list[float] = []
    offset = 0

    total = len(chunks)
    for index, (chunk, error) in enumerate(chunks):
        for primitive in chunk.primitives:
            count = primitive.points.shape[0]
            points.append(primitive.points)
            uvs.append(primitive.uvs.reshape(-1))
            faces.extend((primitive.faces + offset).tolist())
            prim_textures.extend([primitive.texture or ''] * len(primitive.faces))
            prim_errors.extend([error] * len(primitive.faces))
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
    geo.setPrimStringAttribValues('texture', prim_textures)

    geo.addAttrib(hou.attribType.Prim, 'error', 0.0)
    geo.setPrimFloatAttribValues(
        'error', np.array(prim_errors, dtype=np.float64).tolist()
    )
