"""Decode GLB tiles into points, faces, UVs and textures."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import trimesh
from PIL import Image

_GLB_ERRORS = (
    OSError,  # tile unreadable/truncated, or texture write failed
    ValueError,  # bad GLB header, JSON (JSONDecodeError) or buffer shapes
    KeyError,  # malformed glTF structure
    IndexError,  # out-of-range buffer/accessor indices
)


class GlbError(RuntimeError):
    """A GLB tile could not be decoded."""


@dataclass
class Primitive:
    """A decoded mesh primitive with its texture."""

    points: np.ndarray
    faces: np.ndarray
    uvs: np.ndarray
    texture: str | None = None


@dataclass
class Chunk:
    """A decoded GLB tile."""

    primitives: tuple[Primitive, ...] = ()


def load_chunk(glb_path: str) -> Chunk:
    """
    Return the decoded chunk for a GLB file.

    :raises GlbError: if the tile cannot be read or decoded.
    """

    try:
        scene = trimesh.load_scene(glb_path)

        primitives = []
        for index, geometry in enumerate(scene.dump()):
            if not isinstance(geometry, trimesh.Trimesh):
                continue
            points = np.asarray(geometry.vertices, dtype=np.float64)
            uvs = getattr(geometry.visual, 'uv', None)
            if uvs is None:
                uvs = np.zeros((points.shape[0], 2), dtype=np.float64)

            faces = np.asarray(geometry.faces, dtype=np.int64)
            uvs = np.asarray(uvs, dtype=np.float64)
            texture = _write_texture(geometry, glb_path, index)

            primitive = Primitive(points=points, faces=faces, uvs=uvs, texture=texture)
            primitives.append(primitive)

        return Chunk(primitives=tuple(primitives))

    except _GLB_ERRORS as error:
        raise GlbError(f'Could not decode {glb_path}: {error}') from error


def _write_texture(mesh: trimesh.Trimesh, glb_path: str, index: int) -> str | None:
    """Write the mesh's base color texture next to the GLB and return its path."""

    material = getattr(mesh.visual, 'material', None)
    image = getattr(material, 'baseColorTexture', None)
    if not isinstance(image, Image.Image):
        return None
    path = f'{os.path.splitext(glb_path)[0]}_{index}.jpg'
    if not os.path.isfile(path):
        image.save(path, quality=95)
    return path
