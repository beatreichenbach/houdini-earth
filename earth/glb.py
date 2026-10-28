"""Decode GLB tiles into points, faces, UVs and textures."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import trimesh
from PIL import Image

_GLB_ERRORS = (
    OSError,  # tile unreadable/truncated
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
    texture: str


@dataclass
class Chunk:
    """A decoded GLB tile."""

    name: str = ''
    primitives: tuple[Primitive, ...] = ()
    attribute_ids: dict[str, str] = field(default_factory=dict)


def load_chunk(glb_path: str, extract_textures: bool = True) -> Chunk:
    """
    Return the decoded chunk for a GLB file.

    :raises GlbError: if the tile cannot be read or decoded.
    """

    try:
        scene = trimesh.load_scene(glb_path)

        name = os.path.splitext(os.path.basename(glb_path))[0]
        primitives = []

        for index, geometry in enumerate(scene.dump()):
            if not isinstance(geometry, trimesh.Trimesh):
                continue

            points = np.asarray(geometry.vertices, dtype=np.float64)
            faces = np.asarray(geometry.faces, dtype=np.int64)
            uvs = getattr(geometry.visual, 'uv', None)
            if uvs is None:
                uvs = np.zeros((points.shape[0], 2), dtype=np.float64)
            uvs = np.asarray(uvs, dtype=np.float64)
            texture = ''
            if extract_textures:
                texture_path = f'{os.path.splitext(glb_path)[0]}_{index}.jpg'
                texture = _extract_texture(geometry, texture_path)

            primitive = Primitive(points=points, faces=faces, uvs=uvs, texture=texture)
            primitives.append(primitive)

        chunk = Chunk(name, tuple(primitives))
        return chunk

    except _GLB_ERRORS as error:
        raise GlbError(f'Could not decode {glb_path}: {error}') from error


def _extract_texture(mesh: trimesh.Trimesh, path: str) -> str:
    """Extract and return the mesh's texture."""

    material = getattr(mesh.visual, 'material', None)
    image = getattr(material, 'baseColorTexture', None)
    if isinstance(image, Image.Image):
        if not os.path.isfile(path):
            image.save(path, quality=95)
        return path
    else:
        return ''
