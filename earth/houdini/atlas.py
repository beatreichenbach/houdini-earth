"""Build texture atlases from geometry with per-primitive textures."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence

import hou
import numpy as np
from PIL import Image

from .. import atlas as atlas_module
from ..atlas import AtlasPage, Placement

logger = logging.getLogger(__name__)

Progress = Callable[[float, str], None]


def build(
    geo: hou.Geometry,
    resolution: int,
    padding: int,
    directory: str,
    prefix: str,
    extension: str,
    progress: Progress | None = None,
) -> None:
    """Pack the geometry's per-primitive textures into atlas pages."""

    if geo.findPrimAttrib('texture') is None:
        raise ValueError('geometry has no texture attribute')

    prims = list(geo.prims())
    textures = [str(prim.attribValue('texture')) for prim in prims]

    # Pack textures
    if progress is not None:
        progress(0, 'Packing textures ...')
    sizes = _sizes(textures)
    pages = atlas_module.plan_pages(sizes, resolution, padding)

    # Writing atlases
    placements: dict[str, Placement] = {}
    paths: dict[str, str] = {}
    os.makedirs(directory, exist_ok=True)
    total = len(pages)
    for index, page in enumerate(pages):
        path = os.path.join(directory, f'{prefix}{page.index}.{extension}')
        _write_page(page, path, extension)
        for placement in page.placements:
            placements[placement.texture] = placement
            paths[placement.texture] = path

        if progress is not None:
            progress((index + 1) / total, f'Writing atlas {index + 1}/{total}')

    # Remap uvs
    if progress is not None:
        progress(0, 'Remapping uvs ...')

    # Move every primitive's vertices into its texture's atlas rect
    uv_attrib = geo.findVertexAttrib('uv')
    if uv_attrib:
        values = np.asarray(geo.vertexFloatAttribValues('uv'), dtype=np.float64)
        values = values.reshape(-1, uv_attrib.size())
        result = values.copy()
        total = len(prims)
        for index, (prim, texture) in enumerate(zip(prims, textures, strict=True)):
            placement = placements.get(texture)
            if placement is None:
                continue

            indices = [vertex.linearNumber() for vertex in prim.vertices()]
            remapped = atlas_module.remap_uv(values[indices, :2], placement, resolution)
            result[indices, 0] = remapped[:, 0]
            result[indices, 1] = remapped[:, 1]

            if progress is not None:
                progress((index + 1) / total, f'Remapping uvs {index + 1}/{total}')

        geo.setVertexFloatAttribValues('uv', result.reshape(-1).tolist())

    # Replace texture attribute
    geo.addAttrib(hou.attribType.Prim, 'atlas', '')
    for prim, texture in zip(prims, textures, strict=True):
        prim.setAttribValue('atlas', paths.get(texture, ''))
    texture_attrib = geo.findPrimAttrib('texture')
    if texture_attrib is not None:
        texture_attrib.destroy()

    logger.debug('Packed %d texture(s) into %d atlas page(s)', len(sizes), len(pages))


def _sizes(textures: Sequence[str]) -> dict[str, tuple[int, int]]:
    """Return the pixel size of every unique, existing texture."""

    sizes: dict[str, tuple[int, int]] = {}
    for texture in textures:
        if not texture or texture in sizes:
            continue
        if not os.path.isfile(texture):
            logger.warning('Texture not found: %s', texture)
            continue
        with Image.open(texture) as image:
            sizes[texture] = image.size
    return sizes


def _write_page(page: AtlasPage, path: str, extension: str) -> None:
    """Composite a page's textures and write it to disk."""

    atlas = Image.new('RGB', (page.size, page.size), (0, 0, 0))
    for placement in page.placements:
        with Image.open(placement.texture) as image:
            atlas.paste(image.convert('RGB'), (placement.x, placement.y))
    if extension.lower() in ('jpg', 'jpeg'):
        atlas.save(path, quality=95)
    else:
        atlas.save(path)
