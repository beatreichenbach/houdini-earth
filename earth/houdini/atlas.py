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

NAME_ATTR = 'name'
TEXTURE_ATTR = 'texture'


def build(
    geo: hou.Geometry,
    resolution: int,
    padding: int,
    directory: str,
    extension: str,
    progress: Progress | None = None,
) -> None:
    """Pack the geometry's per-primitive textures into atlas pages."""

    if geo.findPrimAttrib(TEXTURE_ATTR) is None:
        raise hou.NodeError(f'geometry has no {TEXTURE_ATTR} attribute')

    prims = list(geo.prims())
    textures = [str(prim.attribValue(TEXTURE_ATTR)) for prim in prims]

    if prims and geo.findPrimAttrib(NAME_ATTR) is not None:
        tile = str(prims[0].attribValue(NAME_ATTR))
    else:
        tile = 'tile'

    # Pack textures
    if progress is not None:
        progress(0, 'Packing textures ...')
    sizes = _sizes(textures)

    try:
        pages = atlas_module.plan_pages(sizes, resolution, padding)
    except atlas_module.AtlasError as error:
        raise hou.NodeError(str(error)) from error

    # Writing atlases
    placements: dict[str, tuple[Placement, AtlasPage]] = {}
    paths: dict[str, str] = {}
    os.makedirs(directory, exist_ok=True)
    total = max(1, len(pages))
    for index, page in enumerate(pages):
        name = _atlas_id(tile, page.index)
        path = os.path.join(directory, f'{name}.{extension}')
        _write_page(page, path, extension)
        for placement in page.placements:
            placements[placement.texture] = (placement, page)
            paths[placement.texture] = path

        if progress is not None:
            progress((index + 1) / total, f'Writing atlas {index + 1}/{total}')

    # Remap uvs
    if progress is not None:
        progress(0, 'Remapping uvs ...')

    uv_attrib = geo.findVertexAttrib('uv')
    if uv_attrib:
        values = np.asarray(geo.vertexFloatAttribValues('uv'), dtype=np.float64)
        values = values.reshape(-1, uv_attrib.size())
        result = values.copy()
        total = len(prims)
        for index, (prim, texture) in enumerate(zip(prims, textures, strict=True)):
            found = placements.get(texture)
            if found is None:
                continue

            placement, page = found
            indices = [vertex.linearNumber() for vertex in prim.vertices()]
            remapped = atlas_module.remap_uv(
                values[indices, :2], placement, page.width, page.height
            )
            result[indices, 0] = remapped[:, 0]
            result[indices, 1] = remapped[:, 1]

            if progress is not None:
                progress((index + 1) / total, f'Remapping uvs {index + 1}/{total}')

        geo.setVertexFloatAttribValues('uv', result.reshape(-1).tolist())

    # Attributes
    if not geo.findPrimAttrib(TEXTURE_ATTR):
        geo.addAttrib(hou.attribType.Prim, TEXTURE_ATTR, '')
    for prim, texture in zip(prims, textures, strict=True):
        prim.setAttribValue(TEXTURE_ATTR, paths.get(texture, ''))

    if not geo.findPrimAttrib(NAME_ATTR):
        geo.addAttrib(hou.attribType.Prim, NAME_ATTR, '')
    for prim, texture in zip(prims, textures, strict=True):
        found = placements.get(texture)
        page = found[1] if found is not None else None
        name = _atlas_id(tile, page.index) if page is not None else ''
        prim.setAttribValue(NAME_ATTR, name)

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

    atlas = Image.new('RGB', (page.width, page.height), (0, 0, 0))
    for placement in page.placements:
        with Image.open(placement.texture) as image:
            atlas.paste(image.convert('RGB'), (placement.x, placement.y))

    if extension.lower() in ('jpg', 'jpeg'):
        atlas.save(path, quality=95)
    else:
        atlas.save(path)


def _atlas_id(tile: str, index: int) -> str:
    """Return the identifier used for this tile's index."""

    return f'{tile}_{index}'
