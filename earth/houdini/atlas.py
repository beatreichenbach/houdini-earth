"""Build texture atlases from geometry with per-primitive textures."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence

import hou
from PIL import Image

from .. import atlas as atlas_module
from ..atlas import AtlasPage, Placement

logger = logging.getLogger(__name__)

Progress = Callable[[float], None]

TEXTURE_ATTR = 'texture'
UV_SCALE_ATTR = 'uv_scale'
UV_OFFSET_ATTR = 'uv_offset'
TILE_NAME = 'tile'


def build(
    geo: hou.Geometry,
    use_attribute: bool,
    attribute: str,
    resolution: int,
    padding: int,
    directory: str,
    extension: str,
    progress: Progress | None = None,
) -> None:
    """Pack the geometry's per-primitive textures into atlas pages."""

    if geo.findPrimAttrib(TEXTURE_ATTR) is None:
        raise hou.NodeError(f'geometry has no {TEXTURE_ATTR} attribute')

    prims = geo.prims()
    textures = [str(prim.attribValue(TEXTURE_ATTR)) for prim in prims]

    groups: dict[str, list[int]] = {}
    if use_attribute and geo.findPrimAttrib(attribute) is not None:
        for index, prim in enumerate(prims):
            tile = str(prim.attribValue(attribute))
            if not tile:
                tile = TILE_NAME
            groups.setdefault(tile, []).append(index)
    else:
        groups[TILE_NAME] = list(range(len(prims)))

    # Attributes
    if not geo.findPrimAttrib(UV_SCALE_ATTR):
        geo.addAttrib(hou.attribType.Prim, UV_SCALE_ATTR, (1.0, 1.0))
    if not geo.findPrimAttrib(UV_OFFSET_ATTR):
        geo.addAttrib(hou.attribType.Prim, UV_OFFSET_ATTR, (0.0, 0.0))

    os.makedirs(directory, exist_ok=True)
    total_groups = max(1, len(groups))
    group_percentage = 0
    group_delta = 1.0 / total_groups
    for group_index, (tile, indices) in enumerate(groups.items()):
        # Plan pages
        group_textures = [textures[index] for index in indices]
        sizes = _sizes(group_textures)
        try:
            pages = atlas_module.plan_pages(sizes, resolution, padding)
        except atlas_module.AtlasError as error:
            raise hou.NodeError(str(error)) from error

        # Write pages
        placements: dict[str, tuple[Placement, AtlasPage]] = {}
        paths: dict[str, str] = {}
        total = max(1, len(pages))
        for page_index, page in enumerate(pages):
            file_name = f'{tile}_{page.index}.{extension}'
            path = os.path.join(directory, file_name)
            _write_page(page, path, extension)
            for placement in page.placements:
                placements[placement.texture] = (placement, page)
                paths[placement.texture] = path

            if progress is not None:
                page_percentage = (page_index + 1) / total
                percentage = group_percentage + group_delta * page_percentage
                progress(percentage)

        # Plan remapping
        for index in indices:
            prim = prims[index]
            texture = textures[index]
            found = placements.get(texture)
            if found is None:
                prim.setAttribValue(TEXTURE_ATTR, '')
                prim.setAttribValue(UV_SCALE_ATTR, (1.0, 1.0))
                prim.setAttribValue(UV_OFFSET_ATTR, (0.0, 0.0))
                continue

            placement, page = found
            transform = atlas_module.remap_transform(placement, page.width, page.height)
            prim.setAttribValue(TEXTURE_ATTR, paths[texture])
            prim.setAttribValue(UV_SCALE_ATTR, transform.scale)
            prim.setAttribValue(UV_OFFSET_ATTR, transform.offset)

        group_percentage = (group_index + 1) / total_groups
        if progress is not None:
            progress(group_percentage)


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
