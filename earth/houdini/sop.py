"""Entry points called by the Earth HDA Python nodes."""

from __future__ import annotations

import logging
import os
from typing import cast

import hou

from .. import glb
from ..cache import Cache, remove_files
from ..geodesy import tangent_frame, tangent_offset
from ..region import RadialRegion, RectangularRegion
from ..tiles import TileLoader
from . import atlas, geometry

logger = logging.getLogger(__name__)


def build_geometry(
    coordinates: tuple[float, float],
    center: tuple[float, float],
    shape: str,
    size: tuple[float, float],
    min_error: float,
    max_error: float,
    falloff: float,
    falloff_exponent: float,
    extract_textures: bool,
    pack_error: float,
    google_api_key: str,
    cachedir: str,
    max_meshes: int,
) -> None:
    """Build the selected tiles in one download and geometry pass."""

    node = cast(hou.SopNode, hou.pwd())

    geo = node.geometry()
    if geo is None:
        return

    frame = tangent_frame(coordinates[0], coordinates[1])
    region_center = tangent_offset(frame, center[0], center[1])
    if shape == 'rectangle':
        region = RectangularRegion(
            center=region_center,
            east=frame.east,
            south=frame.south,
            width=size[0],
            height=size[1],
            falloff=falloff,
            exponent=falloff_exponent,
        )
    else:
        region = RadialRegion(
            center=region_center,
            radius=min(size[0], size[1]) * 0.5,
            falloff=falloff,
            exponent=falloff_exponent,
        )

    geo.clear()
    geo.addAttrib(hou.attribType.Global, 'xform', (0.0,) * 16)
    geo.setGlobalAttribValue('xform', frame.transform.flatten().tolist())

    loader = TileLoader(api_key=google_api_key, cache=Cache(cachedir))
    chunks: list[glb.Chunk] = []
    problems: list[str] = []

    with hou.InterruptableOperation('Building Earth', 'Selecting tiles', True) as op:  # ty: ignore[invalid-context-manager]

        def download_progress(progress: float, label: str) -> None:
            op.updateLongProgress(max(progress, 0.0) * 0.5, label)

        loader.progress = download_progress
        manifest = loader.collect(
            region=region,
            min_error=min_error,
            max_error=max_error,
            max_meshes=max_meshes,
            pack_error=pack_error,
        )
        problems.extend(manifest.warnings)

        # Decode and assemble the meshes
        total = len(manifest.tiles)
        for index, tile in enumerate(manifest.tiles):
            if not os.path.isfile(tile.mesh):
                problems.append(f'{tile.mesh} not found')
                continue

            try:
                chunk = glb.load_chunk(tile.mesh, extract_textures=extract_textures)
            except glb.GlbError as error:
                problems.append(str(error))
                continue

            if tile.tile_id is not None:
                chunk.name = tile.tile_id
            chunks.append(chunk)

            if total:
                op.updateLongProgress(
                    0.5 + 0.25 * (index + 1) / total,
                    f'Loading mesh {index + 1}/{total}',
                )

        def build_progress(progress: float, label: str) -> None:
            op.updateLongProgress(0.75 + 0.25 * progress, label)

        geometry.build_chunk(geo, chunks, build_progress)
        op.updateLongProgress(1.0, 'Finished')

    logger.debug(
        'Decoded %d of %d tile mesh(es), %d warning(s)',
        len(chunks),
        len(manifest.tiles),
        len(problems),
    )

    if problems:
        for problem in problems:
            logger.warning(problem)
        raise hou.NodeWarning('mesh(es) could not be loaded')


def build_atlas(
    resolution: int,
    padding: int,
    directory: str,
    extension: str,
) -> None:
    """Pack the per-tile textures into atlas pages."""

    node = cast(hou.SopNode, hou.pwd())

    geo = node.geometry()
    if geo is None:
        return

    # Build the atlas
    with hou.InterruptableOperation('Building atlas', 'Packing textures', True) as op:  # ty: ignore[invalid-context-manager]
        atlas.build(
            geo,
            resolution,
            padding,
            directory,
            extension,
            progress=op.updateLongProgress,
        )


def clear_cache() -> None:
    """Delete the downloaded tiles from the cache."""

    node = cast(hou.SopNode, hou.pwd())
    parm = cast(hou.Parm, node.parm('cachedir'))
    cachedir = parm.evalAsString()
    Cache(cachedir).clear()

    node.cook(force=True)


def clear_maps() -> None:
    """Delete the downloaded tiles from the cache."""

    node = cast(hou.SopNode, hou.pwd())
    parm = cast(hou.Parm, node.parm('dir'))
    directory = parm.evalAsString()
    remove_files(directory)

    node.cook(force=True)
