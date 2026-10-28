"""Entry points called by the Earth HDA Python nodes."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import cast

import hou

from ..cache import Cache, remove_files
from ..geodesy import tangent_frame, tangent_offset
from ..region import RadialRegion, RectangularRegion
from ..tiles import TileLoader
from . import atlas, geometry

logger = logging.getLogger(__name__)

InterruptableOperation = cast(
    Callable[..., AbstractContextManager[hou.InterruptableOperation]],
    hou.InterruptableOperation,
)


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
        radius = min(size[0], size[1]) * 0.5
        region = RadialRegion(
            center=region_center,
            radius=radius,
            falloff=falloff,
            exponent=falloff_exponent,
        )

    loader = TileLoader(api_key=google_api_key, cache=Cache(cachedir))
    warnings: list[str] = []

    with InterruptableOperation('Initializing', 'Building Earth', True) as op:
        # Collect tiles
        with InterruptableOperation('Loading Tiles') as sub_op:
            loader.progress = sub_op.updateProgress
            manifest = loader.collect(
                region=region,
                min_error=min_error,
                max_error=max_error,
                max_meshes=max_meshes,
                pack_error=pack_error,
            )
            warnings.extend(manifest.errors)
        op.updateLongProgress(0.5)

        # Load meshes
        with InterruptableOperation('Loading Meshes') as sub_op:
            chunks = geometry.load_chunks(
                tiles=manifest.tiles,
                extract_textures=extract_textures,
                progress=sub_op.updateProgress,
                warnings=warnings,
            )
        op.updateLongProgress(0.75)

        # Build Meshes
        with InterruptableOperation('Loading Meshes') as sub_op:
            geo.clear()

            geometry.build_chunk(geo=geo, chunks=chunks, progress=sub_op.updateProgress)

            geo.addAttrib(hou.attribType.Global, 'xform', (0.0,) * 16)
            geo.setGlobalAttribValue('xform', frame.transform.flatten().tolist())
        op.updateLongProgress(1.0)

    if warnings:
        for warning in warnings:
            logger.warning(warning)
        raise hou.NodeWarning('Some meshes could not be loaded.')


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
    with InterruptableOperation('Building atlas', open_interrupt_dialog=True) as op:
        atlas.build(
            geo,
            resolution,
            padding,
            directory,
            extension,
            progress=op.updateProgress,
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
