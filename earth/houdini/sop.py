"""Entry points called by the Earth HDA Python nodes."""

from __future__ import annotations

import logging
import os
from typing import cast

import hou

from .. import glb
from ..cache import Cache
from ..geodesy import tangent_frame
from ..region import RadialRegion, RectangularRegion
from ..tiles import TileLoader
from . import atlas, geometry

logger = logging.getLogger(__name__)


def download_tiles() -> None:
    """Download and select the 3D Tiles for the node's location."""

    node = cast(hou.SopNode, hou.pwd())
    parent = cast(hou.SopNode, node.parent())

    # Parameters
    coordinates = cast(tuple[float, float], parent.evalParmTuple('coordinates'))
    latitude, longitude = coordinates
    parm = cast(hou.Parm, parent.parm('falloff'))
    falloff = parm.evalAsFloat()

    shape = parent.evalParm('shape')
    frame = tangent_frame(latitude, longitude)

    size = cast(tuple[float, float], parent.evalParmTuple('size'))
    width, height = size
    parm = cast(hou.Parm, parent.parm('radius'))
    radius = parm.evalAsFloat()

    parm = cast(hou.Parm, parent.parm('google_api_key'))
    google_api_key = parm.evalAsString()
    parm = cast(hou.Parm, parent.parm('cachedir'))
    cachedir = parm.evalAsString()

    parm = cast(hou.Parm, parent.parm('min_error'))
    min_error = parm.evalAsFloat()
    parm = cast(hou.Parm, parent.parm('max_error'))
    max_error = parm.evalAsFloat()
    parm = cast(hou.Parm, parent.parm('max_meshes'))
    max_meshes = parm.evalAsInt()

    logger.debug(
        'Parameters: lat=%s lon=%s shape=%s falloff=%s error=%s-%s max_meshes=%s',
        latitude,
        longitude,
        shape,
        falloff,
        min_error,
        max_error,
        max_meshes,
    )

    if shape == 'rectangle':
        region = RectangularRegion(
            center=frame.center,
            east=frame.east,
            north=frame.north,
            width=width,
            height=height,
            falloff=falloff,
        )
    else:
        region = RadialRegion(center=frame.center, radius=radius, falloff=falloff)

    # Download
    loader = TileLoader(api_key=google_api_key, cache=Cache(cachedir))
    with hou.InterruptableOperation('Downloading', 'Selecting tiles', True) as op:  # ty: ignore[invalid-context-manager]
        loader.progress = op.updateLongProgress
        manifest = loader.collect(
            region=region,
            min_error=min_error,
            max_error=max_error,
            max_meshes=max_meshes,
        )

    logger.debug(
        'Manifest: %d tile(s), %d warning(s)',
        len(manifest.tiles),
        len(manifest.warnings),
    )

    # Store tiles on geo
    geo = node.geometry()
    if geo is None:
        return

    # Store the local tangent transform for the optional Transform By Attribute node.
    geo.addAttrib(hou.attribType.Global, 'xform', (0.0,) * 16)
    geo.setGlobalAttribValue('xform', frame.transform.flatten().tolist())

    # Create one point per tile carrying its mesh path and geometric error.
    geo.addAttrib(hou.attribType.Point, 'mesh', '')
    geo.addAttrib(hou.attribType.Point, 'error', 0.0)
    geo.createPoints([(0.0, 0.0, 0.0)] * len(manifest.tiles))
    geo.setPointStringAttribValues('mesh', [tile.mesh for tile in manifest.tiles])
    geo.setPointFloatAttribValues('error', [tile.error for tile in manifest.tiles])

    if manifest.warnings:
        for warning in manifest.warnings:
            logger.warning(warning)
        raise hou.NodeWarning('tile request(s) failed')


def build_geometry() -> None:
    """Assemble the downloaded GLB tiles into Houdini geometry."""

    node = cast(hou.SopNode, hou.pwd())
    geo = node.geometry()
    if geo is None:
        return

    meshes = [point.stringAttribValue('mesh') for point in geo.points()]

    # Remove the tile marker points and attributes, keeping detail attributes
    for name in ('mesh', 'error'):
        attrib = geo.findPointAttrib(name)
        if attrib is not None:
            attrib.destroy()
    geo.deletePoints(geo.points())

    chunks: list[glb.Chunk] = []
    problems: list[str] = []
    with hou.InterruptableOperation('Assembling GLTF', 'Loading meshes', True) as op:  # ty: ignore[invalid-context-manager]
        total = len(meshes)
        for index, path in enumerate(meshes):
            if not os.path.isfile(path):
                problems.append(f'{path} not found')
                continue

            try:
                chunk = glb.load_chunk(path)
            except glb.GlbError as error:
                problems.append(str(error))
                continue

            chunks.append(chunk)

            progress = (index + 1) / total
            op.updateLongProgress(progress, f'Loading mesh {index + 1}/{total}')
        geometry.build_chunk(geo, chunks, op.updateLongProgress)

    logger.debug('Decoded %d of %d tile mesh(es)', len(chunks), len(meshes))

    if problems:
        for problem in problems:
            logger.warning(problem)
        raise hou.NodeWarning('mesh(es) could not be loaded')


def build_atlas() -> None:
    """Pack the per-tile textures into atlas pages."""

    node = cast(hou.SopNode, hou.pwd())
    parent = cast(hou.SopNode, node.parent())

    geo = node.geometry()
    if geo is None:
        return

    resolution = cast(hou.Parm, parent.parm('resolution')).evalAsInt()
    padding = cast(hou.Parm, parent.parm('padding')).evalAsInt()
    directory = cast(hou.Parm, parent.parm('dir')).evalAsString()
    prefix = cast(hou.Parm, parent.parm('prefix')).evalAsString()
    extension = cast(hou.Parm, parent.parm('ext')).evalAsString()

    logger.debug(
        'Atlas: resolution=%s padding=%s directory=%s prefix=%s extension=%s',
        resolution,
        padding,
        directory,
        prefix,
        extension,
    )

    with hou.InterruptableOperation('Building atlas', 'Packing textures', True) as op:  # ty: ignore[invalid-context-manager]
        atlas.build(
            geo,
            resolution,
            padding,
            directory,
            prefix,
            extension,
            op.updateLongProgress,
        )


def clear_cache() -> None:
    """Delete the downloaded tiles from the cache."""

    node = cast(hou.SopNode, hou.pwd())
    parm = cast(hou.Parm, node.parm('cachedir'))
    cachedir = parm.evalAsString()
    Cache(cachedir).clear()
