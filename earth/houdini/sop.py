"""Entry points called by the Earth HDA Python nodes."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import cast

import hou

from .. import glb
from ..cache import Cache
from ..geodesy import tangent_frame
from ..region import RadialRegion, RectangularRegion
from ..tiles import Progress, TileLoader, TileResult
from . import geometry


def _progress(operation: hou.InterruptableOperation) -> Progress:
    """Return a progress callback that updates an interruptable operation."""

    def update(fraction: float, message: str) -> None:
        operation.updateLongProgress(fraction, message)

    return update


def _warn(node: hou.SopNode, issues: Sequence[str], subject: str) -> None:
    """Report a single warning summarizing the issues."""

    if not issues:
        return
    shown = '; '.join(issues[:5])
    more = len(issues) - 5
    suffix = f' (+{more} more)' if more > 0 else ''
    node.warning(  # ty: ignore[unresolved-attribute]
        f'{len(issues)} {subject}: {shown}{suffix}'
    )


def _create_tile_points(geo: hou.Geometry, tiles: Sequence[TileResult]) -> None:
    """Create one point per tile carrying its mesh path and geometric error."""

    geo.addAttrib(hou.attribType.Point, 'mesh', '')
    geo.addAttrib(hou.attribType.Point, 'error', 0.0)

    geo.createPoints([(0.0, 0.0, 0.0)] * len(tiles))
    geo.setPointStringAttribValues('mesh', [tile.mesh for tile in tiles])
    geo.setPointFloatAttribValues('error', [tile.error for tile in tiles])


def _consume_tile_points(geo: hou.Geometry) -> None:
    """Remove the tile marker points and attributes, keeping detail attributes."""

    for name in ('mesh', 'error'):
        attrib = geo.findPointAttrib(name)
        if attrib is not None:
            attrib.destroy()
    geo.deletePoints(geo.points())


def tiles() -> None:
    """Download and select the 3D Tiles for the node's location."""

    node = cast(hou.SopNode, hou.pwd())

    geo = node.geometry()
    if geo is None:
        return

    parm = cast(hou.Parm, node.parm('latitude'))
    latitude = parm.evalAsFloat()
    parm = cast(hou.Parm, node.parm('longitude'))
    longitude = parm.evalAsFloat()
    parm = cast(hou.Parm, node.parm('falloff'))
    falloff = parm.evalAsFloat()

    shape = node.evalParm('shape')
    frame = tangent_frame(latitude, longitude)

    parm = cast(hou.Parm, node.parm('width'))
    width = parm.evalAsFloat()
    parm = cast(hou.Parm, node.parm('height'))
    height = parm.evalAsFloat()
    parm = cast(hou.Parm, node.parm('radius'))
    radius = parm.evalAsFloat()

    parm = cast(hou.Parm, node.parm('google_api_key'))
    google_api_key = parm.evalAsString()
    parm = cast(hou.Parm, node.parm('cachedir'))
    cachedir = parm.evalAsString()

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

    loader = TileLoader(api_key=google_api_key, cache=Cache(cachedir))

    parm = cast(hou.Parm, node.parm('min_error'))
    min_error = parm.evalAsFloat()
    parm = cast(hou.Parm, node.parm('max_error'))
    max_error = parm.evalAsFloat()
    parm = cast(hou.Parm, node.parm('max_meshes'))
    max_meshes = parm.evalAsInt()

    with hou.InterruptableOperation(  # ty: ignore[invalid-context-manager]
        'Get geometry with Google API',
        long_operation_name='Selecting tiles',
        open_interrupt_dialog=True,
    ) as operation:
        loader.progress = _progress(operation)
        manifest = loader.collect(
            region=region,
            min_error=min_error,
            max_error=max_error,
            max_meshes=max_meshes,
        )

    # Store the local tangent transform for the optional Transform By Attribute node.
    geo.addAttrib(hou.attribType.Global, 'xform', (0.0,) * 16)
    geo.setGlobalAttribValue('xform', frame.transform.flatten().tolist())

    _create_tile_points(geo, manifest.tiles)
    _warn(node, manifest.warnings, 'tile request(s) failed')


def build() -> None:
    """Assemble the downloaded GLB tiles into Houdini geometry."""

    node = cast(hou.SopNode, hou.pwd())
    geo = node.geometry()
    assert geo is not None

    meshes = [point.stringAttribValue('mesh') for point in geo.points()]
    _consume_tile_points(geo)

    chunks: list[glb.Chunk] = []
    problems: list[str] = []
    with hou.InterruptableOperation(  # ty: ignore[invalid-context-manager]
        'Assembling GLTF',
        long_operation_name='Loading meshes',
        open_interrupt_dialog=True,
    ) as operation:
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
            operation.updateLongProgress(
                (index + 1) / total, f'Loading mesh {index + 1}/{total}'
            )
        geometry.build_geometry(geo, chunks)

    _warn(node, problems, 'mesh(es) could not be loaded')


def clear_cache() -> None:
    """Delete the downloaded tiles from the cache."""

    node = cast(hou.SopNode, hou.pwd())
    parm = cast(hou.Parm, node.parm('cachedir'))
    cachedir = parm.evalAsString()
    Cache(cachedir).clear()
