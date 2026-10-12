"""Entry points called by the Earth HDA Python nodes."""

from __future__ import annotations

import os
from typing import Any, cast

import hou
import numpy as np

from .. import glb
from ..cache import Cache
from ..region import RadialRegion, RectangularRegion, Region
from ..tiles import Progress, TileLoader
from . import geometry


def _node(kwargs: dict[str, Any]) -> hou.SopNode:
    """Return the Python SOP running this code."""

    node = kwargs.get('node')
    return cast(hou.SopNode, node if node is not None else hou.pwd())


def _param(node: hou.Node, name: str) -> hou.Parm | None:
    """Return a parameter by name, searching the node and its parents."""

    current: hou.Node | None = node
    while current is not None:
        op = cast(hou.OpNode, current)
        parm = op.parm(name)
        if parm is not None:
            return parm
        current = op.parent()
    return None


def _float(node: hou.Node, name: str, default: float) -> float:
    """Return a parameter evaluated as a float."""

    parm = _param(node, name)
    return float(parm.eval()) if parm is not None else default


def _int(node: hou.Node, name: str, default: int) -> int:
    """Return a parameter evaluated as an int."""

    parm = _param(node, name)
    return int(parm.eval()) if parm is not None else default


def _str(node: hou.Node, name: str, default: str) -> str:
    """Return a parameter evaluated as a string."""

    parm = _param(node, name)
    return str(parm.eval()) if parm is not None else default


def _bool(node: hou.Node, name: str, default: bool) -> bool:
    """Return a parameter evaluated as a bool."""

    parm = _param(node, name)
    return bool(parm.eval()) if parm is not None else default


def _menu(node: hou.Node, name: str, default: str) -> str:
    """Return a menu parameter evaluated as its token."""

    parm = _param(node, name)
    return parm.evalAsString() if parm is not None else default


def _detail_vector(geo: hou.Geometry, name: str) -> np.ndarray:
    """Return a detail attribute evaluated as a 3-vector."""

    attrib = geo.findGlobalAttrib(name)
    assert attrib is not None, f'missing detail attribute {name!r}'
    value = cast(tuple[float, ...], geo.attribValue(attrib))
    return np.array(value[:3])


def _progress(operation: hou.InterruptableOperation) -> Progress:
    """Return a progress callback that updates an interruptable operation."""

    def update(fraction: float, message: str) -> None:
        operation.updateLongProgress(fraction, message)

    return update


def _set_string_list(geo: hou.Geometry, name: str, values: list[str]) -> None:
    """Store a string list on a detail attribute."""

    geo.addAttrib(hou.attribType.Global, name, [''])
    geo.setGlobalAttribValue(name, values)


def tiles(kwargs: dict[str, Any]) -> None:
    """Download and select the 3D Tiles for the node's location."""

    node = _node(kwargs)
    geo = node.geometry()
    assert geo is not None

    center = _detail_vector(geo, 'center')
    falloff = _float(node, 'falloff', 100.0)
    region: Region
    if _menu(node, 'shape', 'circle') == 'rectangle':
        region = RectangularRegion(
            center=center,
            east=_detail_vector(geo, 'east'),
            north=_detail_vector(geo, 'north'),
            width=_float(node, 'width', 200.0),
            height=_float(node, 'height', 100.0),
            falloff=falloff,
        )
    else:
        region = RadialRegion(
            center=center,
            radius=_float(node, 'radius', 500.0),
            falloff=falloff,
        )

    loader = TileLoader(
        api_key=_str(node, 'google_api_key', ''),
        cache=Cache(_str(node, 'file_cache_path', '')),
        cached=_bool(node, 'cached', True),
    )

    with hou.InterruptableOperation(  # ty: ignore[invalid-context-manager]
        'Get geometry with Google API',
        long_operation_name='Selecting tiles',
        open_interrupt_dialog=True,
    ) as operation:
        loader.progress = _progress(operation)
        manifest = loader.collect(
            region,
            min_error=_float(node, 'min_error', 100.0),
            max_error=_float(node, 'max_error', 60000.0),
            max_meshes=_int(node, 'max_meshes', 1000),
        )

    _set_string_list(geo, 'meshes', manifest.meshes)
    _set_string_list(geo, 'boxes', manifest.boxes)
    _set_string_list(geo, 'errors', [str(error) for error in manifest.errors])


def build(kwargs: dict[str, Any]) -> None:
    """Assemble the downloaded GLB tiles into Houdini geometry."""

    node = _node(kwargs)
    geo = node.geometry()
    assert geo is not None
    meshes = geo.stringListAttribValue('meshes')
    errors = [float(value) for value in geo.stringListAttribValue('errors')]

    chunks: list[tuple[glb.Chunk, float]] = []
    with hou.InterruptableOperation(  # ty: ignore[invalid-context-manager]
        'Assembling GLTF',
        long_operation_name='Loading meshes',
        open_interrupt_dialog=True,
    ) as operation:
        total = len(meshes)
        for index, path in enumerate(meshes):
            if not os.path.isfile(path):
                print(f'File {path} not found')
                continue
            chunk = glb.load_chunk(path)
            error = errors[index] if index < len(errors) else 0.0
            chunks.append((chunk, error))
            operation.updateLongProgress(
                (index + 1) / total, f'Loading mesh {index + 1}/{total}'
            )
        geometry.build_geometry(geo, chunks)

    attr = geo.addAttrib(hou.attribType.Global, 'copyright', '')
    geo.setGlobalAttribValue(attr, geometry.copyright_string(chunks))


def clear_cache(kwargs: dict[str, Any]) -> None:
    """Delete the downloaded tiles from the cache."""

    node = _node(kwargs)
    Cache(_str(node, 'file_cache_path', '')).clear()
