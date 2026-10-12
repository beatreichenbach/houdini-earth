"""Decode GLB tiles into points, faces, UVs and textures."""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field

import DracoPy  # ty: ignore[unresolved-import]
import numpy as np
import pygltflib as GLTF2

_COMPONENT_DTYPES = {
    GLTF2.BYTE: np.int8,
    GLTF2.UNSIGNED_BYTE: np.uint8,
    GLTF2.SHORT: np.int16,
    GLTF2.UNSIGNED_SHORT: np.uint16,
    GLTF2.UNSIGNED_INT: np.uint32,
    GLTF2.FLOAT: np.float32,
}

_NUM_COMPONENTS = {
    GLTF2.SCALAR: 1,
    GLTF2.VEC2: 2,
    GLTF2.VEC3: 3,
    GLTF2.VEC4: 4,
    GLTF2.MAT2: 4,
    GLTF2.MAT3: 9,
    GLTF2.MAT4: 16,
}

_GLB_ERRORS = (
    OSError,
    ValueError,
    RuntimeError,
    KeyError,
    IndexError,
    TypeError,
    struct.error,
    AssertionError,
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

    primitives: list[Primitive] = field(default_factory=list)


def _read_accessor(gltf: GLTF2.GLTF2, index: int) -> np.ndarray:
    """Return the array stored in a glTF accessor."""

    accessor = gltf.accessors[index]
    assert accessor.bufferView is not None
    buffer_view = gltf.bufferViews[accessor.bufferView]
    buffer = gltf.buffers[buffer_view.buffer]
    raw = gltf.get_data_from_buffer_uri(buffer.uri)

    components = _NUM_COMPONENTS[accessor.type]
    dtype = _COMPONENT_DTYPES[accessor.componentType]
    element_size = np.dtype(dtype).itemsize * components

    base = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
    stride = buffer_view.byteStride or element_size

    if stride == element_size:
        byte_length = accessor.count * element_size
        chunk = raw[base : base + byte_length]
        array = np.frombuffer(chunk, dtype=dtype, count=accessor.count * components)
        return array.reshape((accessor.count, components))

    out = np.empty((accessor.count, components), dtype=dtype)
    for k in range(accessor.count):
        offset = base + k * stride
        chunk = raw[offset : offset + element_size]
        out[k, :] = np.frombuffer(chunk, dtype=dtype, count=components)
    return out


def _extract_textures(gltf: GLTF2.GLTF2, glb_path: str) -> list[str]:
    """Write the GLB's texture images next to it and return their paths."""

    name = os.path.splitext(glb_path)[0]
    paths = []
    for index in range(len(gltf.images)):
        path = f'{name}_{index}.jpg'
        buffer_view_index = gltf.images[index].bufferView
        if not os.path.isfile(path) and buffer_view_index is not None:
            buffer_view = gltf.bufferViews[buffer_view_index]
            buffer = gltf.buffers[buffer_view.buffer]
            data = gltf.get_data_from_buffer_uri(buffer.uri)
            offset = buffer_view.byteOffset or 0
            with open(path, 'wb') as file:
                file.write(data[offset : offset + buffer_view.byteLength])
        paths.append(path)
    return paths


def _primitive_texture(
    gltf: GLTF2.GLTF2, primitive: GLTF2.Primitive, textures: list[str], index: int
) -> str | None:
    """Return the texture path used by a primitive."""

    material_index = primitive.material
    if material_index is not None:
        material = gltf.materials[material_index]
        base_color = material.pbrMetallicRoughness
        if base_color is not None and base_color.baseColorTexture is not None:
            texture_index = base_color.baseColorTexture.index
            source = gltf.textures[texture_index].source
            if source is not None and source < len(textures):
                return textures[source]
    if index < len(textures):
        return textures[index]
    return None


def _read_draco(
    gltf: GLTF2.GLTF2, primitive: GLTF2.Primitive
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return points, faces and UVs decoded from Draco compression."""

    assert primitive.extensions is not None
    extension = primitive.extensions['KHR_draco_mesh_compression']
    buffer_view = gltf.bufferViews[extension['bufferView']]
    buffer = gltf.buffers[buffer_view.buffer]
    data = gltf.get_data_from_buffer_uri(buffer.uri)
    offset = buffer_view.byteOffset or 0
    decoded = DracoPy.decode(data[offset : offset + buffer_view.byteLength])
    points = np.array(decoded.points, dtype=np.float64)
    faces = np.array(decoded.faces, dtype=np.int64)
    uvs = np.array(decoded.tex_coord, dtype=np.float64)
    return points, faces, uvs


def _read_primitive(
    gltf: GLTF2.GLTF2, primitive: GLTF2.Primitive
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return points, faces and UVs for a GLB primitive."""

    if primitive.extensions and 'KHR_draco_mesh_compression' in primitive.extensions:
        points, faces, uvs = _read_draco(gltf, primitive)
    else:
        position = primitive.attributes.POSITION
        assert position is not None
        points = _read_accessor(gltf, position).astype(np.float64)

        texcoord = getattr(primitive.attributes, 'TEXCOORD_0', None)
        if texcoord is not None:
            uvs = _read_accessor(gltf, texcoord).astype(np.float64)
        else:
            uvs = np.zeros((points.shape[0], 2), dtype=np.float64)

        if primitive.indices is not None:
            indices = (
                _read_accessor(gltf, primitive.indices).reshape(-1).astype(np.int64)
            )
        else:
            indices = np.arange(points.shape[0], dtype=np.int64)

        triangle_count = len(indices) // 3
        faces = indices[: triangle_count * 3].reshape((triangle_count, 3))

    # Transform the points by the first node's matrix.
    matrix = np.eye(4)
    if gltf.nodes and gltf.nodes[0].matrix is not None:
        matrix = np.array(gltf.nodes[0].matrix).reshape(4, 4)
    homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    return (homogeneous @ matrix)[:, :3], faces, uvs


def load_chunk(glb_path: str) -> Chunk:
    """
    Return the decoded chunk for a GLB file.

    :raises GlbError: if the tile cannot be read or decoded.
    """

    try:
        gltf = GLTF2.GLTF2().load(glb_path)
        textures = _extract_textures(gltf, glb_path)

        chunk = Chunk()
        mesh = gltf.meshes[0]
        for index, primitive in enumerate(mesh.primitives):
            points, faces, uvs = _read_primitive(gltf, primitive)
            uvs = uvs * np.array((1.0, -1.0)) + np.array((0.0, 1.0))
            texture = _primitive_texture(gltf, primitive, textures, index)
            chunk.primitives.append(Primitive(points, faces, uvs, texture))
        return chunk
    except _GLB_ERRORS as error:
        raise GlbError(f'Could not decode {glb_path}: {error}') from error
