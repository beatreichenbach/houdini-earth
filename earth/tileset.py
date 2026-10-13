"""Model of a 3D Tiles tileset tree."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np


class Refine(StrEnum):
    """How a tile's children combine with its own content."""

    REPLACE = 'REPLACE'
    ADD = 'ADD'


@dataclass(frozen=True)
class BoundingVolume:
    """An oriented bounding box of a tile."""

    box: tuple[float, ...]

    def matrix(self) -> np.ndarray:
        """Return the box-to-mesh transform as a 4x4 matrix."""

        b = self.box
        bounding_box_to_mesh = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        box_matrix = np.array(
            [
                [b[3], b[4], b[5], 0.0],
                [b[6], b[7], b[8], 0.0],
                [b[9], b[10], b[11], 0.0],
                [b[0], b[1], b[2], 1.0],
            ]
        )
        return box_matrix @ bounding_box_to_mesh

    def signed_distance(self, point: np.ndarray) -> float:
        """Return the signed distance from a point to the box surface."""

        matrix = self.matrix()
        axis = matrix[:3, :3]
        size = np.linalg.norm(axis, axis=1)
        frame = np.eye(4)
        frame[:3, :3] = axis / size[:, None]
        frame[3, :3] = matrix[3, :3]
        inverse = np.linalg.inv(frame)

        p = np.append(point, 1.0) @ inverse
        q = np.abs(p[:3]) - size
        outside = np.linalg.norm(np.maximum(q, 0.0))
        inside = min(float(np.max(q)), 0.0)
        return float(outside + inside)


@dataclass
class Tile:
    """A node in the tileset tree."""

    geometric_error: float = 0.0
    bounding_volume: BoundingVolume | None = None
    content_uri: str | None = None
    refine: Refine | None = None
    children: tuple[Tile, ...] = ()

    @property
    def is_external_tileset(self) -> bool:
        """Return whether the content URI points to another tileset."""

        if self.content_uri is None:
            return False
        return self.content_uri.split('?')[0].endswith('.json')


def parse_tile(data: dict[str, Any]) -> Tile:
    """
    Return a Tile parsed from a tileset JSON node.

    :raises NotImplementedError: if a tile uses an unsupported bounding volume.
    """

    uri: str | None = None
    content = data.get('content')
    if isinstance(content, dict):
        uri = content.get('uri')

    error = float(data.get('geometricError', 0.0))
    bounding_volume = _bounding_volume(data)
    refine = Refine(data['refine']) if data.get('refine') else None
    children = tuple(parse_tile(child) for child in data.get('children', []))

    tile = Tile(
        geometric_error=error,
        bounding_volume=bounding_volume,
        content_uri=uri,
        refine=refine,
        children=children,
    )
    return tile


def _bounding_volume(data: dict) -> BoundingVolume | None:
    """Return the Bounding Volume from JSON data."""

    volume = data.get('boundingVolume')
    if isinstance(volume, dict):
        box = volume.get('box')
        if isinstance(box, list):
            bounding_volume = BoundingVolume(tuple(float(value) for value in box))
            return bounding_volume
        else:
            keys = sorted(volume) if isinstance(volume, dict) else volume
            raise NotImplementedError(f'Unsupported bounding volume: {keys}')
    return None
