"""Model of a 3D Tiles tileset tree."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from urllib.parse import urlsplit

import numpy as np


class Refine(StrEnum):
    """How a tile's children combine with its own content."""

    REPLACE = 'REPLACE'
    ADD = 'ADD'


@dataclass(frozen=True)
class BoundingVolume:
    """An oriented bounding box of a tile."""

    box: tuple[float, ...]

    @property
    def center(self) -> np.ndarray:
        """Return the box center in the mesh frame."""

        b = self.box
        return np.array([b[0], b[2], -b[1]])

    @property
    def axes(self) -> np.ndarray:
        """Return the box half-axis vectors in the mesh frame."""

        b = self.box
        return np.array(
            [
                [b[3], b[5], -b[4]],
                [b[6], b[8], -b[7]],
                [b[9], b[11], -b[10]],
            ]
        )

    def signed_distance(self, point: np.ndarray) -> float:
        """Return the signed distance from a point to the box surface."""

        axes = self.axes
        size = np.linalg.norm(axes, axis=1)
        rotation = axes / size[:, None]
        local = rotation @ (point - self.center)
        q = np.abs(local) - size
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
        return urlsplit(self.content_uri).path.endswith('.json')


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


def _bounding_volume(data: dict[str, Any]) -> BoundingVolume | None:
    """Return the Bounding Volume from JSON data."""

    volume = data.get('boundingVolume')
    if isinstance(volume, dict):
        box = volume.get('box')
        if isinstance(box, list):
            bounding_volume = BoundingVolume(tuple(float(value) for value in box))
            return bounding_volume
        else:
            raise NotImplementedError(f'unsupported bounding volume: {sorted(volume)}')
    return None
