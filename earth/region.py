"""Selection regions that decide which tiles to keep and at what LOD."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .tileset import BoundingVolume


class Region(ABC):
    """A selection region with a falloff."""

    @abstractmethod
    def evaluate(self, volume: BoundingVolume) -> float:
        """Return the blend from 0 (near) to 1 (far), negative if not kept."""

    @staticmethod
    def _falloff_blend(distance: float, radius: float, falloff: float) -> float:
        if distance > radius + falloff:
            return -1.0
        if falloff <= 0.0:
            return 0.0 if distance <= radius else -1.0
        blend = (distance - radius) / falloff
        return min(max(blend, 0.0), 1.0)


@dataclass(frozen=True, eq=False)
class RadialRegion(Region):
    """A circular region with full detail inside the radius."""

    center: np.ndarray
    radius: float
    falloff: float

    def evaluate(self, volume: BoundingVolume) -> float:
        distance = volume.signed_distance(self.center)
        falloff_blend = self._falloff_blend(distance, self.radius, self.falloff)
        return falloff_blend


@dataclass(frozen=True, eq=False)
class RectangularRegion(Region):
    """A rectangle, aligned to east/north, with full detail inside it."""

    center: np.ndarray
    east: np.ndarray
    north: np.ndarray
    width: float
    height: float
    falloff: float

    def evaluate(self, volume: BoundingVolume) -> float:
        x0, x1, y0, y1 = self._extent(volume)
        half_width = self.width * 0.5
        half_height = self.height * 0.5
        gap_x = max(-half_width - x1, x0 - half_width, 0.0)
        gap_y = max(-half_height - y1, y0 - half_height, 0.0)
        falloff_blend = self._falloff_blend(max(gap_x, gap_y), 0.0, self.falloff)
        return falloff_blend

    def _extent(self, volume: BoundingVolume) -> tuple[float, float, float, float]:
        """Return the tile's bounds on the tangent plane as x0, x1, y0, y1."""

        b = volume.box
        delta = np.array(b[0:3]) - self.center
        cx = float(delta @ self.east)
        cy = float(delta @ self.north)
        axis = np.array([b[3:6], b[6:9], b[9:12]])
        ex = float(np.abs(axis @ self.east).sum())
        ey = float(np.abs(axis @ self.north).sum())
        return cx - ex, cx + ex, cy - ey, cy + ey
