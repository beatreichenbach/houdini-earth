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
    def _falloff_blend(
        distance: float, radius: float, falloff: float, exponent: float = 1.0
    ) -> float:
        if falloff <= 0.0:
            return 0.0 if distance <= radius else -1.0
        if distance > radius + falloff:
            return -1.0
        blend = (distance - radius) / falloff
        blend = min(max(blend, 0.0), 1.0)
        if exponent <= 0.0:
            return 0.0 if blend <= 0.0 else 1.0
        return blend**exponent


@dataclass(frozen=True, eq=False)
class RadialRegion(Region):
    """A circular region with full detail inside the radius."""

    center: np.ndarray
    radius: float
    falloff: float
    exponent: float = 1.0

    def evaluate(self, volume: BoundingVolume) -> float:
        distance = volume.signed_distance(self.center)
        falloff_blend = self._falloff_blend(
            distance, self.radius, self.falloff, self.exponent
        )
        return falloff_blend


@dataclass(frozen=True, eq=False)
class RectangularRegion(Region):
    """A rectangle, aligned to east/south, with full detail inside it."""

    center: np.ndarray
    east: np.ndarray
    south: np.ndarray
    width: float
    height: float
    falloff: float
    exponent: float = 1.0

    def evaluate(self, volume: BoundingVolume) -> float:
        half_width = self.width * 0.5
        half_height = self.height * 0.5

        # Reject tiles outside the expanded rectangle's circumscribed circle.
        # The exact rectangle test below still decides which remaining tiles
        # are kept.
        falloff = max(self.falloff, 0.0)
        broad_phase_radius = float(
            np.hypot(half_width + falloff, half_height + falloff)
        )
        if volume.signed_distance(self.center) > broad_phase_radius:
            return -1.0

        x0, x1, y0, y1 = self._extent(volume)
        gap_x = self._axis_gap(x0, x1, half_width)
        gap_y = self._axis_gap(y0, y1, half_height)
        falloff_blend = self._falloff_blend(
            max(gap_x, gap_y), 0.0, self.falloff, self.exponent
        )
        return falloff_blend

    def _extent(self, volume: BoundingVolume) -> tuple[float, float, float, float]:
        """Return the tile's east/south bounds as x0, x1, y0, y1."""

        delta = volume.center - self.center
        cx = float(delta @ self.east)
        cy = float(delta @ self.south)
        axes = volume.axes
        ex = float(np.abs(axes @ self.east).sum())
        ey = float(np.abs(axes @ self.south).sum())
        return cx - ex, cx + ex, cy - ey, cy + ey

    @staticmethod
    def _axis_gap(lo: float, hi: float, half_extent: float) -> float:
        """Return the gap from a box extent to a centered half-extent."""

        return max(-half_extent - hi, lo - half_extent, 0.0)
