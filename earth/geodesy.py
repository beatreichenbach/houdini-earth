"""WGS84 to local tangent frame conversions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

A = 6378137.0
B = 6356752.3
EPS = 0.01


def wgs84_to_mesh(
    latitude: float, longitude: float, altitude: float = 0.0
) -> np.ndarray:
    """Return the WGS84 position in the 3D Tiles mesh frame as [x, z, -y]."""

    # The mesh frame is [X, Z, -Y] of standard ECEF coordinates.
    e2 = 1.0 - (B * B) / (A * A)
    lat = np.radians(latitude)
    lon = np.radians(-longitude)
    n = A / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    x = (n + altitude) * np.cos(lat) * np.cos(lon)
    y = (B * B / (A * A) * n + altitude) * np.sin(lat)
    z = (n + altitude) * np.cos(lat) * np.sin(lon)
    return np.array([x, y, z])


@dataclass(frozen=True, eq=False)
class Frame:
    """The east/south/up tangent frame at a latitude and longitude.

    The transform maps local +X to east, +Y to up and +Z to south.
    """

    center: np.ndarray
    east: np.ndarray
    south: np.ndarray
    up: np.ndarray
    transform: np.ndarray


def tangent_frame(latitude: float, longitude: float) -> Frame:
    """Return the local tangent frame for a latitude and longitude."""

    center = wgs84_to_mesh(latitude, longitude)

    top = wgs84_to_mesh(latitude + EPS, longitude)
    bottom = wgs84_to_mesh(latitude - EPS, longitude)
    left = wgs84_to_mesh(latitude, longitude + EPS)
    right = wgs84_to_mesh(latitude, longitude - EPS)
    south = bottom - top
    east = left - right

    south = south / np.linalg.norm(south)
    east = east / np.linalg.norm(east)
    up = np.cross(south, east)
    transform = np.array(
        [
            [east[0], east[1], east[2], 0.0],
            [up[0], up[1], up[2], 0.0],
            [south[0], south[1], south[2], 0.0],
            [center[0], center[1], center[2], 1.0],
        ]
    )
    return Frame(center=center, east=east, south=south, up=up, transform=transform)


def tangent_offset(frame: Frame, east: float, south: float) -> np.ndarray:
    """Return the mesh-frame point offset from a tangent frame by east and south."""

    return frame.center + frame.east * east + frame.south * south
