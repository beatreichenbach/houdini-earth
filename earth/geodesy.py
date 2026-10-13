"""WGS84 to local tangent frame conversions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

A = 6378137.0
B = 6356752.3
EPS = 0.01


def wgs84_to_cartesian(
    latitude: float, longitude: float, altitude: float = 0.0
) -> np.ndarray:
    """Return the WGS84 ECEF position for a latitude and longitude."""

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
    """The east/north/up tangent frame at a latitude and longitude."""

    center: np.ndarray
    east: np.ndarray
    north: np.ndarray
    up: np.ndarray
    transform: np.ndarray


def tangent_frame(latitude: float, longitude: float) -> Frame:
    """Return the local tangent frame for a latitude and longitude."""

    center = wgs84_to_cartesian(latitude, longitude)

    top = wgs84_to_cartesian(latitude + EPS, longitude)
    bottom = wgs84_to_cartesian(latitude - EPS, longitude)
    left = wgs84_to_cartesian(latitude, longitude + EPS)
    right = wgs84_to_cartesian(latitude, longitude - EPS)
    north = top - bottom
    east = left - right

    north = north / np.linalg.norm(north)
    east = east / np.linalg.norm(east)
    up = np.cross(east, north)
    transform = np.array(
        [
            [north[0], north[1], north[2], 0.0],
            [up[0], up[1], up[2], 0.0],
            [east[0], east[1], east[2], 0.0],
            [center[0], center[1], center[2], 1.0],
        ]
    )
    return Frame(center=center, east=east, north=north, up=up, transform=transform)
