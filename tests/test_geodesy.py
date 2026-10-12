import numpy as np
from numpy.testing import assert_allclose

from earth.geodesy import tangent_frame, wgs84_to_cartesian


def test_wgs84_position_is_near_earth_radius() -> None:
    center = wgs84_to_cartesian(45.0, 7.0)
    assert 6.35e6 < np.linalg.norm(center) < 6.40e6


def test_tangent_frame_is_orthonormal() -> None:
    frame = tangent_frame(45.0, 7.0)
    basis = np.array([frame.north, frame.up, frame.east])
    assert_allclose(basis @ basis.T, np.eye(3), atol=1e-6)


def test_transform_maps_local_origin_to_center() -> None:
    frame = tangent_frame(45.0, 7.0)
    inverse = np.linalg.inv(frame.transform)
    local = np.append(frame.center, 1.0) @ inverse
    assert_allclose(local, [0.0, 0.0, 0.0, 1.0], atol=1e-3)
