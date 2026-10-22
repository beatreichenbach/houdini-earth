import numpy as np
from numpy.testing import assert_allclose

from earth.geodesy import tangent_frame, tangent_offset, wgs84_to_mesh


def test_wgs84_position_is_near_earth_radius() -> None:
    center = wgs84_to_mesh(45.0, 7.0)
    assert 6.35e6 < np.linalg.norm(center) < 6.40e6


def test_tangent_frame_is_orthonormal() -> None:
    frame = tangent_frame(45.0, 7.0)
    basis = np.array([frame.east, frame.up, frame.south])
    assert_allclose(basis @ basis.T, np.eye(3), atol=1e-6)


def test_transform_maps_local_origin_to_center() -> None:
    frame = tangent_frame(45.0, 7.0)
    inverse = np.linalg.inv(frame.transform)
    local = np.append(frame.center, 1.0) @ inverse
    assert_allclose(local, [0.0, 0.0, 0.0, 1.0], atol=1e-3)


def test_transform_orientation() -> None:
    frame = tangent_frame(45.0, 7.0)
    x = np.array([1.0, 0.0, 0.0, 0.0]) @ frame.transform
    y = np.array([0.0, 1.0, 0.0, 0.0]) @ frame.transform
    z = np.array([0.0, 0.0, 1.0, 0.0]) @ frame.transform
    assert_allclose(x[:3], frame.east, atol=1e-6)
    assert_allclose(y[:3], frame.up, atol=1e-6)
    assert_allclose(z[:3], frame.south, atol=1e-6)


def test_tangent_offset_moves_east_and_south() -> None:
    frame = tangent_frame(45.0, 7.0)
    delta = tangent_offset(frame, 250.0, 120.0) - frame.center
    assert_allclose(delta @ frame.east, 250.0, atol=1e-6)
    assert_allclose(delta @ frame.south, 120.0, atol=1e-6)
    assert_allclose(delta @ frame.up, 0.0, atol=1e-6)
