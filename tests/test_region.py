import numpy as np
from numpy.testing import assert_allclose

from earth.geodesy import tangent_frame, tangent_offset
from earth.region import RadialRegion, RectangularRegion
from earth.tileset import BoundingVolume

EAST = np.array([1.0, 0.0, 0.0])
SOUTH = np.array([0.0, 0.0, 1.0])


def _cube(east: float = 0.0, south: float = 0.0, half: float = 1.0) -> BoundingVolume:
    return BoundingVolume((east, -south, 0.0, half, 0, 0, 0, half, 0, 0, 0, half))


def _volume_at(point: np.ndarray, half: float = 1.0) -> BoundingVolume:
    raw = np.array([point[0], -point[2], point[1]])
    return BoundingVolume((*raw, half, 0, 0, 0, half, 0, 0, 0, half))


def test_radial_inside_blend_zero() -> None:
    region = RadialRegion(np.zeros(3), radius=5.0, falloff=10.0)
    blend = region.evaluate(_cube())
    assert blend >= 0.0
    assert blend == 0.0


def test_radial_partial_blend() -> None:
    region = RadialRegion(np.array([10.0, 0.0, 0.0]), radius=5.0, falloff=10.0)
    blend = region.evaluate(_cube())
    assert blend >= 0.0
    assert_allclose(blend, 0.4)


def test_radial_prunes_outside() -> None:
    region = RadialRegion(np.array([100.0, 0.0, 0.0]), radius=5.0, falloff=10.0)
    assert region.evaluate(_cube()) < 0.0


def test_radial_zero_falloff_is_a_hard_cut() -> None:
    inside = RadialRegion(np.array([3.0, 0.0, 0.0]), radius=5.0, falloff=0.0)
    outside = RadialRegion(np.array([10.0, 0.0, 0.0]), radius=5.0, falloff=0.0)
    assert inside.evaluate(_cube()) >= 0.0
    assert outside.evaluate(_cube()) < 0.0


def test_radial_falloff_exponent() -> None:
    region = RadialRegion(
        np.array([10.0, 0.0, 0.0]),
        radius=5.0,
        falloff=10.0,
        exponent=2.0,
    )
    assert_allclose(region.evaluate(_cube()), 0.16)


def test_radial_zero_exponent_jumps_to_max() -> None:
    region = RadialRegion(
        np.array([10.0, 0.0, 0.0]),
        radius=5.0,
        falloff=10.0,
        exponent=0.0,
    )
    assert region.evaluate(_cube()) == 1.0


def test_rect_inside_blend_zero() -> None:
    region = RectangularRegion(
        np.zeros(3), EAST, SOUTH, width=10.0, height=10.0, falloff=2.0
    )
    blend = region.evaluate(_cube())
    assert blend >= 0.0
    assert blend == 0.0


def test_rect_width_falloff() -> None:
    region = RectangularRegion(
        np.zeros(3), EAST, SOUTH, width=10.0, height=10.0, falloff=4.0
    )
    blend = region.evaluate(_cube(east=8.0))
    assert blend >= 0.0
    assert_allclose(blend, 0.5)


def test_rect_height_falloff() -> None:
    region = RectangularRegion(
        np.zeros(3), EAST, SOUTH, width=10.0, height=4.0, falloff=2.0
    )
    blend = region.evaluate(_cube(south=4.0))
    assert blend >= 0.0
    assert_allclose(blend, 0.5)


def test_rect_prunes_outside_falloff() -> None:
    region = RectangularRegion(
        np.zeros(3), EAST, SOUTH, width=10.0, height=10.0, falloff=2.0
    )
    assert region.evaluate(_cube(east=9.0)) < 0.0


def test_rect_falloff_exponent() -> None:
    region = RectangularRegion(
        np.zeros(3),
        EAST,
        SOUTH,
        width=10.0,
        height=10.0,
        falloff=4.0,
        exponent=2.0,
    )
    assert_allclose(region.evaluate(_cube(east=8.0)), 0.25)


def test_rect_uses_tile_frame_box_center() -> None:
    frame = tangent_frame(40.7243887, -73.997701)
    center = tangent_offset(frame, 0.0, 0.0)
    volume = _volume_at(center)
    region = RectangularRegion(
        center, frame.east, frame.south, width=10.0, height=10.0, falloff=0.0
    )
    assert region.evaluate(volume) == 0.0


def test_rect_prunes_tile_far_above_plane() -> None:
    frame = tangent_frame(40.7243887, -73.997701)
    center = tangent_offset(frame, 0.0, 0.0)
    volume = _volume_at(center + frame.up * 1_000_000.0, half=100.0)
    region = RectangularRegion(
        center, frame.east, frame.south, width=10.0, height=10.0, falloff=0.0
    )
    assert region.evaluate(volume) < 0.0


def test_rect_broad_phase_keeps_corner_tile_inside_falloff() -> None:
    region = RectangularRegion(
        np.zeros(3), EAST, SOUTH, width=200.0, height=400.0, falloff=50.0
    )
    blend = region.evaluate(_cube(east=150.0, south=250.0))
    assert blend >= 0.0
    assert_allclose(blend, 0.98)


def test_rect_negative_falloff_is_a_hard_cut() -> None:
    region = RectangularRegion(
        np.zeros(3), EAST, SOUTH, width=10.0, height=10.0, falloff=-1.0
    )
    assert region.evaluate(_cube(east=4.0)) == 0.0
    assert region.evaluate(_cube(east=7.0)) < 0.0
