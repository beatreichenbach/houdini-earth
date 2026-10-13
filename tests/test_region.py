import numpy as np
from numpy.testing import assert_allclose

from earth.region import RadialRegion, RectangularRegion
from earth.tileset import BoundingVolume

EAST = np.array([1.0, 0.0, 0.0])
NORTH = np.array([0.0, 1.0, 0.0])


def _cube(cx: float = 0.0, cy: float = 0.0, half: float = 1.0) -> BoundingVolume:
    return BoundingVolume((cx, cy, 0.0, half, 0, 0, 0, half, 0, 0, 0, half))


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


def test_rect_inside_blend_zero() -> None:
    region = RectangularRegion(
        np.zeros(3), EAST, NORTH, width=10.0, height=10.0, falloff=2.0
    )
    blend = region.evaluate(_cube())
    assert blend >= 0.0
    assert blend == 0.0


def test_rect_width_falloff() -> None:
    region = RectangularRegion(
        np.zeros(3), EAST, NORTH, width=10.0, height=10.0, falloff=4.0
    )
    blend = region.evaluate(_cube(cx=8.0))
    assert blend >= 0.0
    assert_allclose(blend, 0.5)


def test_rect_height_falloff() -> None:
    region = RectangularRegion(
        np.zeros(3), EAST, NORTH, width=10.0, height=4.0, falloff=2.0
    )
    blend = region.evaluate(_cube(cy=4.0))
    assert blend >= 0.0
    assert_allclose(blend, 0.5)


def test_rect_prunes_outside_falloff() -> None:
    region = RectangularRegion(
        np.zeros(3), EAST, NORTH, width=10.0, height=10.0, falloff=2.0
    )
    assert region.evaluate(_cube(cx=9.0)) < 0.0
