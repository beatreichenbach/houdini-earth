import numpy as np
from numpy.testing import assert_allclose

from earth.tileset import BoundingVolume, Refine, parse_tile


def _cube() -> BoundingVolume:
    return BoundingVolume((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))


def test_bounding_volume_center_and_axes_use_mesh_frame() -> None:
    volume = BoundingVolume((10, 20, 30, 1, 2, 3, 4, 5, 6, 7, 8, 9))
    assert_allclose(volume.center, [10.0, 30.0, -20.0])
    assert_allclose(
        volume.axes,
        [[1.0, 3.0, -2.0], [4.0, 6.0, -5.0], [7.0, 9.0, -8.0]],
    )


def test_signed_distance_inside_is_non_positive() -> None:
    assert _cube().signed_distance(np.array([0.0, 0.0, 0.0])) <= 0.0


def test_signed_distance_outside_is_positive() -> None:
    assert _cube().signed_distance(np.array([10.0, 0.0, 0.0])) > 0.0


def test_signed_distance_uses_mesh_frame_axes() -> None:
    volume = BoundingVolume((10, 20, 30, 2, 0, 0, 0, 3, 0, 0, 0, 4))
    assert_allclose(volume.signed_distance(np.array([10.0, 35.0, -20.0])), 1.0)


def test_parse_tile_reads_content_and_refine() -> None:
    tile = parse_tile(
        {
            'geometricError': 5.0,
            'refine': 'ADD',
            'boundingVolume': {'box': [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]},
            'content': {'uri': '/tiles/0.glb'},
        }
    )
    assert tile.geometric_error == 5.0
    assert tile.refine == Refine.ADD
    assert tile.bounding_volume is not None
    assert tile.content_uri == '/tiles/0.glb'
    assert not tile.is_external_tileset


def test_external_tileset_detection() -> None:
    tile = parse_tile({'content': {'uri': '/tiles/0.json?session=x'}})
    assert tile.is_external_tileset


def test_parse_tile_children() -> None:
    tile = parse_tile({'children': [{'geometricError': 1.0}]})
    assert len(tile.children) == 1
    assert tile.children[0].geometric_error == 1.0
