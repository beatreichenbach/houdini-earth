import numpy as np

from earth.tileset import BoundingVolume, Refine, parse_tile


def _cube() -> BoundingVolume:
    return BoundingVolume((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))


def test_signed_distance_inside_is_non_positive() -> None:
    assert _cube().signed_distance(np.array([0.0, 0.0, 0.0])) <= 0.0


def test_signed_distance_outside_is_positive() -> None:
    assert _cube().signed_distance(np.array([10.0, 0.0, 0.0])) > 0.0


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
