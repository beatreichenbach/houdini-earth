import json
from pathlib import Path

import numpy as np
from pytest import MonkeyPatch

from earth import tiles as tiles_module
from earth.cache import Cache
from earth.tiles import TileLoader, _desired_error


class MockResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.text = content.decode()

    def raise_for_status(self) -> None:
        return None


def _root(*children: dict) -> dict:
    return {
        'asset': {'version': '1.0'},
        'geometricError': 500.0,
        'refine': 'REPLACE',
        'root': {
            'boundingVolume': {
                'box': [0, 0, 0, 1000, 0, 0, 0, 1000, 0, 0, 0, 1000],
            },
            'geometricError': 500.0,
            'children': list(children),
        },
    }


def _leaf(center_x: float, error: float, uri: str) -> dict:
    return {
        'boundingVolume': {
            'box': [center_x, 0, 0, 100, 0, 0, 0, 100, 0, 0, 0, 100],
        },
        'geometricError': error,
        'content': {'uri': uri},
    }


def _install(monkeypatch: MonkeyPatch, root: dict) -> None:
    def get(url: str, timeout: float | None = None) -> MockResponse:
        payload = json.dumps(root).encode() if 'root.json' in url else b'glb'
        return MockResponse(payload)

    monkeypatch.setattr(tiles_module.requests, 'get', get)


def test_desired_error_clamps() -> None:
    assert _desired_error(0.0, 0.0, 100.0, 1.0, 50.0) == 1.0
    assert _desired_error(100.0, 0.0, 100.0, 1.0, 50.0) == 50.0
    assert _desired_error(500.0, 0.0, 100.0, 1.0, 50.0) == 50.0


def test_collect_selects_and_downloads(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    _install(monkeypatch, _root(_leaf(0.0, 10.0, '/tile.glb?session=abc')))
    loader = TileLoader('key', Cache(tmp_path), cached=True)
    manifest = loader.collect(
        np.array([0.0, 0.0, 0.0]),
        min_error=1.0,
        max_error=1000.0,
        min_dist=0.0,
        max_dist=10000.0,
        max_meshes=10,
    )
    assert manifest.errors == [10.0]
    assert len(manifest.meshes) == 1
    assert Path(manifest.meshes[0]).is_file()


def test_collect_prunes_beyond_max_dist(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    _install(monkeypatch, _root(_leaf(0.0, 10.0, '/tile.glb?session=abc')))
    loader = TileLoader('key', Cache(tmp_path), cached=True)
    manifest = loader.collect(
        np.array([100000.0, 0.0, 0.0]),
        min_error=1.0,
        max_error=1000.0,
        min_dist=0.0,
        max_dist=10.0,
        max_meshes=10,
    )
    assert manifest.meshes == []


def test_collect_sorts_nearest_first_and_applies_max_meshes(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    _install(
        monkeypatch,
        _root(
            _leaf(0.0, 10.0, '/near.glb?session=abc'),
            _leaf(500.0, 20.0, '/far.glb?session=abc'),
        ),
    )
    loader = TileLoader('key', Cache(tmp_path), cached=True)
    manifest = loader.collect(
        np.array([0.0, 0.0, 0.0]),
        min_error=1.0,
        max_error=1000.0,
        min_dist=0.0,
        max_dist=10000.0,
        max_meshes=1,
    )
    assert manifest.errors == [10.0]
