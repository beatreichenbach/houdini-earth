import json
from pathlib import Path

import numpy as np
from pytest import MonkeyPatch

from earth import tiles as tiles_module
from earth.cache import Cache
from earth.region import RadialRegion
from earth.tiles import TileLoader, tile_key
from earth.tileset import Tile


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


def _group(center_x: float, error: float, uri: str, *children: dict) -> dict:
    return {
        'boundingVolume': {
            'box': [center_x, 0, 0, 100, 0, 0, 0, 100, 0, 0, 0, 100],
        },
        'geometricError': error,
        'content': {'uri': uri},
        'children': list(children),
    }


def _install(monkeypatch: MonkeyPatch, root: dict) -> None:
    def get(session: object, url: str, timeout: float | None = None) -> MockResponse:
        payload = json.dumps(root).encode() if 'root.json' in url else b'glb'
        return MockResponse(payload)

    monkeypatch.setattr(tiles_module.requests.Session, 'get', get)


def test_collect_selects_and_downloads(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    _install(monkeypatch, _root(_leaf(0.0, 10.0, '/tile.glb?session=abc')))
    loader = TileLoader('key', Cache(tmp_path))
    region = RadialRegion(np.array([0.0, 0.0, 0.0]), radius=1000.0, falloff=1000.0)
    manifest = loader.collect(
        region,
        min_error=1.0,
        max_error=1000.0,
        max_meshes=10,
    )
    assert [tile.error for tile in manifest.tiles] == [10.0]
    assert len(manifest.tiles) == 1
    assert Path(manifest.tiles[0].mesh).is_file()


def test_collect_prunes_beyond_falloff(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    _install(monkeypatch, _root(_leaf(0.0, 10.0, '/tile.glb?session=abc')))
    loader = TileLoader('key', Cache(tmp_path))
    region = RadialRegion(np.array([100000.0, 0.0, 0.0]), radius=10.0, falloff=10.0)
    manifest = loader.collect(
        region,
        min_error=1.0,
        max_error=1000.0,
        max_meshes=10,
    )
    assert manifest.tiles == []


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
    loader = TileLoader('key', Cache(tmp_path))
    region = RadialRegion(np.array([0.0, 0.0, 0.0]), radius=0.0, falloff=10000.0)
    manifest = loader.collect(
        region,
        min_error=1.0,
        max_error=1000.0,
        max_meshes=1,
    )
    assert [tile.error for tile in manifest.tiles] == [10.0]


def test_content_url_keeps_existing_session(tmp_path: Path) -> None:
    loader = TileLoader('api-key', Cache(tmp_path))
    loader.context = 'session=abc'
    url = loader._content_url('/v1/3dtiles/files/tile.json?session=abc')
    assert url.count('?') == 1
    assert url.count('session=') == 1
    assert 'key=api-key' in url


def test_content_url_adds_context_when_missing(tmp_path: Path) -> None:
    loader = TileLoader('api-key', Cache(tmp_path))
    loader.context = 'session=abc'
    url = loader._content_url('/v1/3dtiles/files/tile.glb')
    assert url == (
        'https://tile.googleapis.com/v1/3dtiles/files/tile.glb?session=abc&key=api-key'
    )


def test_collect_assigns_and_inherits_pack_id(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    _install(
        monkeypatch,
        _root(
            _group(
                0.0,
                100.0,
                '/group-a.glb?session=abc',
                _leaf(0.0, 10.0, '/a1.glb?session=abc'),
                _leaf(0.0, 10.0, '/a2.glb?session=abc'),
            ),
            _group(
                100.0,
                100.0,
                '/group-b.glb?session=abc',
                _leaf(100.0, 10.0, '/b1.glb?session=abc'),
                _leaf(100.0, 10.0, '/b2.glb?session=abc'),
            ),
        ),
    )
    loader = TileLoader('key', Cache(tmp_path))
    region = RadialRegion(np.array([0.0, 0.0, 0.0]), radius=1000.0, falloff=1000.0)
    manifest = loader.collect(
        region,
        min_error=1.0,
        max_error=1000.0,
        max_meshes=10,
        pack_error=100.0,
    )
    pack_ids = [tile.tile_id for tile in manifest.tiles]
    assert len(pack_ids) == 4
    assert None not in pack_ids
    assert len(set(pack_ids)) == 2


def test_collect_falls_back_to_tile_id_without_pack_root(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    _install(
        monkeypatch,
        _root(
            _leaf(0.0, 10.0, '/a.glb?session=abc'),
            _leaf(0.0, 20.0, '/b.glb?session=abc'),
        ),
    )
    loader = TileLoader('key', Cache(tmp_path))
    region = RadialRegion(np.array([0.0, 0.0, 0.0]), radius=1000.0, falloff=1000.0)
    manifest = loader.collect(
        region,
        min_error=1.0,
        max_error=1000.0,
        max_meshes=10,
        pack_error=1.0,
    )
    pack_ids = [tile.tile_id for tile in manifest.tiles]
    assert len(pack_ids) == 2
    assert None not in pack_ids
    assert len(set(pack_ids)) == 2


def test_collect_zero_pack_error_disables_pack_id(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    _install(monkeypatch, _root(_leaf(0.0, 10.0, '/tile.glb?session=abc')))
    loader = TileLoader('key', Cache(tmp_path))
    region = RadialRegion(np.array([0.0, 0.0, 0.0]), radius=1000.0, falloff=1000.0)
    manifest = loader.collect(
        region,
        min_error=1.0,
        max_error=1000.0,
        max_meshes=10,
        pack_error=0.0,
    )
    assert manifest.tiles[0].tile_id is None


def test_download_warning_omits_api_key(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    def get(session: object, url: str, timeout: float | None = None) -> MockResponse:
        raise tiles_module.requests.ConnectionError(f'failed {url}')

    monkeypatch.setattr(tiles_module.requests.Session, 'get', get)
    loader = TileLoader('secret-key', Cache(tmp_path))
    loader._download('https://tile.googleapis.com/tile.glb?key=secret-key')
    assert loader.warnings
    assert 'secret-key' not in loader.warnings[0]


def test_tile_key_ignores_session() -> None:
    first = Tile(geometric_error=10.0, content_uri='/tile.glb?session=first')
    second = Tile(geometric_error=10.0, content_uri='/tile.glb?session=second')
    assert tile_key(first) == tile_key(second)
    assert tile_key(first) != tile_key(
        Tile(geometric_error=10.0, content_uri='/other.glb')
    )
