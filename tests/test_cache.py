from pathlib import Path

from earth.cache import Cache


def test_cache_write_read_roundtrip(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    url = 'https://tile.googleapis.com/tile.glb?session=abc'
    cache.write(url, b'data')
    assert cache.read(url) == b'data'


def test_cache_read_missing_returns_none(tmp_path: Path) -> None:
    assert Cache(tmp_path).read('https://x/missing.glb') is None
