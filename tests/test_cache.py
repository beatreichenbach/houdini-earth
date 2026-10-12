from pathlib import Path

from earth.cache import Cache, hash_filename


def test_hash_filename_keeps_extension() -> None:
    hashed = hash_filename('a-very-long-tile-name.glb')
    assert hashed.endswith('.glb')
    assert len(hashed) == 64 + 4


def test_cache_write_read_roundtrip(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    url = 'https://tile.googleapis.com/tile.glb?session=abc'
    cache.write(url, b'data')
    assert cache.read(url) == b'data'


def test_cache_read_missing_returns_none(tmp_path: Path) -> None:
    assert Cache(tmp_path).read('https://x/missing.glb') is None


def test_cache_clear(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    cache.write('https://x/a.glb', b'data')
    cache.write('https://x/b.json', b'{}')
    cache.clear()
    assert list(tmp_path.glob('*')) == []
