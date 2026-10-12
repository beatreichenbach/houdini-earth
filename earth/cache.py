"""Disk cache for downloaded 3D Tiles content."""

from __future__ import annotations

import contextlib
import hashlib
import os
from pathlib import Path


def hash_filename(filename: str) -> str:
    """Return a sha256-hashed filename, keeping the original extension."""

    name, ext = os.path.splitext(filename)
    return hashlib.sha256(name.encode()).hexdigest() + ext


class Cache:
    """Store downloaded tiles on disk under hashed filenames."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)

    def file(self, url: str) -> Path:
        """Return the cache path for a tile URL."""

        return self.path / hash_filename(url.split('?')[0].split('/')[-1])

    def read(self, url: str) -> bytes | None:
        """Return the cached bytes for a URL, or None if not cached."""

        path = self.file(url)
        if path.is_file():
            return path.read_bytes()
        return None

    def write(self, url: str, data: bytes) -> Path:
        """Write bytes for a URL into the cache and return the path."""

        path = self.file(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def clear(self) -> None:
        """Delete all cached tiles."""

        for pattern in ('*.json', '*.jpg', '*.glb'):
            for path in self.path.glob(pattern):
                with contextlib.suppress(PermissionError):
                    path.unlink(missing_ok=True)
