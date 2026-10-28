"""Disk cache for downloaded 3D Tiles content."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlsplit


class Cache:
    """Store downloaded tiles on disk under their URL filenames."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)

    def file(self, url: str) -> Path:
        """Return the cache path for a tile URL."""

        filename = urlsplit(url).path.rsplit('/', 1)[-1]
        if not filename:
            raise ValueError(f'url has no filename: {url}')
        return self.path / filename

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
        """Delete cached tileset, mesh, and texture files."""

        remove_files(self.path, suffixes=('.json', '.glb', '.jpg'))

    def clear_metadata(self) -> None:
        """Delete cached tileset metadata, leaving geometry intact."""

        remove_files(self.path, suffixes=('.json',))


def remove_files(directory: os.PathLike | str, suffixes: Sequence[str] = ()) -> None:
    """Remove files from a directory, optionally limited to suffixes."""

    directory = Path(directory)

    if not directory.is_dir():
        return

    for path in directory.rglob('*'):
        if path.is_file() and (not suffixes or path.suffix.lower() in suffixes):
            with contextlib.suppress(PermissionError):
                path.unlink(missing_ok=True)
