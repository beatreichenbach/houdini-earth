"""Fetch and select Google 3D Tiles for a location."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import requests

from .cache import Cache
from .tileset import Refine, Tile, parse_tile

TILE_URL = 'https://tile.googleapis.com'
ROOT_URI = '/v1/3dtiles/root.json'

Progress = Callable[[float, str], None]


@dataclass
class TileSelection:
    """A downloaded mesh candidate."""

    uri: str
    error: float
    box: tuple[float, ...]
    distance: float


@dataclass
class TileManifest:
    """The mesh, bounding box and error lists produced by a selection."""

    meshes: list[str] = field(default_factory=list)
    boxes: list[str] = field(default_factory=list)
    errors: list[float] = field(default_factory=list)


def _desired_error(
    distance: float,
    min_dist: float,
    max_dist: float,
    min_error: float,
    max_error: float,
) -> float:
    """Return the LOD error allowed at a distance from the center."""

    if max_dist <= min_dist:
        return min_error
    t = (distance - min_dist) / (max_dist - min_dist)
    desired = min_error + t * (max_error - min_error)
    return min(max(desired, min_error), max_error)


class TileLoader:
    """Traverse a tileset, select tiles by distance and download them."""

    def __init__(
        self,
        api_key: str,
        cache: Cache,
        cached: bool = True,
        timeout: float = 30.0,
        progress: Progress | None = None,
    ) -> None:
        self.api_key = api_key
        self.cache = cache
        self.cached = cached
        self.timeout = timeout
        self.progress = progress
        self.context = ''

    def _content_url(self, uri: str, context: str | None = None) -> str:
        """Return the absolute URL for a tileset content URI."""

        context = self.context if context is None else context
        return f'{TILE_URL}{uri}?{context}&key={self.api_key}'

    def _download(self, url: str, cached: bool) -> bytes | None:
        """Return the bytes for a URL, reading and writing the cache."""

        if cached:
            data = self.cache.read(url)
            if data is not None:
                return data
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as error:
            print(f'Request error for {url}: {error}')
            return None
        self.cache.write(url, response.content)
        return response.content

    def _request_ok(self, url: str) -> bool:
        """Return whether a URL can be fetched successfully."""

        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException:
            return False
        return True

    @staticmethod
    def _find_key(node: dict, key: str) -> str:
        """Return the first key found in a node's content."""

        content = node.get('content', {})
        if key in content:
            return str(content[key])
        for child in node.get('children', []):
            found = TileLoader._find_key(child, key)
            if found:
                return found
        return ''

    @staticmethod
    def _extract_context(node: dict) -> str:
        """Return the session context embedded in a tileset's content URIs."""

        uri = TileLoader._find_key(node, 'uri')
        if not uri or '?' not in uri:
            return ''
        return uri.split('?')[-1]

    def root(self) -> dict:
        """Return the root tileset JSON, using the cache when possible."""

        root_file = self.cache.path / 'root.json'
        if self.cached and root_file.is_file():
            cached_root = json.loads(root_file.read_text())
            context = self._extract_context(cached_root['root'])
            if context:
                uri = self._find_key(cached_root['root'], 'uri').split('?')[0]
                if self._request_ok(self._content_url(uri, context)):
                    self.context = context
                    return cached_root
            self.cache.clear()

        data = self._download(f'{TILE_URL}{ROOT_URI}?key={self.api_key}', False)
        if data is None:
            raise RuntimeError('Could not fetch the 3D Tiles root.')
        root = json.loads(data)
        root_file.parent.mkdir(parents=True, exist_ok=True)
        root_file.write_bytes(data)
        self.context = self._extract_context(root['root'])
        return root

    def _visit(
        self,
        tile: Tile,
        refine: Refine,
        center: np.ndarray,
        min_error: float,
        max_error: float,
        min_dist: float,
        max_dist: float,
        selections: list[TileSelection],
    ) -> None:
        """Collect the mesh tiles below a node that fall within max_dist."""

        distance = 0.0
        if tile.bounding_volume is not None:
            distance = tile.bounding_volume.signed_distance(center)
            if distance > max_dist:
                return

        active_refine = tile.refine or refine
        desired = _desired_error(distance, min_dist, max_dist, min_error, max_error)
        selected = desired > tile.geometric_error or not tile.children

        if selected and tile.content_uri is not None:
            if tile.is_external_tileset:
                data = self._download(self._content_url(tile.content_uri), self.cached)
                if data is not None:
                    external = json.loads(data)
                    self._visit(
                        parse_tile(external['root']),
                        active_refine,
                        center,
                        min_error,
                        max_error,
                        min_dist,
                        max_dist,
                        selections,
                    )
            else:
                box = tile.bounding_volume.box if tile.bounding_volume else ()
                selections.append(
                    TileSelection(tile.content_uri, tile.geometric_error, box, distance)
                )
            if active_refine == Refine.REPLACE:
                return

        for child in tile.children:
            self._visit(
                child,
                active_refine,
                center,
                min_error,
                max_error,
                min_dist,
                max_dist,
                selections,
            )

    def collect(
        self,
        center: np.ndarray,
        min_error: float,
        max_error: float,
        min_dist: float,
        max_dist: float,
        max_meshes: int,
        refine: Refine = Refine.REPLACE,
    ) -> TileManifest:
        """Return the selected tiles, nearest first, downloaded up to max_meshes."""

        root = self.root()
        initial_refine = Refine(root['refine']) if root.get('refine') else refine
        selections: list[TileSelection] = []
        self._visit(
            parse_tile(root['root']),
            initial_refine,
            center,
            min_error,
            max_error,
            min_dist,
            max_dist,
            selections,
        )
        selections.sort(key=lambda s: s.distance)
        if max_meshes > 0:
            selections = selections[:max_meshes]

        manifest = TileManifest()
        total = len(selections)
        for index, selection in enumerate(selections):
            url = self._content_url(selection.uri)
            if self._download(url, self.cached) is None:
                continue
            manifest.meshes.append(str(self.cache.file(url)))
            manifest.boxes.append(','.join(str(x) for x in selection.box))
            manifest.errors.append(selection.error)
            if self.progress is not None:
                self.progress(
                    (index + 1) / total, f'Downloading mesh {index + 1}/{total}'
                )
        return manifest

    def clear(self) -> None:
        """Delete all cached tiles."""

        self.cache.clear()
