"""Fetch and select Google 3D Tiles for a location."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from urllib.parse import parse_qsl, urlencode, urlsplit

import requests

from .cache import Cache
from .region import Region
from .tileset import Refine, Tile, parse_tile

logger = logging.getLogger(__name__)

TILE_URL = 'https://tile.googleapis.com'
ROOT_URI = '/v1/3dtiles/root.json'

Progress = Callable[[float], None]


def tile_key(tile: Tile) -> str:
    """Return a stable identifier for a tile."""

    if tile.content_uri is not None:
        key = tile.content_uri.split('?', 1)[0]
    else:
        box = tile.bounding_volume.box if tile.bounding_volume is not None else ()
        key = repr((tile.geometric_error, box))
    return hashlib.sha256(key.encode()).hexdigest()


@dataclass
class TileSelection:
    """A downloaded mesh candidate."""

    uri: str
    error: float
    blend: float
    attribute_ids: dict[str, str] = field(default_factory=dict)


@dataclass
class TileResult:
    """A downloaded mesh with its geometric error."""

    mesh: str
    error: float
    attribute_ids: dict[str, str] = field(default_factory=dict)


@dataclass
class TileManifest:
    """The tiles produced by a selection."""

    tiles: list[TileResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class TileLoader:
    """Traverse a tileset, select tiles by distance and download them."""

    def __init__(
        self,
        api_key: str,
        cache: Cache,
        timeout: float = 10.0,
        progress: Progress | None = None,
    ) -> None:
        self.api_key = api_key
        self.cache = cache
        self.timeout = timeout
        self.progress = progress
        self.context = ''
        self.errors: list[str] = []
        self._session = requests.Session()
        self._session.headers['X-Goog-Api-Key'] = api_key

    def root(self) -> dict:
        """Return the root tileset JSON, using the cache when possible."""

        # Cached root
        root_file = self.cache.path / 'root.json'
        if root_file.is_file():
            cached_root = json.loads(root_file.read_text())
            uri = TileLoader._find_key(cached_root['root'], 'uri')
            if uri:
                self.context = urlsplit(uri).query
                if self._request_ok(self._content_url(uri)):
                    return cached_root

            logger.debug('Cached root tileset is stale, redownloading')
            self.cache.clear_metadata()

        # Fresh root
        data = self._download(self._content_url(ROOT_URI))
        if data is None:
            raise RuntimeError('could not download the 3D Tiles root')

        root = json.loads(data)
        root_file.parent.mkdir(parents=True, exist_ok=True)
        root_file.write_bytes(data)

        uri = TileLoader._find_key(root['root'], 'uri')
        self.context = urlsplit(uri).query
        return root

    def collect(
        self,
        region: Region,
        min_error: float = 2.0,
        max_error: float = 60000,
        max_meshes: int = 10000,
        attributes: Sequence[tuple[str, float]] = (),
        refine: Refine = Refine.REPLACE,
    ) -> TileManifest:
        """Return the selected tiles, nearest first, downloaded up to max_meshes."""

        self.errors = []
        root = self.root()
        initial_refine = Refine(root['refine']) if root.get('refine') else refine

        # Select tiles
        selections: list[TileSelection] = []
        self._download_tile(
            parse_tile(root['root']),
            initial_refine,
            region,
            min_error,
            max_error,
            attributes,
            selections,
        )
        selections.sort(key=lambda s: s.blend)
        total = len(selections)
        logger.debug('Selected %d tile(s)', total)

        # Limit tiles
        if 0 < max_meshes < total:
            self.errors.append(f'Number of selected meshes exceeds limit: {total}')
            total = max_meshes
            selections = selections[:max_meshes]

        # Download meshes
        manifest = TileManifest()
        for index, selection in enumerate(selections):
            url = self._content_url(selection.uri)
            if self._download(url) is None:
                continue

            tile = TileResult(
                mesh=str(self.cache.file(url)),
                error=selection.error,
                attribute_ids=selection.attribute_ids,
            )
            manifest.tiles.append(tile)
            if self.progress is not None:
                progress = (index + 1) / total
                self.progress(progress)

        logger.debug('Downloaded %d tiles', total)
        manifest.errors = list(self.errors)
        return manifest

    def _content_url(self, uri: str) -> str:
        """Return the absolute URL for a tileset content URI."""

        url = f'{TILE_URL}{uri}'
        result = urlsplit(url)
        params = dict(parse_qsl(result.query))
        for name, value in parse_qsl(self.context):
            params.setdefault(name, value)
        content_url = result._replace(query=urlencode(params)).geturl()
        return content_url

    def _download_tile(
        self,
        tile: Tile,
        refine: Refine,
        region: Region,
        min_error: float,
        max_error: float,
        attributes: Sequence[tuple[str, float]],
        selections: list[TileSelection],
        attribute_ids: tuple[str | None, ...] | None = None,
    ) -> None:
        """Collect the mesh tiles below a node that fall within the region."""

        # Prune tiles without a bounding volume
        if tile.bounding_volume is None:
            logger.debug('Pruning tile without a bounding volume: %s', tile.content_uri)
            return

        blend = region.evaluate(tile.bounding_volume)
        if blend < 0:
            return

        if attribute_ids is None:
            attribute_ids = (None,) * len(attributes)

        key = tile_key(tile)
        ids = list(attribute_ids)
        for index, (_, error) in enumerate(attributes):
            if ids[index] is None and tile.geometric_error <= error:
                ids[index] = key

        active_refine = tile.refine or refine
        desired = min_error + blend * (max_error - min_error)
        selected = desired > tile.geometric_error or not tile.children

        # Add selected content
        if selected and tile.content_uri is not None:
            if tile.is_external_tileset:
                data = self._download(self._content_url(tile.content_uri))
                if data is not None:
                    external_root = json.loads(data)
                    external_tile = parse_tile(external_root['root'])
                    self._download_tile(
                        external_tile,
                        active_refine,
                        region,
                        min_error,
                        max_error,
                        attributes,
                        selections,
                        tuple(ids),
                    )

            else:
                selected_ids: dict[str, str] = {}
                for (name, _), group_id in zip(attributes, ids, strict=True):
                    selected_ids[name] = group_id if group_id is not None else key

                selection = TileSelection(
                    tile.content_uri,
                    tile.geometric_error,
                    blend,
                    selected_ids,
                )
                selections.append(selection)

            if active_refine == Refine.REPLACE:
                return

        # Descend into children
        for child in tile.children:
            self._download_tile(
                child,
                active_refine,
                region,
                min_error,
                max_error,
                attributes,
                selections,
                tuple(ids),
            )

    def _download(self, url: str) -> bytes | None:
        """Return the bytes for a URL, reading and writing the cache."""

        try:
            data = self.cache.read(url)
        except ValueError:
            return None
        if data is not None:
            return data

        if self.progress is not None:
            # Poll for interruption without showing progress.
            self.progress(-1.0)

        try:
            response = self._session.get(url, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as error:
            path = urlsplit(url).path
            self.errors.append(f'Request error for {path}: {type(error).__name__}')
            return None

        self.cache.write(url, response.content)
        return response.content

    def _request_ok(self, url: str) -> bool:
        """Return whether a URL can be downloaded successfully."""

        try:
            response = self._session.get(url, timeout=self.timeout)
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
