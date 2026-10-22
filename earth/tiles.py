"""Fetch and select Google 3D Tiles for a location."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests

from .cache import Cache
from .region import Region
from .tileset import Refine, Tile, parse_tile

logger = logging.getLogger(__name__)

TILE_URL = 'https://tile.googleapis.com'
ROOT_URI = '/v1/3dtiles/root.json'

Progress = Callable[[float, str], None]


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
    tile_id: str | None = None


@dataclass
class TileResult:
    """A downloaded mesh with its geometric error."""

    mesh: str
    error: float
    tile_id: str | None = None


@dataclass
class TileManifest:
    """The tiles produced by a selection."""

    tiles: list[TileResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class TileLoader:
    """Traverse a tileset, select tiles by distance and download them."""

    def __init__(
        self,
        api_key: str,
        cache: Cache,
        timeout: float = 30.0,
        progress: Progress | None = None,
    ) -> None:
        self.api_key = api_key
        self.cache = cache
        self.timeout = timeout
        self.progress = progress
        self.context = ''
        self.warnings: list[str] = []
        self._session = requests.Session()

    def root(self) -> dict:
        """Return the root tileset JSON, using the cache when possible."""

        # Cached root
        root_file = self.cache.path / 'root.json'
        if root_file.is_file():
            cached_root = json.loads(root_file.read_text())
            context = self._extract_context(cached_root['root'])
            if context:
                uri = self._find_key(cached_root['root'], 'uri').split('?')[0]
                if self._request_ok(self._content_url(uri, context)):
                    logger.debug('Reusing cached root tileset (context %s)', context)
                    self.context = context
                    return cached_root

            logger.debug('Cached root tileset is stale, refetching')
            self.cache.clear_metadata()

        # Fresh root
        logger.debug('Fetching root tileset')
        data = self._download(f'{TILE_URL}{ROOT_URI}?key={self.api_key}')
        if data is None:
            raise RuntimeError('could not fetch the 3D Tiles root')

        root = json.loads(data)
        root_file.parent.mkdir(parents=True, exist_ok=True)
        root_file.write_bytes(data)
        self.context = self._extract_context(root['root'])
        return root

    def collect(
        self,
        region: Region,
        min_error: float = 2.0,
        max_error: float = 60000,
        max_meshes: int = 10000,
        pack_error: float = 0.0,
        refine: Refine = Refine.REPLACE,
    ) -> TileManifest:
        """Return the selected tiles, nearest first, downloaded up to max_meshes."""

        self.warnings = []
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
            pack_error,
            selections,
        )
        selections.sort(key=lambda s: s.blend)
        if max_meshes > 0:
            selections = selections[:max_meshes]
        logger.debug('Selected %d tile(s)', len(selections))

        # Download meshes
        manifest = TileManifest()
        total = len(selections)
        for index, selection in enumerate(selections):
            url = self._content_url(selection.uri)
            if self._download(url) is None:
                continue

            tile = TileResult(
                mesh=str(self.cache.file(url)),
                error=selection.error,
                tile_id=selection.tile_id,
            )
            manifest.tiles.append(tile)
            if self.progress is not None:
                progress = (index + 1) / total
                self.progress(progress, f'Downloading mesh {index + 1}/{total}')

        logger.debug('Downloaded %d of %d tile(s)', len(manifest.tiles), total)
        manifest.warnings = list(self.warnings)
        return manifest

    def clear(self) -> None:
        """Delete all cached tiles."""

        self.cache.clear()

    def _content_url(self, uri: str, context: str | None = None) -> str:
        """Return the absolute URL for a tileset content URI."""

        context = self.context if context is None else context
        parts = urlsplit(f'{TILE_URL}{uri}')
        params = dict(parse_qsl(parts.query))
        for name, value in parse_qsl(context):
            params.setdefault(name, value)
        params['key'] = self.api_key
        components = (
            parts.scheme,
            parts.netloc,
            parts.path,
            urlencode(params),
            parts.fragment,
        )
        url = urlunsplit(components)
        return url

    def _download_tile(
        self,
        tile: Tile,
        refine: Refine,
        region: Region,
        min_error: float,
        max_error: float,
        pack_error: float,
        selections: list[TileSelection],
        tile_id: str | None = None,
    ) -> None:
        """Collect the mesh tiles below a node that fall within the region."""

        # Prune tiles without a bounding volume
        if tile.bounding_volume is None:
            logger.warning(
                'Pruning tile without a bounding volume: %s',
                tile.content_uri or '<no content>',
            )
            return

        blend = region.evaluate(tile.bounding_volume)
        if blend < 0:
            return

        if tile_id is None and pack_error > 0 and tile.geometric_error <= pack_error:
            tile_id = tile_key(tile)

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
                        pack_error,
                        selections,
                        tile_id,
                    )

            else:
                selected_tile_id = tile_id
                if selected_tile_id is None and pack_error > 0:
                    selected_tile_id = tile_key(tile)

                selection = TileSelection(
                    tile.content_uri,
                    tile.geometric_error,
                    blend,
                    selected_tile_id,
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
                pack_error,
                selections,
                tile_id,
            )

    def _download(self, url: str) -> bytes | None:
        """Return the bytes for a URL, reading and writing the cache."""

        data = self.cache.read(url)
        if data is not None:
            return data

        if self.progress is not None:
            # Poll for interruption without showing progress.
            self.progress(-1.0, 'Fetching tiles')

        try:
            response = self._session.get(url, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as error:
            path = urlsplit(url).path
            self.warnings.append(f'Request error for {path}: {type(error).__name__}')
            return None

        self.cache.write(url, response.content)
        return response.content

    def _request_ok(self, url: str) -> bool:
        """Return whether a URL can be fetched successfully."""

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

    @staticmethod
    def _extract_context(node: dict) -> str:
        """Return the session context embedded in a tileset's content URIs."""

        uri = TileLoader._find_key(node, 'uri')
        if not uri or '?' not in uri:
            return ''
        return uri.split('?')[-1]
