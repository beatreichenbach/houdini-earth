"""Pack per-tile textures into atlas pages and remap their UVs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CELL = 256


@dataclass(frozen=True)
class Placement:
    """The rectangle a texture occupies in an atlas page."""

    texture: str
    page: int
    x: int
    y: int
    width: int
    height: int


@dataclass
class AtlasPage:
    """An atlas page and the placements it contains."""

    index: int
    size: int
    placements: list[Placement] = field(default_factory=list)


def plan_pages(
    sizes: dict[str, tuple[int, int]],
    resolution: int,
    padding: int = 0,
) -> list[AtlasPage]:
    """Return atlas pages that shelf-pack the textures into 256px cells."""

    columns = resolution // CELL
    size = columns * CELL
    items = sorted(sizes.items(), key=lambda item: (-item[1][1], -item[1][0], item[0]))

    atlas_pages: list[AtlasPage] = []
    placements: list[Placement] = []
    page = 0
    column = 0
    shelf = 0
    shelf_rows = 0

    for texture, (width, height) in items:
        cols = _cells(width + 2 * padding)
        rows = _cells(height + 2 * padding)
        if cols > columns or rows > columns:
            raise ValueError(f'{texture} does not fit in a {resolution}px atlas')

        if column + cols > columns:
            shelf += shelf_rows
            column = 0
            shelf_rows = 0

        if shelf + rows > columns:
            atlas_page = AtlasPage(page, size, placements)
            atlas_pages.append(atlas_page)
            page += 1
            placements = []
            column = 0
            shelf = 0
            shelf_rows = 0

        placement = Placement(
            texture=texture,
            page=page,
            x=column * CELL + padding,
            y=shelf * CELL + padding,
            width=width,
            height=height,
        )
        placements.append(placement)
        column += cols
        shelf_rows = max(shelf_rows, rows)

    if placements:
        atlas_page = AtlasPage(page, size, placements)
        atlas_pages.append(atlas_page)
    return atlas_pages


def remap_uv(uv: np.ndarray, placement: Placement, resolution: int) -> np.ndarray:
    """Return UVs remapped from tile space into the placement's atlas rect."""

    left = placement.x + uv[:, 0] * placement.width
    u = left / resolution
    top = resolution - placement.y - placement.height + uv[:, 1] * placement.height
    v = top / resolution
    return np.stack((u, v), axis=1)


def _cells(size: int) -> int:
    """Return how many 256px cells a size needs."""

    return max(1, -(-size // CELL))
