"""Pack per-tile textures into atlas pages and remap their UVs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CELL = 256


class AtlasError(ValueError):
    """A texture cannot be packed into an atlas."""


@dataclass(frozen=True)
class Placement:
    """The rectangle a texture occupies in an atlas page."""

    texture: str
    x: int
    y: int
    width: int
    height: int


@dataclass
class AtlasPage:
    """An atlas page and the placements it contains."""

    index: int
    width: int
    height: int
    placements: list[Placement] = field(default_factory=list)


def plan_pages(
    sizes: dict[str, tuple[int, int]],
    resolution: int,
    padding: int = 0,
) -> list[AtlasPage]:
    """Return atlas pages that shelf-pack the textures into 256px cells."""

    columns = max(1, resolution // CELL)

    # Sort textures by height
    items = sorted(sizes.items(), key=lambda item: (-item[1][1], -item[1][0], item[0]))

    atlas_pages: list[AtlasPage] = []
    placements: list[Placement] = []
    page = 0
    column = 0
    shelf = 0
    shelf_rows = 0
    page_width = 0
    page_height = 0

    for texture, (width, height) in items:
        cols = _cell_count(width + 2 * padding)
        rows = _cell_count(height + 2 * padding)
        if cols > columns or rows > columns:
            raise AtlasError(f'{texture} does not fit in a {resolution}px atlas')

        # New shelf
        if column + cols > columns:
            shelf += shelf_rows
            column = 0
            shelf_rows = 0

        # New page
        if shelf + rows > columns:
            atlas_page = AtlasPage(page, page_width, page_height, placements)
            atlas_pages.append(atlas_page)
            page += 1
            placements = []
            column = 0
            shelf = 0
            shelf_rows = 0
            page_width = 0
            page_height = 0

        placement = Placement(
            texture=texture,
            x=column * CELL + padding,
            y=shelf * CELL + padding,
            width=width,
            height=height,
        )
        placements.append(placement)
        column += cols
        shelf_rows = max(shelf_rows, rows)
        page_width = max(page_width, placement.x + placement.width + padding)
        page_height = max(page_height, placement.y + placement.height + padding)

    if placements:
        atlas_page = AtlasPage(page, page_width, page_height, placements)
        atlas_pages.append(atlas_page)
    return atlas_pages


def remap_uv(
    uv: np.ndarray, placement: Placement, page_width: int, page_height: int
) -> np.ndarray:
    """Return UVs remapped from tile space into the placement's atlas rect."""

    left = placement.x + uv[:, 0] * placement.width
    u = left / page_width
    top = page_height - placement.y - placement.height + uv[:, 1] * placement.height
    v = top / page_height
    remapped = np.stack((u, v), axis=1)
    return remapped


def _cell_count(size: int) -> int:
    """Return how many cells a size needs."""

    return max(1, -(-size // CELL))
