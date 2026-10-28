"""Pack per-tile textures into atlas pages and remap their UVs."""

from __future__ import annotations

from dataclasses import dataclass, field


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


@dataclass
class UVTransform:
    offset: tuple[float, float]
    scale: tuple[float, float]


def plan_pages(
    sizes: dict[str, tuple[int, int]],
    resolution: int,
    padding: int = 0,
) -> list[AtlasPage]:
    """Return atlas pages that shelf-pack the textures at pixel resolution."""

    items = sorted(sizes.items(), key=lambda item: (-item[1][1], -item[1][0], item[0]))

    atlas_pages: list[AtlasPage] = []
    placements: list[Placement] = []
    cursor_x = 0
    shelf_y = 0
    shelf_height = 0
    page_width = 0
    page_height = 0
    page_index = 0

    for texture, (width, height) in items:
        item_width = width + 2 * padding
        item_height = height + 2 * padding
        if item_width > resolution or item_height > resolution:
            raise AtlasError(f'{texture} does not fit in a {resolution}px atlas')

        if cursor_x + item_width > resolution:
            shelf_y += shelf_height
            cursor_x = 0
            shelf_height = 0

        if shelf_y + item_height > resolution:
            atlas_page = AtlasPage(page_index, page_width, page_height, placements)
            atlas_pages.append(atlas_page)
            placements = []
            cursor_x = 0
            shelf_y = 0
            shelf_height = 0
            page_width = 0
            page_height = 0
            page_index += 1

        placement = Placement(
            texture=texture,
            x=cursor_x + padding,
            y=shelf_y + padding,
            width=width,
            height=height,
        )
        placements.append(placement)
        cursor_x += item_width
        shelf_height = max(shelf_height, item_height)
        page_width = max(page_width, placement.x + placement.width + padding)
        page_height = max(page_height, placement.y + placement.height + padding)

    if placements:
        atlas_page = AtlasPage(page_index, page_width, page_height, placements)
        atlas_pages.append(atlas_page)
    return atlas_pages


def remap_transform(
    placement: Placement, page_width: int, page_height: int
) -> UVTransform:
    """Return the scale and offset that remap a tile UV into an atlas rect."""

    offset = (
        placement.x / page_width,
        (page_height - placement.y - placement.height) / page_height,
    )
    scale = (placement.width / page_width, placement.height / page_height)
    return UVTransform(offset, scale)
