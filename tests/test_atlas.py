import numpy as np
import pytest

from earth.atlas import AtlasError, Placement, plan_pages, remap_uv


def test_plan_pages_trims_to_used_size() -> None:
    pages = plan_pages({'a.png': (512, 256)}, 1024, padding=0)

    assert len(pages) == 1
    assert pages[0].width == 512
    assert pages[0].height == 256
    assert pages[0].index == 0


def test_plan_pages_includes_padding_in_page_size() -> None:
    pages = plan_pages({'a.png': (512, 256)}, 1024, padding=4)

    assert pages[0].placements[0].x == 4
    assert pages[0].placements[0].y == 4
    assert pages[0].width == 520
    assert pages[0].height == 264


def test_plan_pages_splits_oversized_groups_across_pages() -> None:
    sizes = {f'{i}.png': (1024, 1024) for i in range(5)}
    pages = plan_pages(sizes, 1024, padding=0)

    assert len(pages) == 5
    assert [page.index for page in pages] == list(range(5))
    assert all(page.width == 1024 and page.height == 1024 for page in pages)


def test_plan_pages_raises_for_oversized_texture() -> None:
    with pytest.raises(AtlasError):
        plan_pages({'big.png': (2048, 256)}, 1024, padding=0)


def test_remap_uv_uses_page_dimensions() -> None:
    placement = Placement('a.png', x=0, y=0, width=256, height=256)
    uv = np.array([[0.0, 0.0], [1.0, 1.0]])

    remapped = remap_uv(uv, placement, 512, 256)

    assert np.allclose(remapped, [[0.0, 0.0], [0.5, 1.0]])
