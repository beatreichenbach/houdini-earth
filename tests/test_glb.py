from pathlib import Path

import numpy as np
import PIL.Image
import trimesh
from trimesh.exchange.gltf import export_glb
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from earth.glb import load_chunk


def _export(scene: trimesh.Scene, tmp_path: Path) -> str:
    glb = tmp_path / 'tile.glb'
    glb.write_bytes(export_glb(scene))
    return str(glb)


def test_load_chunk_textured(tmp_path: Path) -> None:
    mesh = trimesh.creation.box()
    image = PIL.Image.new('RGB', (4, 4), (255, 0, 0))
    mesh.visual = TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2)),
        material=PBRMaterial(baseColorTexture=image),
    )
    path = _export(trimesh.Scene(mesh), tmp_path)

    chunk = load_chunk(path)

    assert len(chunk.primitives) == 1
    primitive = chunk.primitives[0]
    assert primitive.points.shape == (8, 3)
    assert primitive.faces.shape == (12, 3)
    assert primitive.uvs.shape == (8, 2)
    assert primitive.texture


def test_load_chunk_plain(tmp_path: Path) -> None:
    path = _export(trimesh.Scene(trimesh.creation.box()), tmp_path)

    chunk = load_chunk(path)

    assert len(chunk.primitives) == 1
    primitive = chunk.primitives[0]
    assert primitive.texture == ''
    assert primitive.uvs.shape == (8, 2)
