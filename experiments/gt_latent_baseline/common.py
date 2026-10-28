"""Paths, pinned model settings, and shared artifact/geometry helpers."""
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'papers/GenRecon'
OUT = ROOT / 'outputs/gt_latent_baseline'
UPSTREAM = 'eaf1468118d20469d17079a4a19737297d2ef87b'
HF_REPO = 'microsoft/TRELLIS.2-4B'
HF_REVISION = 'af44b45f2e35a493886929c6d786e563ec68364d'
ENCODER = 'shape_enc_next_dc_f16c32_fp16'
DECODER = 'shape_dec_next_dc_f16c32_fp16'
Y_TO_Z = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
RESOLUTION = 512


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text())


def fingerprint(path):
    path = Path(path)
    with path.open('rb') as f:
        checksum = hashlib.file_digest(f, 'sha256').hexdigest()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=checksum)


def checkpoint_files(name):
    from huggingface_hub import hf_hub_download
    return {ext: Path(hf_hub_download(HF_REPO, f'ckpts/{name}.{ext}', revision=HF_REVISION))
            for ext in ('json', 'safetensors')}


def geometry(path):
    return trimesh.load_scene(path, process=False).to_mesh()


def clip(mesh, low, high):
    """Clip triangles exactly at cube boundaries without adding cap surfaces."""
    result = mesh.copy()
    for axis in range(3):
        for sign, bound in [(1, low[axis]), (-1, high[axis])]:
            normal = np.zeros(3)
            normal[axis] = sign
            origin = np.zeros(3)
            origin[axis] = bound
            result = result.slice_plane(origin, normal, cap=False)
            if not len(result.faces):
                return result
    return result


def mesh_valid(mesh):
    assert len(mesh.faces) and len(mesh.vertices), 'Empty geometry'
    assert np.isfinite(mesh.vertices).all(), 'Nonfinite geometry'


def export(mesh, path):
    mesh_valid(mesh)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)


