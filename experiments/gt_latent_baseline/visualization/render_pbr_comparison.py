"""Render base-color material previews with identical cameras, using CPU rays.

GLBs contain full PBR maps; this PNG uses diffuse lighting for color inspection.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import trimesh

from ..common import ROOT, Y_TO_Z, read


def scene_parts(scene, cut_y):
    parts = []
    for node in scene.graph.nodes_geometry:
        tf, name = scene.graph[node]
        mesh = scene.geometry[name].copy()
        mesh.apply_transform(tf)
        visual = mesh.visual
        material = getattr(visual, 'material', None)
        if not isinstance(visual, trimesh.visual.TextureVisuals):
            color = [255, 255, 255, 255] if visual.kind is None else visual.main_color
            material = trimesh.visual.material.PBRMaterial(baseColorFactor=color)
            mesh.visual = trimesh.visual.TextureVisuals(uv=np.zeros((len(mesh.vertices), 2)), material=material)
        elif visual.uv is None:
            mesh.visual = trimesh.visual.TextureVisuals(uv=np.zeros((len(mesh.vertices), 2)), material=material)
        mesh = mesh.slice_plane([0, cut_y, 0], [0, -1, 0], cap=False)
        if len(mesh.faces):
            parts.append(mesh)
    q = o3d.t.geometry.RaycastingScene(nthreads=4)
    for mesh in parts:
        q.add_triangles(o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor(np.asarray(mesh.vertices, np.float32)),
            o3d.core.Tensor(np.asarray(mesh.faces, np.uint32))))
    return q, parts


def render(q, parts, center, diagonal, direction, width=700):
    eye = center + direction / np.linalg.norm(direction) * diagonal * 1.5
    forward = center - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 1., 0.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    yy, xx = np.mgrid[:width, :width]
    rays_d = (forward + (xx[..., None] + .5 - width / 2) / width * .65 * right
              - (yy[..., None] + .5 - width / 2) / width * .65 * up)
    rays = np.concatenate([np.broadcast_to(eye, rays_d.shape), rays_d], -1).astype(np.float32)
    result = q.cast_rays(o3d.core.Tensor(rays), nthreads=4)
    gids = result['geometry_ids'].numpy()
    fids = result['primitive_ids'].numpy()
    bary = result['primitive_uvs'].numpy()
    normals = result['primitive_normals'].numpy()
    image = np.ones((width, width, 3)) * .96
    light = np.array([.4, .8, .6])
    light /= np.linalg.norm(light)
    shade = .55 + .45 * np.abs(normals @ light)
    for gid, mesh in enumerate(parts):
        mask = gids == gid
        if not mask.any():
            continue
        b = bary[mask]
        weights = np.concatenate([1 - b.sum(axis=1, keepdims=True), b], axis=1)
        uv = (mesh.visual.uv[mesh.faces[fids[mask]]] * weights[..., None]).sum(axis=1)
        mat = mesh.visual.material
        if not isinstance(mat, trimesh.visual.material.PBRMaterial):
            mat = mat.to_pbr()
        factor = np.ones(3) if mat.baseColorFactor is None else mat.baseColorFactor[:3] / 255
        color = np.broadcast_to(factor, (len(uv), 3)).copy()
        if mat.baseColorTexture is not None:
            texture = np.asarray(mat.baseColorTexture.convert('RGB'), float) / 255
            # glTF/trimesh UV origin is lower left; images have top-left origin.
            x = np.mod(uv[:, 0], 1) * (texture.shape[1] - 1)
            y = (1 - np.mod(uv[:, 1], 1)) * (texture.shape[0] - 1)
            x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
            x1, y1 = np.minimum(x0 + 1, texture.shape[1] - 1), np.minimum(y0 + 1, texture.shape[0] - 1)
            u, v = (x - x0)[:, None], (y - y0)[:, None]
            sampled = ((1-u)*(1-v)*texture[y0, x0] + u*(1-v)*texture[y0, x1]
                       + (1-u)*v*texture[y1, x0] + u*v*texture[y1, x1])
            color *= sampled
        image[mask] = color * shade[mask, None]
    return np.clip(image, 0, 1)


def main(dest):
    if dest.name.startswith('chunk_'):
        reference = trimesh.load_scene(dest / 'normalized_pbr.glb', process=False)
        matrix = np.array(read(dest / 'transform.json')['M_chunk_zup_to_source_yup']) @ Y_TO_Z
        reference.apply_transform(matrix)
    else:
        reference = trimesh.load_scene(ROOT / 'outputs/3dfront_benchmark' / dest.name / 'reference/full_room.glb', process=False)
    decoded = trimesh.load_scene(dest / 'decoded_pbr.glb', process=False)
    center = reference.bounds.mean(axis=0)
    diagonal = np.linalg.norm(reference.extents)
    cut_y = reference.bounds[1, 1] - reference.extents[1] * .12
    queries = [scene_parts(s, cut_y) for s in (reference, decoded)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for row, direction in enumerate([np.array([.7, 1.6, 1.2]), np.array([-1.1, 1.6, -.7])]):
        for col, (q, parts) in enumerate(queries):
            axes[row, col].imshow(render(q, parts, center, diagonal, direction))
            axes[row, col].axis('off')
            axes[row, col].set_title('Original source materials' if col == 0 else 'Exact decoded shape + decoded GT PBR')
    fig.suptitle(f'{dest.name}: Ground-truth latent reconstruction baseline\n'
        'Identical cameras; ceiling cutaway; diffuse base-color preview. GLB retains full PBR maps.')
    fig.tight_layout()
    fig.savefig(dest / 'comparison_pbr.png', dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    main(parser.parse_args().destination)
