"""CPU raycast cutaway previews; GLBs and quantitative geometry stay intact."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from ..common import OUT, geometry, read
from ..evaluation.surface import query


def render(q, center, diagonal, direction, width=600):
    eye = center + direction / np.linalg.norm(direction) * diagonal * 1.5
    forward = center - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 1., 0.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    yy, xx = np.mgrid[:width, :width]
    rays_d = (forward + (xx[..., None] + .5 - width / 2) / width * .65 * right
              - (yy[..., None] + .5 - width / 2) / width * .65 * up)
    rays = np.concatenate([np.broadcast_to(eye, rays_d.shape), rays_d], axis=-1).astype(np.float32)
    result = q.cast_rays(o3d.core.Tensor(rays), nthreads=4)
    mask = np.isfinite(result['t_hit'].numpy())
    normals = result['primitive_normals'].numpy()
    light = np.array([.4, .8, .6])
    light /= np.linalg.norm(light)
    shade = .35 + .65 * np.abs(normals @ light)
    image = np.ones((width, width, 3)) * .96
    image[mask] = np.array([.52, .69, .78]) * shade[mask, None]
    return image


def main(dest):
    reference = geometry(dest / 'original.glb')
    decoded = geometry(dest / 'decoded.glb')
    center = reference.bounds.mean(axis=0)
    diagonal = np.linalg.norm(reference.extents)
    # Display-only horizontal cut removes the ceiling and top wall strip.
    cut_y = reference.bounds[1, 1] - reference.extents[1] * .12
    originals = [reference, decoded]
    displays = [m.slice_plane([0, cut_y, 0], [0, -1, 0], cap=False) for m in originals]
    queries = [query(m) for m in displays]
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), facecolor='white')
    for row, direction in enumerate([np.array([.7, 1.6, 1.2]), np.array([-1.1, 1.6, -.7])]):
        for col, q in enumerate(queries):
            axes[row, col].imshow(render(q, center, diagonal, direction))
            axes[row, col].axis('off')
            axes[row, col].set_title('Original ground truth' if col == 0 else 'Raw latent → shape decoder (512)')
    fig.suptitle(f'{dest.parent.name if dest.name.startswith("chunk_") else dest.name}: Ground-truth latent reconstruction baseline\n'
                 'Identical cameras and source frame; display cutaway removes ceiling. No alignment.', fontsize=13)
    fig.tight_layout()
    fig.savefig(dest / 'comparison.png', dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    main(parser.parse_args().destination)
