"""Where stage 1's extra cells lie, as seen from the input views.

An extra cell is one that stage 1 marks occupied and that contains no scan surface (occupancy.py). Its
centre is looked at from every input view, inside the square centre crop GenRecon uses, with the scan's
depth as the visible surface:

    free     a view sees past it: it is more than half a cell in front of the visible surface
    behind   no view sees past it, but a view sees it more than half a cell behind the visible surface
    near     within half a cell of the visible surface in every view that sees it
    unseen   no view sees it

Depth from the input views can only show that the free cells are empty.

Usage:
    uv run experiments/genrecon_oracle_occupancy/free_space.py 32

prints the mean over the scenes of the fraction of stage 1's cells that are extra and of the fraction of
the extra cells in each class.
"""

import argparse
import json
import sys

import numpy as np
import open3d as o3d

from occupancy import BASELINE, GT, ROOT, input_views, layout, predicted_cells

sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_scannetpp as ev  # noqa: E402

RESOLUTION = 256  # pixels per side of the scan's depth map per input view
CLASSES = ["free", "behind", "near", "unseen"]


def classify(scene: str, n: int) -> dict:
    run_dir = BASELINE / f"{n}views" / scene
    lay = layout(run_dir)
    pred = predicted_cells(run_dir, lay)
    scan_grid = np.zeros(lay["shape"], dtype=bool)
    scan_grid[tuple(np.load(GT / f"{n}views/{scene}.npz")["all"].astype(int).T)] = True
    extra = pred[~scan_grid[tuple(pred.T)]]
    centres = lay["origin"] + (extra + 0.5) * lay["cell"]
    margin = lay["cell"] / 2

    scan_scene = ev.raycasting_scene(ev.load_scan(ev.DATA / scene))
    u, v = np.meshgrid(*[(np.arange(RESOLUTION) + 0.5) / RESOLUTION] * 2)
    free, behind, seen = (np.zeros(len(extra), dtype=bool) for _ in range(3))
    for w2c, K in input_views(run_dir, lay):
        # The scan's depth: with directions of camera z = 1, t_hit is the depth.
        rotation, translation = w2c[:, :3], w2c[:, 3]
        directions = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1) @ rotation
        origins = np.broadcast_to(-np.linalg.solve(rotation, translation), directions.shape)
        rays = np.concatenate([origins, directions], -1).astype(np.float32)
        depth = scan_scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()

        # The depth of each cell centre, and the scan's depth at its pixel.
        cam = centres @ rotation.T + translation
        z = cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            px = np.floor((K[0, 0] * cam[:, 0] / z + K[0, 2]) * RESOLUTION)
            py = np.floor((K[1, 1] * cam[:, 1] / z + K[1, 2]) * RESOLUTION)
        inside = (z > 0.05) & (px >= 0) & (px < RESOLUTION) & (py >= 0) & (py < RESOLUTION)
        surface = np.full(len(extra), np.inf)
        surface[inside] = depth[py[inside].astype(int), px[inside].astype(int)]
        seen |= inside
        free |= inside & (z < surface - margin)
        behind |= inside & (z > surface + margin)

    labels = np.select([free, behind, seen], [0, 1, 2], 3)
    return {"extra": len(extra) / len(pred), **{c: (labels == i).mean() for i, c in enumerate(CLASSES)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("views", type=int, help="number of input views")
    args = parser.parse_args()

    results = []
    for gt in sorted((GT / f"{args.views}views").glob("*.npz")):
        results.append(classify(gt.stem, args.views))
        print(f"{gt.stem} done", flush=True)
    mean = {name: float(np.mean([r[name] for r in results])) for name in results[0]}
    print(f"mean over {len(results)} scenes:")
    print(json.dumps(mean, indent=2))


if __name__ == "__main__":
    main()
