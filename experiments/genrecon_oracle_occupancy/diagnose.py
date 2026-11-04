"""The depth error of the baseline and the oracle, split by the orientation of the scan surface.

At up to 10 of the official test views per scene, renders the depth of the scan and of both meshes (as
scripts/evaluate_scannetpp.py does) and splits the pixels by the scan surface they show: facing up
(floors, table tops), facing down (ceilings), vertical (walls) or other. Prints the mean over the scenes
of, per run and surface,
    share               fraction of the pixels where both the scan and the mesh have a surface
    mae                 mean |mesh depth - scan depth| in metres
    median              median of mesh depth - scan depth: > 0 means the mesh is behind the scan
    behind_5cm          fraction of the pixels where the mesh is more than 5 cm behind the scan
    in_front_5cm        and more than 5 cm in front of it
    missing             fraction of the scan's pixels where the mesh has no surface

Usage:
    uv run experiments/genrecon_oracle_occupancy/diagnose.py 32 [--workers N]
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import open3d as o3d

from occupancy import BASELINE, OUT, ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_scannetpp as ev  # noqa: E402

MAX_VIEWS = 10
SURFACES = ["up", "down", "vertical", "other"]
STATS = ["share", "mae", "median", "behind_5cm", "in_front_5cm", "missing"]


def diagnose(scene: str, n: int) -> dict:
    """{run: {surface: statistics}} of one scene."""
    scene_dir = ev.DATA / scene
    views = list(ev.test_views(scene_dir))
    views = [views[i] for i in np.linspace(0, len(views) - 1, min(MAX_VIEWS, len(views))).round().astype(int)]

    # Depth of the scan and the orientation of the surface each pixel shows.
    scan_scene = ev.raycasting_scene(ev.load_scan(scene_dir))
    scan = []
    for K, cam_to_world, valid in views:
        height, width = valid.shape
        u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
        directions = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
        directions = directions @ cam_to_world[:3, :3].T
        rays = np.concatenate([np.broadcast_to(cam_to_world[:3, 3], directions.shape), directions], -1)
        hits = scan_scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))
        depth, normals = hits["t_hit"].numpy(), hits["primitive_normals"].numpy()
        normals[(normals * directions).sum(-1) > 0] *= -1  # facing the camera
        z = normals[..., 2]
        surface = np.select([z > 0.8, z < -0.8, np.abs(z) < 0.3], [0, 1, 2], 3)
        scan.append((depth, surface, np.isfinite(depth) & valid))
    del scan_scene

    result = {}
    for run, run_dir in {"baseline": BASELINE, "oracle": OUT}.items():
        mesh_scene = ev.raycasting_scene(ev.load_prediction(run_dir / f"{n}views" / scene / "mesh.ply.zst"))
        errors, surfaces, missing = [], [], np.zeros(len(SURFACES))
        for (K, cam_to_world, valid), (scan_depth, surface, scan_hit) in zip(views, scan):
            depth = ev.render_depth(mesh_scene, K, cam_to_world, valid.shape[1], valid.shape[0])
            both = scan_hit & np.isfinite(depth)
            errors.append(depth[both] - scan_depth[both])
            surfaces.append(surface[both])
            missing += np.bincount(surface[scan_hit & ~np.isfinite(depth)], minlength=len(SURFACES))
        errors, surfaces = np.concatenate(errors), np.concatenate(surfaces)
        result[run] = {}
        for i, name in enumerate(SURFACES):
            e = errors[surfaces == i]
            if len(e) > 0:
                result[run][name] = {"share": len(e) / len(errors), "mae": np.abs(e).mean(), "median": np.median(e),
                                     "behind_5cm": (e > 0.05).mean(), "in_front_5cm": (e < -0.05).mean(),
                                     "missing": missing[i] / (missing[i] + len(e))}
        del mesh_scene
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("views", type=int, help="number of input views")
    parser.add_argument("--workers", type=int, default=1, help="scenes in parallel, about 5 GB of RAM each")
    args = parser.parse_args()

    scenes = sorted(d.name for d in (OUT / f"{args.views}views").iterdir() if (d / "mesh.ply.zst").exists())
    with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1) as pool:
        results = list(pool.map(partial(diagnose, n=args.views), scenes))

    mean = {run: {name: {stat: float(np.mean([r[run][name][stat] for r in results if name in r[run]]))
                         for stat in STATS}
                  for name in SURFACES}
            for run in results[0]}
    print(f"mean over {len(scenes)} scenes:")
    print(json.dumps(mean, indent=2))


if __name__ == "__main__":
    main()
