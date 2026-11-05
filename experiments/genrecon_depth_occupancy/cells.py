"""Stage 1's cells edited with depth from the input views.

The new cells of a scene are the surface cells of the depth plus stage 1's cells outside every input image.
All other stage 1 cells are deleted. A surface cell holds depth points from at least MIN_VIEWS views. Stage
1's cells, the chunk layout and the cameras come from the GenRecon baseline run of the scene. Three sources:

    scan        depth rendered from the scan at the true cameras, needs one view per surface cell
    colmap      VGGT-Omega's depth (experiments/vggt_omega_depth), with each view's depth multiplied by the median
                of COLMAP depth / VGGT-Omega depth over the COLMAP points the view observes, without the pixels at depth
                edges. Needs two views per surface cell, because a single view's error would thicken the walls
    depth_only  colmap's depth with one view per surface cell and no stage 1 cells, so the grid comes from depth alone

Usage:
    uv run experiments/genrecon_depth_occupancy/cells.py {scan,colmap,depth_only} 32

writes outputs/genrecon_depth_occupancy/<source>/cells/<N>views/<scene>.npz in the format inject.py reads
(a scene that already has one is not computed again), compares the cells with the scan's and prints the mean
over the scenes.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "experiments/genrecon_oracle_occupancy")]
import evaluate_scannetpp as ev  # noqa: E402
from occupancy import BASELINE, GT, R, compare, layout, predicted_cells  # noqa: E402

OUT = ROOT / "outputs/genrecon_depth_occupancy"
WIDTH, HEIGHT = 624, 416  # VGGT-Omega's depth resolution, used for both sources
MIN_VIEWS = {"scan": 1, "colmap": 2, "depth_only": 1}
EDGE_RATIO = 0.1  # pixels whose 3x3 depth range exceeds this share of their depth are dropped


def cameras(scene: str, names: list[str]) -> tuple:
    """World-to-camera rotations and translations of the views in the scan's frame, and K at WIDTH x HEIGHT."""
    meta = json.loads((ev.DATA / scene / "dslr/nerfstudio/transforms_undistorted.json").read_text())
    frames = {f["file_path"]: f for f in meta["frames"]}
    cam_to_world = [ev.NERFSTUDIO_TO_SCAN @ np.array(frames[name]["transform_matrix"]) @ ev.OPENGL_TO_OPENCV for name in names]
    rotations = np.array([c[:3, :3].T for c in cam_to_world])
    translations = np.array([-c[:3, :3].T @ c[:3, 3] for c in cam_to_world])
    s = WIDTH / meta["w"]
    return rotations, translations, np.array([[meta["fl_x"] * s, 0, meta["cx"] * s], [0, meta["fl_y"] * s, meta["cy"] * s], [0, 0, 1]])


def project(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray, K: np.ndarray) -> tuple:
    """Pixel indices and depth of world points, and whether each lies in front of the camera inside the image."""
    cam = points @ rotation.T + translation
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.floor(K[0, 0] * cam[:, 0] / z + K[0, 2])
        v = np.floor(K[1, 1] * cam[:, 1] / z + K[1, 2])
    inside = (z > 0.05) & (u >= 0) & (u < WIDTH) & (v >= 0) & (v < HEIGHT)
    return u, v, z, inside


def scan_depth(scene: str, rotations: np.ndarray, translations: np.ndarray, K: np.ndarray) -> list[np.ndarray]:
    scan = ev.raycasting_scene(ev.load_scan(ev.DATA / scene))
    cam_to_world = np.tile(np.eye(4), (len(rotations), 1, 1))
    cam_to_world[:, :3, :3] = rotations.transpose(0, 2, 1)
    cam_to_world[:, :3, 3] = np.einsum("nji,nj->ni", rotations, -translations)
    return [ev.render_depth(scan, K, c, WIDTH, HEIGHT) for c in cam_to_world]


def observed_points(scene: str, names: list[str]) -> list[np.ndarray]:
    """For each view, the 3D positions of the COLMAP points it observes."""
    colmap = ev.DATA / scene / "dslr/colmap"
    with open(colmap / "points3D.txt") as f:
        rows = np.array([line.split(maxsplit=4)[:4] for line in f if line[0] != "#"], dtype=np.float64)
    positions = dict(zip(rows[:, 0].astype(int), rows[:, 1:]))
    observed = {}
    with open(colmap / "images.txt") as f:
        lines = (line for line in f if line[0] != "#")
        for header in lines:  # two lines per image: its name, then x, y and the point's ID per observation
            observations = next(lines)
            name = header.split()[9]
            if name in names:
                observed[name] = np.array([positions[i] for i in map(int, observations.split()[2::3]) if i in positions])
    return [observed[name] for name in names]


def drop_edges(depth: np.ndarray) -> np.ndarray:
    """NaN at depth discontinuities, where predicted depth smears between the two surfaces."""
    return np.where(maximum_filter(depth, 3) - minimum_filter(depth, 3) < EDGE_RATIO * depth, depth, np.nan)


def colmap_scaled_depth(scene: str, n: int, names: list[str], rotations: np.ndarray, translations: np.ndarray, K: np.ndarray) -> list[np.ndarray]:
    pred = np.load(ROOT / f"outputs/vggt_omega_depth/{n}views/{scene}.npz")
    if list(pred["names"]) != names:
        raise RuntimeError(f"{scene}: VGGT-Omega's views differ from the baseline's")
    depths = []
    for depth, points, rotation, translation in zip(pred["depth"], observed_points(scene, names), rotations, translations):
        u, v, z, inside = project(points, rotation, translation, K)
        depths.append(drop_edges(np.median(z[inside] / depth[v[inside].astype(int), u[inside].astype(int)]) * depth))
    return depths


def surface_cells(depths: list[np.ndarray], rotations: np.ndarray, translations: np.ndarray, K: np.ndarray, lay: dict, min_views: int) -> np.ndarray:
    """Cells inside a chunk that hold depth points from at least min_views views."""
    u, v = np.meshgrid(np.arange(WIDTH) + 0.5, np.arange(HEIGHT) + 0.5)
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=-1)
    per_view = []
    for depth, rotation, translation in zip(depths, rotations, translations):
        valid = np.isfinite(depth) & (depth > 0.05)
        points = (rays[valid] * depth[valid, None] - translation) @ rotation
        cells = np.floor((points - lay["origin"]) / lay["cell"]).astype(int)
        inside = np.zeros(len(cells), dtype=bool)
        for offset in lay["offsets"]:
            inside |= ((cells >= offset) & (cells < offset + R)).all(axis=1)
        per_view.append(np.unique(cells[inside], axis=0))
    cells, views = np.unique(np.concatenate(per_view), axis=0, return_counts=True)
    return cells[views >= min_views]


def edit(scene: str, n: int, source: str) -> dict:
    run_dir = BASELINE / f"{n}views" / scene
    lay = layout(run_dir)
    frames = json.loads((ROOT / f"data/scannetpp/eval_inputs/{n}views/{scene}/dslr/nerfstudio/transforms_undistorted.json").read_text())["frames"]
    names = sorted(f["file_path"] for f in frames)
    rotations, translations, K = cameras(scene, names)
    depths = (scan_depth(scene, rotations, translations, K) if source == "scan"
              else colmap_scaled_depth(scene, n, names, rotations, translations, K))
    cells = surface_cells(depths, rotations, translations, K, lay, MIN_VIEWS[source])

    if source != "depth_only":
        pred = predicted_cells(run_dir, lay)
        centres = lay["origin"] + (pred + 0.5) * lay["cell"]
        covered = np.zeros(len(pred), dtype=bool)
        for rotation, translation in zip(rotations, translations):
            covered |= project(centres, rotation, translation, K)[3]
        cells = np.unique(np.concatenate([cells, pred[~covered]]), axis=0)
    cells = cells.astype(np.int16)
    return {"all": cells, "visible": cells, "centres": lay["centres"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", choices=MIN_VIEWS)
    parser.add_argument("views", type=int, nargs="+", help="numbers of input views")
    parser.add_argument("--scenes", nargs="*", help="default: every scene with scan cells from occupancy.py")
    args = parser.parse_args()

    for n in args.views:
        (OUT / args.source / f"cells/{n}views").mkdir(parents=True, exist_ok=True)
        results = []
        for scene in args.scenes or sorted(p.stem for p in (GT / f"{n}views").glob("*.npz")):
            path = OUT / args.source / f"cells/{n}views/{scene}.npz"
            if not path.exists():
                np.savez_compressed(path, **edit(scene, n, args.source))
                print(f"computed {path}", flush=True)
            lay = layout(BASELINE / f"{n}views" / scene)
            results.append(compare(np.load(path)["visible"].astype(int), np.load(GT / f"{n}views/{scene}.npz")["all"].astype(int), lay["shape"]))
        mean = {name: float(np.mean([r[name] for r in results])) for name in results[0]}
        print(f"{n} views, mean over {len(results)} scenes:")
        print(json.dumps(mean, indent=2))


if __name__ == "__main__":
    main()
