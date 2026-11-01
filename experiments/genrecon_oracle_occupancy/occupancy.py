"""The scan's occupancy on GenRecon's sparse-structure grid, compared with the occupancy stage 1 predicts.

GenRecon cuts a scene into overlapping cubic chunks of about 3 m. For each chunk, stage 1 predicts which
cells of a 32^3 grid (about 10 cm per cell) contain surface, and stages 2 and 3 generate geometry only
inside those cells. This voxelizes the ScanNet++ laser scan on the same grid, using the chunk layout and
the input views of the GenRecon baseline run of the scene:

    all      cells that contain scan surface
    visible  cells with scan surface that at least one input view sees: in front of the camera, inside
             the square centre crop GenRecon uses, and not hidden behind other scan surface

Cells are indexed in one grid over the whole scene (as in GenRecon's
FullSceneImagesTo3DPipeline._global_grid_layout): chunk i covers [offset_i, offset_i + 32) per axis.

It then compares stage 1's cells with the `all` cells and prints the mean over the scenes of
    cell_ratio          predicted cells / scan cells
    precision           fraction of predicted cells that contain scan surface
    recall              fraction of scan cells that are predicted
    recall_within_1     fraction of scan cells with a predicted cell at most 1 cell away
    pred_1_away, pred_2_away, pred_3plus_away
                        fraction of predicted cells 1, 2, 3 or more cells away from the nearest scan cell
Distances are in cells, and a diagonal neighbour is 1 cell away.

Usage:
    uv run experiments/genrecon_oracle_occupancy/occupancy.py 32

writes outputs/genrecon_oracle_occupancy/gt/<N>views/<scene>.npz for every scene that has a mesh in
outputs/genrecon_baseline/<N>views/ (a scene that already has one is not computed again).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from scipy.ndimage import distance_transform_cdt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_scannetpp as ev  # noqa: E402

BASELINE = ROOT / "outputs/genrecon_baseline"
OUT = ROOT / "outputs/genrecon_oracle_occupancy"
GT = OUT / "gt"

R = 32  # cells per chunk side
POINTS_PER_M2 = 10_000  # scan surface samples, about 1 cm apart
OCCLUSION_TOLERANCE = 0.03  # metres, a sample is visible if the scan's first hit is at most this much before it


def layout(run_dir: Path) -> dict:
    """Chunk layout of a GenRecon run, from its chunk_transforms.json."""
    chunks = json.loads((run_dir / "chunk_transforms.json").read_text())["chunks"]
    centres = np.array([c["crop_center"] for c in chunks])
    size = chunks[0]["M_chunk_to_original"][0][0]
    rel = (centres - centres[0]) / size  # chunk centres in units of the chunk size, multiples of 1/16
    offsets = np.round((rel - rel.min(axis=0)) * R).astype(int)
    origin = centres[0] + (rel.min(axis=0) - 0.5) * size  # world position of the grid's corner
    return {"centres": centres, "size": size, "cell": size / R, "offsets": offsets, "origin": origin,
            "shape": tuple(offsets.max(axis=0) + R)}


def predicted_cells(run_dir: Path, lay: dict) -> np.ndarray:
    """The cells stage 1 marked occupied, from the coords_<chunk>.ply (world cell centres) of the run."""
    centres = np.concatenate([trimesh.load(p).vertices for p in sorted(run_dir.glob("coords_*.ply"))])
    cells = np.round((centres - lay["origin"]) / lay["cell"] - 0.5).astype(int)
    return np.unique(cells, axis=0)


def input_views(run_dir: Path, lay: dict):
    """Yield (world-to-camera 3x4 in metres, intrinsics) of the views GenRecon used.

    cameras.json stores OpenCV world-to-camera matrices in the chunk-0 frame, and intrinsics normalized
    to the square crop: a point is inside the crop if its u and v are in [0, 1].
    """
    world_to_chunk0 = np.diag([1 / lay["size"]] * 3 + [1.0])
    world_to_chunk0[:3, 3] = -lay["centres"][0] / lay["size"]
    for view in json.loads((run_dir / "cameras.json").read_text())["scene"]:
        w2c = np.array(view["extrinsics_c0"]) @ world_to_chunk0
        yield w2c[:3], np.array(view["intrinsics"])


def visible(points: np.ndarray, scan: trimesh.Trimesh, views) -> np.ndarray:
    """Whether each point on the scan is seen, unoccluded, by at least one view."""
    scene = ev.raycasting_scene(scan)
    seen = np.zeros(len(points), dtype=bool)
    for w2c, K in views:
        cam = points @ w2c[:, :3].T + w2c[:, 3]
        z = cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = K[0, 0] * cam[:, 0] / z + K[0, 2]
            v = K[1, 1] * cam[:, 1] / z + K[1, 2]
        candidates = np.flatnonzero((z > 0.05) & (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1) & ~seen)
        if len(candidates) == 0:
            continue
        centre = -np.linalg.solve(w2c[:, :3], w2c[:, 3])
        direction = points[candidates] - centre
        distance = np.linalg.norm(direction, axis=1)
        rays = np.concatenate([np.broadcast_to(centre, direction.shape), direction / distance[:, None]], axis=1)
        hit = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))["t_hit"].numpy()
        seen[candidates[hit >= distance - OCCLUSION_TOLERANCE]] = True
    return seen


def scan_cells(run_dir: Path) -> dict:
    """The `all` and `visible` cells of a scene on the grid of its GenRecon run."""
    lay = layout(run_dir)
    scan = ev.load_scan(ev.DATA / run_dir.name)
    points, _ = ev.sample_points(scan, int(scan.area * POINTS_PER_M2))
    cells = np.floor((points - lay["origin"]) / lay["cell"]).astype(int)
    inside = np.zeros(len(cells), dtype=bool)  # in at least one chunk
    for offset in lay["offsets"]:
        inside |= ((cells >= offset) & (cells < offset + R)).all(axis=1)
    points, cells = points[inside], cells[inside]
    seen = visible(points, scan, input_views(run_dir, lay))
    return {"all": np.unique(cells, axis=0).astype(np.int16),
            "visible": np.unique(cells[seen], axis=0).astype(np.int16),
            "centres": lay["centres"]}  # so that inject.py can check the chunk layout


def compare(pred: np.ndarray, scan: np.ndarray, shape: tuple) -> dict:
    pred_grid, scan_grid = np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)
    pred_grid[tuple(pred.T)] = True
    scan_grid[tuple(scan.T)] = True
    # distance of each cell to the nearest cell of the other set
    pred_to_scan = distance_transform_cdt(~scan_grid, metric="chessboard")[tuple(pred.T)]
    scan_to_pred = distance_transform_cdt(~pred_grid, metric="chessboard")[tuple(scan.T)]
    return {
        "cell_ratio": len(pred) / len(scan),
        "precision": (pred_to_scan == 0).mean(),
        "recall": (scan_to_pred == 0).mean(),
        "recall_within_1": (scan_to_pred <= 1).mean(),
        "pred_1_away": (pred_to_scan == 1).mean(),
        "pred_2_away": (pred_to_scan == 2).mean(),
        "pred_3plus_away": (pred_to_scan >= 3).mean(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("views", type=int, nargs="+", help="numbers of input views")
    args = parser.parse_args()

    for n in args.views:
        (GT / f"{n}views").mkdir(parents=True, exist_ok=True)
        results = []
        for run_dir in sorted((BASELINE / f"{n}views").iterdir()):
            if not (run_dir / "mesh.ply.zst").exists():
                continue
            path = GT / f"{n}views/{run_dir.name}.npz"
            if not path.exists():
                np.savez_compressed(path, **scan_cells(run_dir))
                print(f"computed {path}", flush=True)
            lay = layout(run_dir)
            results.append(compare(predicted_cells(run_dir, lay), np.load(path)["all"].astype(int), lay["shape"]))

        mean = {name: float(np.mean([r[name] for r in results])) for name in results[0]}
        print(f"{n} views, mean over {len(results)} scenes:")
        print(json.dumps(mean, indent=2))


if __name__ == "__main__":
    main()
