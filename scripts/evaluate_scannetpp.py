"""Compare predicted meshes with the ScanNet++ laser scans and average over the scenes.

2D metrics, from depth rendered at the official DSLR test views:
    depth_mae       mean |predicted depth - scan depth| in metres
    depth_absrel    mean |predicted depth - scan depth| / scan depth
    completeness    fraction of scan pixels that the prediction also covers

3D metrics, from points sampled on both meshes:
    chamfer             mean distance to the nearest point of the other mesh, in metres
    precision, recall   fraction of predicted / scan points within 10 cm of the other mesh
    fscore              harmonic mean of precision and recall
    normal_consistency  mean |cos| between the normals of nearest points

The predicted mesh must be in the scan's coordinate frame (metres), as given by the
ScanNet++ camera poses.

Usage:
    uv run scripts/evaluate_scannetpp.py PRED_DIR

PRED_DIR has one folder per scene, named by its ID, with mesh.ply or a zstd-compressed mesh.ply.zst.
The metrics of each scene are written next to its mesh as metrics.json (a scene that already has
one is not evaluated again), and the mean over the scenes is printed.
"""

import argparse
import io
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

DATA = Path(__file__).resolve().parents[1] / "data/scannetpp/data"

RENDER_SCALE = 0.5 # render the 1752 x 1168 test views at 876 x 584, as GenRecon does
NUM_POINTS = 1_000_000  # points sampled on each mesh for the 3D metrics
CROP_DISTANCE = 0.15  # metres, predicted points farther than this from the scan are not scored, crop from GenRecon
FSCORE_THRESHOLD = 0.10  # metres
NORMAL_CUTOFF = 0.20  # metres, nearest-point pairs farther apart than this count as cos = 0

# ScanNet++'s nerfstudio poses are in a world frame with x and y swapped and z flipped
# relative to the scan (the same matrix as GenRecon's inference/get_images.py).
NERFSTUDIO_TO_SCAN = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=float)
# Nerfstudio cameras look along -z with y up. OpenCV cameras look along +z with y down.
OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


def load_prediction(path: Path) -> trimesh.Trimesh:
    if path.suffix == ".zst":
        ply = subprocess.run(["zstd", "-dc", path], capture_output=True, check=True).stdout
        return trimesh.load(io.BytesIO(ply), file_type="ply", process=False)
    return trimesh.load(path, process=False)


def load_scan(scene_dir: Path) -> trimesh.Trimesh:
    scan = trimesh.load(scene_dir / "scans/mesh_aligned_0.05.ply", process=False)
    # vertex numbers that contain blurred out regions for privacy
    anonymized = [int(i) for i in (scene_dir / "scans/mesh_aligned_0.05_mask.txt").read_text().split()]
    keep = ~np.isin(scan.faces, anonymized).any(axis=1)
    return trimesh.Trimesh(scan.vertices, scan.faces[keep], process=False)


def test_views(scene_dir: Path):
    """Yield (intrinsics K, camera-to-world pose, valid-pixel mask) for each official DSLR test view.

    The pose maps OpenCV camera coordinates (x right, y down, z forward) to scan coordinates.
    """
    meta = json.loads((scene_dir / "dslr/nerfstudio/transforms_undistorted.json").read_text())
    width, height = round(meta["w"] * RENDER_SCALE), round(meta["h"] * RENDER_SCALE)
    K = np.array([
        [meta["fl_x"] * RENDER_SCALE, 0, meta["cx"] * RENDER_SCALE],
        [0, meta["fl_y"] * RENDER_SCALE, meta["cy"] * RENDER_SCALE],
        [0, 0, 1],
    ])
    for frame in meta["test_frames"]:
        cam_to_world = NERFSTUDIO_TO_SCAN @ np.array(frame["transform_matrix"]) @ OPENGL_TO_OPENCV
        # 255 = valid, 0 = anonymized (e.g. a person)
        mask = Image.open(scene_dir / "dslr/resized_undistorted_masks" / frame["mask_path"])
        valid = np.array(mask.resize((width, height), Image.NEAREST)) > 0
        yield K, cam_to_world, valid


# 2D metrics

def raycasting_scene(mesh: trimesh.Trimesh) -> o3d.t.geometry.RaycastingScene:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(mesh.vertices, dtype=o3d.core.float32),
                        o3d.core.Tensor(mesh.faces, dtype=o3d.core.uint32))
    return scene


def render_depth(scene, K: np.ndarray, cam_to_world: np.ndarray, width: int, height: int) -> np.ndarray:
    """Depth along the camera's z axis for every pixel (inf if ray hits nothing)."""
    u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)  # pixel centres
    # Ray direction in camera coordinates, scaled so that its z component is 1.
    # The ray then reaches depth z at parameter t = z, so the returned t_hit is the depth.
    directions = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=-1)
    directions = directions @ cam_to_world[:3, :3].T  # rotate into scan coordinates
    origins = np.broadcast_to(cam_to_world[:3, 3], directions.shape)
    rays = np.concatenate([origins, directions], axis=-1).astype(np.float32)
    return scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()


def metrics_2d(pred: trimesh.Trimesh, scan: trimesh.Trimesh, scene_dir: Path) -> dict:
    pred_scene, scan_scene = raycasting_scene(pred), raycasting_scene(scan)
    abs_errors, rel_errors = [], []
    scan_pixels, both_pixels = 0, 0
    for K, cam_to_world, valid in test_views(scene_dir):
        height, width = valid.shape
        scan_depth = render_depth(scan_scene, K, cam_to_world, width, height)
        pred_depth = render_depth(pred_scene, K, cam_to_world, width, height)

        scan_hit = np.isfinite(scan_depth) & valid  # pixels where the scan has a surface
        both_hit = scan_hit & np.isfinite(pred_depth)  # and the prediction too!
        scan_pixels += scan_hit.sum()
        both_pixels += both_hit.sum()

        error = np.abs(pred_depth[both_hit] - scan_depth[both_hit])
        abs_errors.append(error)
        rel_errors.append(error / scan_depth[both_hit])

    # Pooled over all pixels of all test views, so views with more valid pixels weigh more.
    return {
        "depth_mae": np.concatenate(abs_errors).mean(),
        "depth_absrel": np.concatenate(rel_errors).mean(),
        "completeness": both_pixels / scan_pixels,
    }


# 3D metrics


def sample_points(mesh: trimesh.Trimesh, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Points spread uniformly over the surface (by area), with the normal of the face each lies on."""
    points, face_index = trimesh.sample.sample_surface(mesh, count, seed=0)
    return points, mesh.face_normals[face_index]


def metrics_3d(pred: trimesh.Trimesh, scan: trimesh.Trimesh) -> dict:
    pred_points, pred_normals = sample_points(pred, NUM_POINTS)
    scan_points, scan_normals = sample_points(scan, NUM_POINTS)

    # For each predicted point: distance to, and index of, the nearest scan point.
    pred_to_scan, nearest_scan = cKDTree(scan_points).query(pred_points)

    # Crop: predicted points far from every scanned surface (e.g. seen through a window,
    # where the scan has nothing) cannot be checked, so they are not scored.
    keep = pred_to_scan <= CROP_DISTANCE
    pred_points, pred_normals = pred_points[keep], pred_normals[keep]
    pred_to_scan, nearest_scan = pred_to_scan[keep], nearest_scan[keep]

    # For each scan point: distance to, and index of, the nearest (kept) predicted point.
    scan_to_pred, nearest_pred = cKDTree(pred_points).query(scan_points)

    chamfer = (pred_to_scan.mean() + scan_to_pred.mean()) / 2

    precision = (pred_to_scan < FSCORE_THRESHOLD).mean()  # predicted surface that is correct
    recall = (scan_to_pred < FSCORE_THRESHOLD).mean()  # scanned surface that was reconstructed
    fscore = 2 * precision * recall / (precision + recall)

    # |cos| of the angle between each point's normal and its nearest neighbour's normal.
    # abs() ignores which side of the surface a normal points to.
    pred_cos = np.abs((pred_normals * scan_normals[nearest_scan]).sum(axis=1))
    scan_cos = np.abs((scan_normals * pred_normals[nearest_pred]).sum(axis=1))
    pred_cos[pred_to_scan > NORMAL_CUTOFF] = 0  # too far apart to be the same surface
    scan_cos[scan_to_pred > NORMAL_CUTOFF] = 0
    normal_consistency = (pred_cos.mean() + scan_cos.mean()) / 2

    return {
        "chamfer": chamfer,
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
        "normal_consistency": normal_consistency,
    }


def evaluate(mesh: Path) -> dict:
    scene_dir = DATA / mesh.parent.name
    pred, scan = load_prediction(mesh), load_scan(scene_dir)
    metrics = {**metrics_2d(pred, scan, scene_dir), **metrics_3d(pred, scan)}
    return {name: float(value) for name, value in metrics.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pred_dir", type=Path)
    args = parser.parse_args()

    meshes = sorted(args.pred_dir.glob("*/mesh.ply*"))
    todo = [mesh for mesh in meshes if not (mesh.parent / "metrics.json").exists()]
    # Each scene in a new process: one process does not release all memory between scenes.
    with ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1) as pool:
        for mesh, metrics in zip(todo, pool.map(evaluate, todo)):
            print(f"evaluated {mesh}", flush=True)
            (mesh.parent / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    results = [json.loads((mesh.parent / "metrics.json").read_text()) for mesh in meshes]
    mean = {name: float(np.mean([r[name] for r in results])) for name in results[0]}
    print(f"mean over {len(results)} scenes:")
    print(json.dumps(mean, indent=2))


if __name__ == "__main__":
    main()
