"""A predicted mesh against the dataset's true surface. The mesh must be in the dataset's frame.

2D metrics, from depth rendered at the dataset's test views (pooled over all pixels of all views):
    completeness    fraction of true-surface pixels that the prediction also covers
    depth_mae       mean |predicted depth - true depth| in metres, over pixels both cover
    depth_absrel    mean |predicted depth - true depth| / true depth, over the same pixels

3D metrics, from points sampled on both meshes. Predicted points farther than the dataset's
crop_distance from the true surface are dropped first.
    chamfer             mean distance to the nearest point of the other mesh, in metres
    precision@<d>       fraction of predicted points within d of the true surface
    recall@<d>          fraction of true points within d of the prediction
    fscore@<d>          harmonic mean of the two, for d = 2, 5 and 10 cm
    normal_consistency  mean |cos| between the normals of nearest points
"""

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree

from evaluation.datasets import Dataset, View

RENDER_SCALE = 0.5  # render the test views at half resolution (ScanNet++: 876 x 584), as GenRecon does
NUM_POINTS = 1_000_000  # points sampled on each mesh for the 3D metrics
THRESHOLDS = {"2cm": 0.02, "5cm": 0.05, "10cm": 0.10}
NORMAL_CUTOFF = 0.20  # metres, nearest-point pairs farther apart than this count as cos = 0


def raycasting_scene(mesh: trimesh.Trimesh) -> o3d.t.geometry.RaycastingScene:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(mesh.vertices, dtype=o3d.core.float32),
                        o3d.core.Tensor(mesh.faces, dtype=o3d.core.uint32))
    return scene


def render_depth(scene, K: np.ndarray, cam_to_world: np.ndarray, width: int, height: int) -> np.ndarray:
    """Depth along the camera's z axis for every pixel (inf where the ray hits nothing)."""
    u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)  # pixel centres
    # Ray direction in camera coordinates, scaled so that its z component is 1.
    # The ray then reaches depth z at parameter t = z, so the returned t_hit is the depth.
    directions = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=-1)
    directions = directions @ cam_to_world[:3, :3].T  # rotate into world coordinates
    origins = np.broadcast_to(cam_to_world[:3, 3], directions.shape)
    rays = np.concatenate([origins, directions], axis=-1).astype(np.float32)
    return scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()


def sample_points(mesh: trimesh.Trimesh, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Points spread uniformly over the surface (by area), with the normal of the face each lies on."""
    points, face_index = trimesh.sample.sample_surface(mesh, count, seed=0)
    return points, mesh.face_normals[face_index]


def metrics_2d(pred: trimesh.Trimesh, true: trimesh.Trimesh, views: list[View]) -> dict:
    pred_scene, true_scene = raycasting_scene(pred), raycasting_scene(true)
    abs_errors, rel_errors = [], []
    true_pixels, both_pixels = 0, 0
    for view in views:
        valid = view.valid()
        true_depth = render_depth(true_scene, view.K, view.cam_to_world, view.width, view.height)
        pred_depth = render_depth(pred_scene, view.K, view.cam_to_world, view.width, view.height)

        true_hit = np.isfinite(true_depth) & valid  # pixels where the true surface is
        both_hit = true_hit & np.isfinite(pred_depth)  # and the prediction too
        true_pixels += true_hit.sum()
        both_pixels += both_hit.sum()

        error = np.abs(pred_depth[both_hit] - true_depth[both_hit])
        abs_errors.append(error)
        rel_errors.append(error / true_depth[both_hit])

    # Pooled over all pixels of all test views, so views with more valid pixels weigh more.
    return {
        "completeness": both_pixels / true_pixels,
        "depth_mae": np.concatenate(abs_errors).mean(),
        "depth_absrel": np.concatenate(rel_errors).mean(),
    }


def metrics_3d(pred: trimesh.Trimesh, true: trimesh.Trimesh, crop_distance: float) -> dict:
    pred_points, pred_normals = sample_points(pred, NUM_POINTS)
    true_points, true_normals = sample_points(true, NUM_POINTS)

    # For each predicted point: distance to, and index of, the nearest true point.
    pred_to_true, nearest_true = cKDTree(true_points).query(pred_points)

    # Crop: predicted points far from every true surface cannot be checked, so they are not scored.
    keep = pred_to_true <= crop_distance
    pred_points, pred_normals = pred_points[keep], pred_normals[keep]
    pred_to_true, nearest_true = pred_to_true[keep], nearest_true[keep]

    # For each true point: distance to, and index of, the nearest (kept) predicted point.
    true_to_pred, nearest_pred = cKDTree(pred_points).query(true_points)

    out = {"chamfer": (pred_to_true.mean() + true_to_pred.mean()) / 2}
    for name, d in THRESHOLDS.items():
        precision = (pred_to_true < d).mean()  # predicted surface that is correct
        recall = (true_to_pred < d).mean()  # true surface that was reconstructed
        out[f"precision@{name}"] = precision
        out[f"recall@{name}"] = recall
        out[f"fscore@{name}"] = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    # |cos| of the angle between each point's normal and its nearest neighbour's normal.
    # abs() ignores which side of the surface a normal points to.
    pred_cos = np.abs((pred_normals * true_normals[nearest_true]).sum(axis=1))
    true_cos = np.abs((true_normals * pred_normals[nearest_pred]).sum(axis=1))
    pred_cos[pred_to_true > NORMAL_CUTOFF] = 0  # too far apart to be the same surface
    true_cos[true_to_pred > NORMAL_CUTOFF] = 0
    out["normal_consistency"] = (pred_cos.mean() + true_cos.mean()) / 2
    return out


def evaluate(pred: trimesh.Trimesh, dataset: Dataset, scene: str) -> dict:
    true = dataset.surface(scene)
    views = [view.scaled(RENDER_SCALE) for view in dataset.test_views(scene)]
    metrics = {**metrics_2d(pred, true, views), **metrics_3d(pred, true, dataset.crop_distance)}
    return {name: float(value) for name, value in metrics.items()}
