"""A run's depth at its input views against depth rendered from the true surface at the true cameras.

Pixels without true surface, invalid pixels and pixels without a prediction are left out. Predicted
depth may have any scale, so the error depends on how the scale is set:

    scene   one scale per scene, from the similarity fit of the predicted camera centres onto the true
            ones (align.umeyama)
    view    one scale per view, the median of true depth / predicted depth: the error of the depth's shape
    points  the depth in 3D with the run's own cameras, moved by the same fit: distance to the true surface,
            so the error of the depth and of the cameras together

For each: <name>_median_cm and the fraction of pixels (or points) within 2.5, 5 and 10 cm; for scene and
view also the mean relative error <name>_absrel.
"""

from dataclasses import replace

import numpy as np
import open3d as o3d
import trimesh

from evaluation.align import umeyama
from evaluation.datasets import View
from evaluation.formats import Cameras, Depth
from evaluation.metrics.mesh import raycasting_scene, render_depth

NUM_POINTS = 200_000  # points per scene for the distance to the true surface


def summary(name: str, errors: np.ndarray) -> dict:
    return {f"{name}_median_cm": np.median(errors) * 100, f"{name}_within_2.5cm": (errors < 0.025).mean(),
            f"{name}_within_5cm": (errors < 0.05).mean(), f"{name}_within_10cm": (errors < 0.10).mean()}


def evaluate(depth: Depth, cameras: Cameras, views: list[View], surface: trimesh.Trimesh) -> dict:
    """cameras: the run's, at the resolution of the depth maps. views: the true cameras of the same images."""
    cameras = cameras.subset(depth.names)
    maps = depth.depth
    Rs, ts, Ks = cameras.world_to_camera[:, :3, :3], cameras.world_to_camera[:, :3, 3], cameras.K
    height, width = maps.shape[1:]
    true = {view.name: view for view in views}
    # The true cameras at the size of the depth maps
    views = [replace(true[name].scaled(width / true[name].width), width=width, height=height) for name in depth.names]

    predicted, true_centres = cameras.centres.astype(np.float64), np.array([v.cam_to_world[:3, 3] for v in views])
    fit = umeyama(predicted, true_centres)
    camera_error = np.linalg.norm(fit.apply(predicted) - true_centres, axis=1)

    scene = raycasting_scene(surface)
    rng = np.random.default_rng(0)
    scene_errors, scene_relative, view_errors, view_relative, view_scales, points = [], [], [], [], [], []
    for i, view in enumerate(views):
        truth = render_depth(scene, view.K, view.cam_to_world, width, height)
        valid = view.valid() & np.isfinite(truth) & (maps[i] > 0)
        if not valid.any():
            continue
        v, u = np.nonzero(valid)
        d, t = fit.scale * maps[i][v, u], truth[v, u]
        view_scale = np.median(t / d)
        view_scales.append(view_scale)
        scene_errors.append(np.abs(d - t))
        scene_relative.append(np.abs(d - t) / t)
        view_errors.append(np.abs(view_scale * d - t))
        view_relative.append(np.abs(view_scale * d - t) / t)

        pick = rng.choice(len(u), min(NUM_POINTS // len(views), len(u)), replace=False)
        u, v, z = u[pick], v[pick], maps[i][v[pick], u[pick]]
        cam = np.stack([(u + 0.5 - Ks[i, 0, 2]) / Ks[i, 0, 0] * z, (v + 0.5 - Ks[i, 1, 2]) / Ks[i, 1, 1] * z, z], axis=1)
        points.append(fit.scale * (cam - ts[i]) @ Rs[i] @ fit.rotation.T + fit.translation)

    distance = scene.compute_distance(o3d.core.Tensor(np.concatenate(points).astype(np.float32))).numpy()
    scene_errors, view_errors = np.concatenate(scene_errors), np.concatenate(view_errors)
    metrics = {
        "views": len(views),
        **summary("scene", scene_errors), "scene_absrel": np.concatenate(scene_relative).mean(),
        **summary("view", view_errors), "view_absrel": np.concatenate(view_relative).mean(),
        **summary("points", distance),
        "view_scale_mean": np.mean(view_scales), "view_scale_std": np.std(view_scales),
        "camera_error_mean_m": camera_error.mean(), "camera_error_max_m": camera_error.max(),
    }
    return {name: float(value) for name, value in metrics.items()}
