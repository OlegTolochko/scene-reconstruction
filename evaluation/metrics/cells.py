"""Occupied cells of a run (GenRecon's stage-1 cells) against the cells that contain true surface.

The true surface is voxelized on the run's own grid (formats.Cells), into

    all      cells that contain true surface, inside at least one chunk
    visible  cells with true surface that at least one camera sees: in front of it, inside its image, and
             not hidden behind other true surface

and the run's cells are compared with `all`:

    cell_ratio          predicted cells / true cells
    precision           fraction of predicted cells that contain true surface
    recall              fraction of true cells that are predicted
    f1                  harmonic mean of the two
    recall_within_1     fraction of true cells with a predicted cell at most 1 cell away
    pred_1_away, pred_2_away, pred_3plus_away
                        fraction of predicted cells 1, 2, 3 or more cells away from the nearest true cell
    seen_recall         recall of the visible true cells
    unseen_recall       recall of the true cells that are not visible

Distances are in cells, and a diagonal neighbour is 1 cell away.
"""

import numpy as np
import open3d as o3d
import trimesh
from scipy.ndimage import distance_transform_cdt

from evaluation.align import Similarity
from evaluation.formats import Cameras, Cells
from evaluation.metrics.mesh import raycasting_scene, sample_points

POINTS_PER_M2 = 10_000  # true surface samples, about 1 cm apart
OCCLUSION_TOLERANCE = 0.03  # metres, a sample is visible if the first surface hit is at most this much before it


def visible(points: np.ndarray, surface: trimesh.Trimesh, cameras: Cameras) -> np.ndarray:
    """Whether each point on the surface is seen, unoccluded, by at least one camera."""
    scene = raycasting_scene(surface)
    seen = np.zeros(len(points), dtype=bool)
    for w2c, K, width, height in zip(cameras.world_to_camera, cameras.K, cameras.width, cameras.height):
        cam = points @ w2c[:, :3].T + w2c[:, 3]
        z = cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = K[0, 0] * cam[:, 0] / z + K[0, 2]
            v = K[1, 1] * cam[:, 1] / z + K[1, 2]
        candidates = np.flatnonzero((z > 0.05) & (u >= 0) & (u <= width) & (v >= 0) & (v <= height) & ~seen)
        if len(candidates) == 0:
            continue
        centre = -np.linalg.solve(w2c[:, :3], w2c[:, 3])
        direction = points[candidates] - centre
        distance = np.linalg.norm(direction, axis=1)
        rays = np.concatenate([np.broadcast_to(centre, direction.shape), direction / distance[:, None]], axis=1)
        hit = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))["t_hit"].numpy()
        seen[candidates[hit >= distance - OCCLUSION_TOLERANCE]] = True
    return seen


def true_cells(surface: trimesh.Trimesh, grid: Cells, cameras: Cameras, to_grid: Similarity | None = None) -> dict:
    """The `all` and `visible` cells on the grid of `grid` (its own cells are not used).

    surface and cameras are in the dataset's frame, in metres; to_grid maps that frame to the grid's (None: the
    grid is in the dataset's frame). cameras.K must map to cameras.width x cameras.height.
    """
    points, _ = sample_points(surface, int(surface.area * POINTS_PER_M2))
    cells = np.floor(((points if to_grid is None else to_grid.apply(points)) - grid.origin) / grid.cell_size).astype(int)
    inside = grid.inside(cells)
    points, cells = points[inside], cells[inside]
    seen = visible(points, surface, cameras)
    return {"all": np.unique(cells, axis=0), "visible": np.unique(cells[seen], axis=0)}


def compare(pred: np.ndarray, true: np.ndarray, shape: tuple, true_visible: np.ndarray | None = None) -> dict:
    """pred, true, true_visible: (M, 3) cell indices inside a grid of the given shape."""
    pred_grid, true_grid = np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)
    pred_grid[tuple(pred.T)] = True
    true_grid[tuple(true.T)] = True
    # distance of each cell to the nearest cell of the other set
    pred_to_true = distance_transform_cdt(~true_grid, metric="chessboard")[tuple(pred.T)]
    true_to_pred = distance_transform_cdt(~pred_grid, metric="chessboard")[tuple(true.T)]
    precision, recall = (pred_to_true == 0).mean(), (true_to_pred == 0).mean()
    out = {
        "cell_ratio": len(pred) / len(true),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0,
        "recall_within_1": (true_to_pred <= 1).mean(),
        "pred_1_away": (pred_to_true == 1).mean(),
        "pred_2_away": (pred_to_true == 2).mean(),
        "pred_3plus_away": (pred_to_true >= 3).mean(),
    }
    if true_visible is not None:
        seen_grid = np.zeros(shape, dtype=bool)
        seen_grid[tuple(true_visible.T)] = True
        seen = seen_grid[tuple(true.T)]
        hit = pred_grid[tuple(true.T)]
        out["seen_recall"] = hit[seen].mean() if seen.any() else float("nan")
        out["unseen_recall"] = hit[~seen].mean() if (~seen).any() else float("nan")
    return {name: float(value) for name, value in out.items()}
