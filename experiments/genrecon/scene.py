"""A cameras run as a scene GenRecon's Iphone mode reads: rgb/ and colmap/{cameras,images,points3D}.txt.

The points are the confident depth points (depth.npz), from which GenRecon places its chunks. GenRecon
assumes z points up with the walls along x and y, so the scene is turned that way: one rotation about the
origin, no change of scale or position. The cameras come back turned too; GenRecon's mesh and cells come out
in their frame.
"""

from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from evaluation.datasets import View
from evaluation.formats import Cameras, Depth

CONF_THRESHOLD = 2.0  # keep points with confidence >= this (WorldRecon, paper App. B.4)
MAX_POINTS = 300_000  # random subsample written to points3D.txt
STAT_NB_NEIGHBORS, STAT_STD_RATIO = 20, 3.0  # outlier removal for estimating the rotation only


def depth_points(depth: Depth, cameras: Cameras) -> np.ndarray:
    """The confident depth points of every view, in the cameras' frame."""
    cameras = cameras.subset(depth.names)
    _, height, width = depth.depth.shape
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    points = []
    for d, K, P in zip(depth.depth, cameras.K, cameras.world_to_camera):
        cam = np.stack([(x - K[0, 2]) / K[0, 0] * d, (y - K[1, 2]) / K[1, 1] * d, d], axis=-1)
        points.append(((cam - P[:, 3]) @ P[:, :3]).reshape(-1, 3))
    points = np.concatenate(points)
    points = points[(depth.confidence.reshape(-1) >= CONF_THRESHOLD) & np.isfinite(points).all(axis=1)]
    if len(points) > MAX_POINTS:
        points = points[np.random.default_rng(0).choice(len(points), MAX_POINTS, replace=False)]
    return points


def up_rotation(points: np.ndarray, cameras: Cameras) -> np.ndarray:
    """Rotation taking up (from near-vertical normals) to +z and the walls onto x/y."""
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    _, inliers = pcd.remove_statistical_outlier(nb_neighbors=STAT_NB_NEIGHBORS, std_ratio=STAT_STD_RATIO)
    pcd = pcd.select_by_index(inliers)  # floating points would disturb the normals
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(30))  # neighbour count, so no scale needed
    n = np.asarray(pcd.normals)
    up = -cameras.world_to_camera[:, 1, :3].mean(axis=0)
    up /= np.linalg.norm(up)
    for max_angle in (30, 10):  # coarse, then refine
        d = n @ up
        near = np.abs(d) > np.cos(np.radians(max_angle))
        up = (n[near] * np.sign(d[near])[:, None]).mean(axis=0)
        up /= np.linalg.norm(up)
    R_up = Rotation.align_vectors([[0, 0, 1]], [up])[0].as_matrix()
    walls = n @ R_up.T
    walls = walls[np.abs(walls[:, 2]) < 0.2]
    phi = np.angle(np.exp(4j * np.arctan2(walls[:, 1], walls[:, 0])).mean()) / 4
    return Rotation.from_euler("z", -phi).as_matrix() @ R_up


def write_scene(path: Path, cameras_dir: Path, views: list[View]) -> Cameras:
    """Write the scene of the cameras run in cameras_dir to path; return its cameras as GenRecon sees them."""
    cameras = Cameras.load(cameras_dir / "cameras.json")
    points = depth_points(Depth.load(cameras_dir / "depth.npz"), cameras)
    R = up_rotation(points, cameras)
    turned = Cameras(cameras.names, cameras.K, np.concatenate(
        [cameras.world_to_camera[:, :, :3] @ R.T, cameras.world_to_camera[:, :, 3:]], axis=2),
        cameras.width, cameras.height, cameras.frame)
    by_name = {v.name: v for v in views}
    (path / "rgb").mkdir(parents=True, exist_ok=True)
    (path / "colmap").mkdir(parents=True, exist_ok=True)
    with open(path / "colmap/cameras.txt", "w") as fc, open(path / "colmap/images.txt", "w") as fi:
        for i, (name, K, P, W, H) in enumerate(zip(turned.names, turned.K, turned.world_to_camera, turned.width,
                                                   turned.height), start=1):
            view = by_name[name]
            if not (path / "rgb" / name).exists():
                (path / "rgb" / name).symlink_to(view.image.resolve())
            sx, sy = view.width / W, view.height / H  # K at the image's own size
            fc.write(f"{i} PINHOLE {view.width} {view.height} {K[0, 0] * sx} {K[1, 1] * sy} {K[0, 2] * sx} {K[1, 2] * sy}\n")
            qx, qy, qz, qw = Rotation.from_matrix(P[:, :3]).as_quat()  # COLMAP wants w first
            fi.write(f"{i} {qw} {qx} {qy} {qz} {P[0, 3]} {P[1, 3]} {P[2, 3]} {i} {name}\n\n")
    with open(path / "colmap/points3D.txt", "w") as f:  # GenRecon reads only the positions
        for j, (x, y, z) in enumerate(points @ R.T, start=1):
            f.write(f"{j} {x:.5f} {y:.5f} {z:.5f} 0 0 0 0\n")
    return turned
