import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_DIR = REPO_ROOT / "outputs/unposed_genrecon/scene"  # rgb/ + colmap/ for GenRecon
CONF_THRESHOLD = 2.0  # keep points with confidence >= this (WorldRecon, paper App. B.4)
MAX_POINTS = 300_000  # random subsample written to points3D.txt; None keeps every point
STAT_NB_NEIGHBORS, STAT_STD_RATIO = 20, 3.0  # outlier removal for estimating the alignment only


def unproject(depth, extrinsics, intrinsics):
    """Depth (S, H, W) + OpenCV world-to-camera poses -> world points (S, H, W, 3)."""
    s, h, w = depth.shape
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    fx, fy = intrinsics[:, 0, 0, None, None], intrinsics[:, 1, 1, None, None]
    cx, cy = intrinsics[:, 0, 2, None, None], intrinsics[:, 1, 2, None, None]
    cam = np.stack([(x - cx) / fx * depth, (y - cy) / fy * depth, depth], axis=-1)
    R, t = extrinsics[:, :3, :3], extrinsics[:, :3, 3]
    return np.einsum("sji,shwj->shwi", R, cam - t[:, None, None, :])


def align_rotation(points, extrinsics):
    """Rotation taking up (from near-vertical normals) to +z and the walls onto x/y."""
    cam_up = -extrinsics[:, 1, :3].mean(axis=0)
    up = cam_up / np.linalg.norm(cam_up)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(30))  # neighbour count, so no scale needed
    n = np.asarray(pcd.normals)
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz")
    p.add_argument("--out", type=Path, default=SCENE_DIR, help="Scene folder to write rgb/ and colmap/ into.")
    args = p.parse_args()

    d = np.load(args.npz)
    # VGGT-Omega output written by infer.py (S images of size H x W):
    #   image_paths    (S,)              source image paths
    #   images         (S, 3, H, W)      RGB in [0, 1]
    #   depth          (S, H, W, 1)      depth per pixel, arbitrary scale
    #   depth_conf     (S, H, W)         confidence per depth pixel
    #   extrinsics     (S, 3, 4)         world-to-camera [R | t], OpenCV
    #   intrinsics     (S, 3, 3)         K
    #   camera_tokens  (S, 1, 2048)      unused
    #   registers      (S, 16, 2048)     unused

    images, depth, conf = d["images"], d["depth"], d["depth_conf"]
    extr, intr = d["extrinsics"][:, :3, :4].astype(np.float64), d["intrinsics"].astype(np.float64)
    depth, conf = depth.reshape(depth.shape[:3]), conf.reshape(conf.shape[:3])
    images = images.transpose(0, 2, 3, 1)  # (S, 3, H, W) -> (S, H, W, 3)
    S, H, W = depth.shape

    # Confident points with colours
    pts = unproject(depth, extr, intr).reshape(-1, 3)
    cols = (np.clip(images.reshape(-1, 3), 0, 1) * 255).astype(np.uint8)
    keep = (conf.reshape(-1) >= CONF_THRESHOLD) & np.isfinite(pts).all(1)
    pts, cols = pts[keep], cols[keep]
    if MAX_POINTS is not None and len(pts) > MAX_POINTS:
        idx = np.random.default_rng(0).choice(len(pts), MAX_POINTS, replace=False)
        pts, cols = pts[idx], cols[idx]

    # Ignore floating points when estimating the alignment. All points are still written
    # out: GenRecon cleans the cloud itself, so filtering here too would clean it twice.
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    _, inliers = pcd.remove_statistical_outlier(nb_neighbors=STAT_NB_NEIGHBORS, std_ratio=STAT_STD_RATIO)
    print(f"[convert] {len(inliers)}/{len(pts)} inliers used for alignment")

    # Align up -> +z (GenRecon assumes z up), walls along x/y. Scale is left as VGGT-Omega
    # predicts it, just as GenRecon takes COLMAP's arbitrary scale.
    R_align = align_rotation(pts[inliers], extr)
    pts = pts @ R_align.T
    print(f"[convert] {S} images {W}x{H}, {len(pts)} points")

    # World-to-camera in the rotated frame: X_cam = (R R_align^T) X' + t
    out = args.out
    (out / "rgb").mkdir(parents=True, exist_ok=True)
    (out / "colmap").mkdir(parents=True, exist_ok=True)
    with open(out / "colmap/cameras.txt", "w") as fc, open(out / "colmap/images.txt", "w") as fi:
        for i in range(S):
            name = f"{i:04d}.png"
            img = Image.open(str(d["image_paths"][i])).convert("RGB")
            img.save(out / "rgb" / name)
            # Rescale K to the original image size for GenRecon
            W0, H0 = img.size
            K = intr[i]
            sx, sy = W0 / W, H0 / H
            fc.write(f"{i + 1} PINHOLE {W0} {H0} {K[0, 0] * sx} {K[1, 1] * sy} {K[0, 2] * sx} {K[1, 2] * sy}\n")
            R = extr[i, :, :3] @ R_align.T
            t = extr[i, :, 3]
            qw, qx, qy, qz = Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]]  # COLMAP wants w first
            fi.write(f"{i + 1} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} {i + 1} {name}\n\n")
    with open(out / "colmap/points3D.txt", "w") as f:
        for j, ((x, y, z), (r, g, b)) in enumerate(zip(pts, cols)):
            f.write(f"{j + 1} {x:.5f} {y:.5f} {z:.5f} {r} {g} {b} 0\n")
    print(f"[convert] wrote {out}")


if __name__ == "__main__":
    main()
