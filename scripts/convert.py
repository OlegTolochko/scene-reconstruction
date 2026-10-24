"""Turn a VGGT-Omega .npz (from experiments/vggt_omega_inference/infer.py) into the
scene folder GenRecon reads with --mode Iphone:

    scene/rgb/0000.png ...
    scene/colmap/{cameras,images,points3D}.txt

The world is rotated so the floor is flat with +z up, and scaled so the room height
(as GenRecon measures it) equals --room-height metres.
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image


def unproject(depth, extrinsics, intrinsics):
    """Depth (S, H, W) + OpenCV world-to-camera poses -> world points (S, H, W, 3)."""
    s, h, w = depth.shape
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    fx, fy = intrinsics[:, 0, 0, None, None], intrinsics[:, 1, 1, None, None]
    cx, cy = intrinsics[:, 0, 2, None, None], intrinsics[:, 1, 2, None, None]
    cam = np.stack([(x - cx) / fx * depth, (y - cy) / fy * depth, depth], axis=-1)
    R, t = extrinsics[:, :3, :3], extrinsics[:, :3, 3]
    return np.einsum("sji,shwj->shwi", R, cam - t[:, None, None, :])


def rotation_to_z(up):
    """Rotation matrix that maps the unit vector `up` onto +z."""
    up = up / np.linalg.norm(up)
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(up, z), float(up @ z)
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def estimate_up(points, extrinsics):
    """Up direction: average camera 'up' (OpenCV y points down), refined by a floor plane fit."""
    up = -extrinsics[:, 1, :3].mean(axis=0)
    up /= np.linalg.norm(up)
    height = points @ up
    low = points[height < np.percentile(height, 20)]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(low))
    extent = np.percentile(height, 95) - np.percentile(height, 5)
    (a, b, c, _), _ = pcd.segment_plane(distance_threshold=0.01 * extent, ransac_n=3, num_iterations=2000)
    normal = np.array([a, b, c]) / np.linalg.norm([a, b, c])
    normal *= np.sign(normal @ up)
    if normal @ up > np.cos(np.radians(30)):  # trust the floor only if it roughly agrees
        return normal
    print("[convert] no clear floor found, using camera up direction")
    return up


def rotmat_to_quat(R):
    """3x3 rotation -> COLMAP quaternion (qw, qx, qy, qz)."""
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return w, x, y, z


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz")
    p.add_argument("scene_dir")
    p.add_argument("--room-height", type=float, default=2.7, help="Floor-to-ceiling height in metres.")
    p.add_argument("--conf-percentile", type=float, default=30.0, help="Drop the least confident N%% of points.")
    p.add_argument("--max-points", type=int, default=300_000)
    args = p.parse_args()

    d = np.load(args.npz)
    images, depth, conf = d["images"], d["depth"], d["depth_conf"]
    extr, intr = d["extrinsics"][:, :3, :4].astype(np.float64), d["intrinsics"].astype(np.float64)
    depth, conf = depth.reshape(depth.shape[:3]), conf.reshape(conf.shape[:3])
    if images.shape[1] == 3:  # (S, 3, H, W) -> (S, H, W, 3)
        images = images.transpose(0, 2, 3, 1)
    S, H, W = depth.shape

    # Confident points with colours
    pts = unproject(depth, extr, intr).reshape(-1, 3)
    cols = (np.clip(images.reshape(-1, 3), 0, 1) * 255).astype(np.uint8)
    keep = (conf.reshape(-1) > np.percentile(conf, args.conf_percentile)) & np.isfinite(pts).all(1)
    pts, cols = pts[keep], cols[keep]
    if len(pts) > args.max_points:
        idx = np.random.default_rng(0).choice(len(pts), args.max_points, replace=False)
        pts, cols = pts[idx], cols[idx]

    # Align: up -> +z, floor at z=0, room height -> metres (same percentiles as GenRecon)
    R_align = rotation_to_z(estimate_up(pts, extr))
    pts = pts @ R_align.T
    z_floor, z_ceil = np.percentile(pts[:, 2], 0.2), np.percentile(pts[:, 2], 99.97)
    scale = args.room_height / (z_ceil - z_floor)
    offset = np.array([0.0, 0.0, z_floor])
    pts = (pts - offset) * scale
    print(f"[convert] {S} images {W}x{H}, {len(pts)} points, scale {scale:.3f}")

    # World-to-camera in the new frame: X_cam' = R R_align^T X' + s (t + R R_align^T offset)
    out = Path(args.scene_dir)
    (out / "rgb").mkdir(parents=True, exist_ok=True)
    (out / "colmap").mkdir(parents=True, exist_ok=True)
    with open(out / "colmap/cameras.txt", "w") as fc, open(out / "colmap/images.txt", "w") as fi:
        for i in range(S):
            name = f"{i:04d}.png"
            Image.fromarray((np.clip(images[i], 0, 1) * 255).astype(np.uint8)).save(out / "rgb" / name)
            K = intr[i]
            fc.write(f"{i + 1} PINHOLE {W} {H} {K[0, 0]} {K[1, 1]} {K[0, 2]} {K[1, 2]}\n")
            R = extr[i, :, :3] @ R_align.T
            t = scale * (extr[i, :, 3] + R @ offset)
            qw, qx, qy, qz = rotmat_to_quat(R)
            fi.write(f"{i + 1} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} {i + 1} {name}\n\n")
    with open(out / "colmap/points3D.txt", "w") as f:
        for j, ((x, y, z), (r, g, b)) in enumerate(zip(pts, cols)):
            f.write(f"{j + 1} {x:.5f} {y:.5f} {z:.5f} {r} {g} {b} 0\n")
    print(f"[convert] wrote {out}")


if __name__ == "__main__":
    main()
