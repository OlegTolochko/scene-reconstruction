import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial.transform import Rotation

ROOM_HEIGHT = 2.9  # floor-to-ceiling height in metres the scene is scaled to
CONF_PERCENTILE = 30.0  # drop the least confident N% of points
MAX_POINTS = 300_000  # random subsample written to points3D.txt; None keeps every point


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
    """Rotation taking the scene to WorldRecon's frame (paper App. B.4): gravity up -> +z, walls along x/y."""
    # 1. Initial up: the PCA axis closest to the average camera up (OpenCV camera y points down)
    cam_up = -extrinsics[:, 1, :3].mean(axis=0)
    cam_up /= np.linalg.norm(cam_up)
    _, _, axes = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
    up = axes[np.argmax(np.abs(axes @ cam_up))]
    up *= np.sign(up @ cam_up)

    # Normals on a voxel-downsampled copy (the paper doesn't give its normal settings)
    size = np.linalg.norm(np.percentile(points, 98, axis=0) - np.percentile(points, 2, axis=0))
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).voxel_down_sample(size / 200)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(30))
    normals = np.asarray(pcd.normals)

    # 2. Refine up: sign-consistent average of near-horizontal surface normals, weighted by |n . up|
    dots = normals @ up
    horizontal = np.abs(dots) > 0.7
    if horizontal.any():
        up = (dots[horizontal, None] * normals[horizontal]).sum(axis=0)  # sign(d) * |d| * n = d * n
        up /= np.linalg.norm(up)
    R_up = Rotation.align_vectors([[0, 0, 1]], [up])[0].as_matrix()

    # 3. Yaw: wall normals vote for their horizontal angle modulo 90 degrees, weighted by horizontal
    #    magnitude, smoothed with a periodic Gaussian (bin count and width are our choice)
    normals = normals @ R_up.T
    wall = normals[np.abs(normals[:, 2]) < 0.2, :2]
    if len(wall) == 0:
        print("[convert] no walls found, skipping yaw alignment")
        return R_up
    bins, sigma = 360, np.radians(2.0)
    width = (np.pi / 2) / bins
    angles = np.arctan2(wall[:, 1], wall[:, 0]) % (np.pi / 2)
    hist, _ = np.histogram(angles, bins=bins, range=(0, np.pi / 2), weights=np.linalg.norm(wall, axis=1))
    k = np.arange(bins)
    kernel = np.exp(-0.5 * (np.minimum(k, bins - k) * width / sigma) ** 2)
    votes = np.fft.ifft(np.fft.fft(hist) * np.fft.fft(kernel)).real  # circular convolution
    phi = (np.argmax(votes) + 0.5) * width
    R_yaw = Rotation.from_euler("z", -phi).as_matrix()
    return R_yaw @ R_up


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz")
    p.add_argument("scene_dir")
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
    keep = (conf.reshape(-1) > np.percentile(conf, CONF_PERCENTILE)) & np.isfinite(pts).all(1)
    pts, cols = pts[keep], cols[keep]
    if MAX_POINTS is not None and len(pts) > MAX_POINTS:
        idx = np.random.default_rng(0).choice(len(pts), MAX_POINTS, replace=False)
        pts, cols = pts[idx], cols[idx]

    # Align up -> +z, walls along x/y, then floor at z=0
    R_align = align_rotation(pts, extr)
    pts = pts @ R_align.T
    z_floor, z_ceil = np.percentile(pts[:, 2], 0.2), np.percentile(pts[:, 2], 99.97)
    scale = ROOM_HEIGHT / (z_ceil - z_floor)
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
            qw, qx, qy, qz = Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]]  # COLMAP wants w first
            fi.write(f"{i + 1} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} {i + 1} {name}\n\n")
    with open(out / "colmap/points3D.txt", "w") as f:
        for j, ((x, y, z), (r, g, b)) in enumerate(zip(pts, cols)):
            f.write(f"{j + 1} {x:.5f} {y:.5f} {z:.5f} {r} {g} {b} 0\n")
    print(f"[convert] wrote {out}")


if __name__ == "__main__":
    main()
