import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_DIR = REPO_ROOT / "outputs/unposed_genrecon/scene"  # rgb/ + colmap/ for GenRecon
ROOM_HEIGHT = 2.9  # floor-to-ceiling height in metres the scene is scaled to
CONF_THRESHOLD = 2.0  # keep points with confidence >= this (WorldRecon, paper App. B.4)
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
    """Rotation taking the mean up direction of the cameras to +z"""
    # TODO: this is not robust - WorldRecon uses a more sophisticated method to find the up direction we should probably do the same
    cam_up = -extrinsics[:, 1, :3].mean(axis=0) # Mean up direction in camera 1 coordinates 
    cam_up /= np.linalg.norm(cam_up) # noramlize
    _, _, axes = np.linalg.svd(points - points.mean(axis=0), full_matrices=False) # PCA to find the principal axes of the point cloud
    up = axes[np.argmax(np.abs(axes @ cam_up))] # up direction is most similar to the camera up direction 
    up *= np.sign(up @ cam_up)
    R_up = Rotation.align_vectors([[0, 0, 1]], [up])[0].as_matrix()
    return R_up


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz")
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

    # Align up -> +z, walls along x/y, then scale the room height to ROOM_HEIGHT (no translation, like WorldRecon)
    R_align = align_rotation(pts, extr)
    pts = pts @ R_align.T
    z_floor, z_ceil = np.percentile(pts[:, 2], 0.2), np.percentile(pts[:, 2], 99.97)
    scale = ROOM_HEIGHT / (z_ceil - z_floor)
    pts = pts * scale
    print(f"[convert] {S} images {W}x{H}, {len(pts)} points, scale {scale:.3f}")

    # World-to-camera in the new frame: X_cam' = R R_align^T X' + s t
    out = SCENE_DIR
    (out / "rgb").mkdir(parents=True, exist_ok=True)
    (out / "colmap").mkdir(parents=True, exist_ok=True)
    with open(out / "colmap/cameras.txt", "w") as fc, open(out / "colmap/images.txt", "w") as fi:
        for i in range(S):
            name = f"{i:04d}.png"
            Image.open(str(d["image_paths"][i])).convert("RGB").save(out / "rgb" / name)
            # Rescale K to the original image size for GenRecon
            W0, H0 = Image.open(str(d["image_paths"][i])).size
            K = intr[i]
            sx, sy = W0 / W, H0 / H
            fc.write(f"{i + 1} PINHOLE {W0} {H0} {K[0, 0] * sx} {K[1, 1] * sy} {K[0, 2] * sx} {K[1, 2] * sy}\n")
            R = extr[i, :, :3] @ R_align.T
            t = scale * extr[i, :, 3]
            qw, qx, qy, qz = Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]]  # COLMAP wants w first
            fi.write(f"{i + 1} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} {i + 1} {name}\n\n")
    with open(out / "colmap/points3D.txt", "w") as f:
        for j, ((x, y, z), (r, g, b)) in enumerate(zip(pts, cols)):
            f.write(f"{j + 1} {x:.5f} {y:.5f} {z:.5f} {r} {g} {b} 0\n")
    print(f"[convert] wrote {out}")


if __name__ == "__main__":
    main()
