"""Turn an infer.py .npz into a coloured point cloud (.ply).

Usage:
    python experiments/vggt_omega_inference/to_ply.py outputs/forest_road.npz
Open the .ply in MeshLab, Blender, CloudCompare, or drag it into https://3dviewer.net.
"""

import argparse
from pathlib import Path

import numpy as np


def unproject(depth: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Depth maps (frames, H, W) -> world points (frames, H, W, 3).

    Extrinsics are camera-from-world in OpenCV coordinates, same as the VGGT-Omega demo.
    """
    frames, height, width = depth.shape
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

    fx = intrinsics[:, 0, 0][:, None, None]
    fy = intrinsics[:, 1, 1][:, None, None]
    cx = intrinsics[:, 0, 2][:, None, None]
    cy = intrinsics[:, 1, 2][:, None, None]
    camera_points = np.stack([(x - cx) / fx * depth, (y - cy) / fy * depth, depth], axis=-1)

    rotation = extrinsics[:, :3, :3]
    translation = extrinsics[:, :3, 3]
    return np.einsum("sij,shwj->shwi", rotation.transpose(0, 2, 1), camera_points - translation[:, None, None, :])


def camera_centres(extrinsics: np.ndarray) -> np.ndarray:
    rotation = extrinsics[:, :3, :3]
    translation = extrinsics[:, :3, 3]
    return -np.einsum("sji,sj->si", rotation, translation)


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    vertex = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertex["x"], vertex["y"], vertex["z"] = points.T
    vertex["red"], vertex["green"], vertex["blue"] = colors.T
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(vertex.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", help="Output of infer.py.")
    parser.add_argument("--conf-percentile", type=float, default=20.0, help="Drop the least confident N%% of points.")
    parser.add_argument("--max-points", type=int, default=2_000_000, help="Randomly subsample to at most this many points.")
    parser.add_argument("--no-cameras", action="store_true", help="Don't mark camera positions in red.")
    parser.add_argument("--output", default=None, help="Default: same name as the .npz, with .ply.")
    args = parser.parse_args()

    d = np.load(args.npz)
    depth = d["depth"][..., 0]
    points = unproject(depth, d["extrinsics"], d["intrinsics"]).reshape(-1, 3)
    colors = (d["images"].transpose(0, 2, 3, 1).reshape(-1, 3).clip(0, 1) * 255).astype(np.uint8)
    conf = d["depth_conf"].reshape(-1)

    mask = np.isfinite(points).all(axis=1) & np.isfinite(conf)
    mask &= conf >= np.percentile(conf[mask], args.conf_percentile)
    points, colors = points[mask], colors[mask]

    if len(points) > args.max_points:
        keep = np.random.default_rng(0).choice(len(points), args.max_points, replace=False)
        points, colors = points[keep], colors[keep]

    if not args.no_cameras:
        # Each camera centre as a small red blob so it's visible in the viewer.
        scale = np.linalg.norm(points - np.median(points, axis=0), axis=1).mean() * 0.005
        offsets = np.random.default_rng(0).normal(scale=scale, size=(200, 3))
        cams = (camera_centres(d["extrinsics"])[:, None, :] + offsets[None]).reshape(-1, 3)
        points = np.concatenate([points, cams])
        colors = np.concatenate([colors, np.tile([[255, 0, 0]], (len(cams), 1)).astype(np.uint8)])

    output = Path(args.output) if args.output else Path(args.npz).with_suffix(".ply")
    write_ply(output, points, colors)
    print(f"Saved {len(points):,} points to {output}")


if __name__ == "__main__":
    main()
