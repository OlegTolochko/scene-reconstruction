"""Render an orbiting GIF of an infer.py .npz point cloud.

The camera circles the scene from above and looks at its centre. For indoor scans
the ceiling is cut away so the room is visible, like a dollhouse.

Usage:
    python experiments/vggt_omega_inference/render_gif.py outputs/rgb.npz
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from to_ply import unproject


def look_at(position: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Rows are the camera's right, down and forward axes (OpenCV convention)."""
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(-up, forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down, forward])


def render(points, colors, position, rotation, width, height, focal, splat, background):
    """Draw points as small squares, nearest point wins."""
    cam = (points - position) @ rotation.T
    keep = cam[:, 2] > 1e-3
    cam, col = cam[keep], colors[keep]
    u = (focal * cam[:, 0] / cam[:, 2] + width / 2).astype(np.int32)
    v = (focal * cam[:, 1] / cam[:, 2] + height / 2).astype(np.int32)

    # Far to near, so nearer points are written last and overwrite farther ones.
    order = np.argsort(-cam[:, 2])
    u, v, col = u[order], v[order], col[order]

    offsets = np.arange(splat) - splat // 2
    du, dv = np.meshgrid(offsets, offsets)
    uu = (u[:, None] + du.ravel()[None]).ravel()
    vv = (v[:, None] + dv.ravel()[None]).ravel()
    cc = np.repeat(col, splat * splat, axis=0)
    inside = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)

    image = np.empty((height, width, 3), np.uint8)
    image[:] = background
    image[vv[inside], uu[inside]] = cc[inside]
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", help="Output of infer.py.")
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--size", type=int, nargs=2, default=[640, 480], metavar=("W", "H"))
    parser.add_argument("--elevation", type=float, default=45.0, help="Camera angle above the horizon, degrees.")
    parser.add_argument("--distance", type=float, default=2.8, help="Orbit radius as a multiple of the scene radius.")
    parser.add_argument("--fov", type=float, default=50.0, help="Horizontal field of view, degrees.")
    parser.add_argument("--ceiling-cut", type=float, default=15.0, help="Drop the top N%% of points by height.")
    parser.add_argument("--conf-percentile", type=float, default=30.0, help="Drop the least confident N%% of points.")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output", default=None, help="Default: same name as the .npz, with .gif.")
    args = parser.parse_args()

    d = np.load(args.npz)
    extrinsics = d["extrinsics"]
    points = unproject(d["depth"][..., 0], extrinsics, d["intrinsics"]).reshape(-1, 3)
    colors = (d["images"].transpose(0, 2, 3, 1).reshape(-1, 3).clip(0, 1) * 255).astype(np.uint8)
    conf = d["depth_conf"].reshape(-1)

    mask = np.isfinite(points).all(axis=1) & np.isfinite(conf)
    mask &= conf >= np.percentile(conf[mask], args.conf_percentile)
    points, colors = points[mask], colors[mask]

    # World "up" is the average of the capture cameras' up directions (minus their y axis).
    up = -extrinsics[:, 1, :3].mean(axis=0)
    up /= np.linalg.norm(up)
    height = points @ up
    centre = np.median(points, axis=0)

    # Remove the ceiling and far outliers.
    horizontal = points - centre - np.outer((points - centre) @ up, up)
    radius = np.percentile(np.linalg.norm(horizontal, axis=1), 90)
    keep = (height < np.percentile(height, 100 - args.ceiling_cut)) & (height > np.percentile(height, 1))
    keep &= np.linalg.norm(horizontal, axis=1) < 1.5 * radius
    points, colors = points[keep], colors[keep]
    centre = np.median(points, axis=0)

    # Start the orbit behind the average capture direction.
    forward = extrinsics[:, 2, :3].mean(axis=0)
    first = forward - (forward @ up) * up
    first = first / np.linalg.norm(first) if np.linalg.norm(first) > 1e-6 else np.cross(up, [1.0, 0, 0])
    second = np.cross(up, first)

    # Render at twice the size and downscale, for smoother edges.
    width, height_px = args.size
    ss = 2
    focal = ss * width / 2 / np.tan(np.radians(args.fov) / 2)
    elevation = np.radians(args.elevation)
    frames = []
    for i in range(args.frames):
        angle = 2 * np.pi * i / args.frames
        direction = np.cos(elevation) * (-np.cos(angle) * first - np.sin(angle) * second) + np.sin(elevation) * up
        position = centre + args.distance * radius * direction
        rotation = look_at(position, centre, up)
        image = render(points, colors, position, rotation, ss * width, ss * height_px, focal, splat=3, background=(17, 17, 20))
        frames.append(Image.fromarray(image).resize((width, height_px), Image.LANCZOS))
        print(f"\rFrame {i + 1}/{args.frames}", end="", flush=True)
    print()

    # One shared palette for every frame, so colours don't flicker.
    sample = Image.fromarray(np.concatenate([np.asarray(f) for f in frames[:: max(1, args.frames // 8)]]))
    palette = sample.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    gif = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]

    output = Path(args.output) if args.output else Path(args.npz).with_suffix(".gif")
    gif[0].save(output, save_all=True, append_images=gif[1:], duration=1000 // args.fps, loop=0, optimize=True)
    print(f"Saved {output} ({output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
