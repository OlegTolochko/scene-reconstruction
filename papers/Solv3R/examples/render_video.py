"""Renders a saved reconstruction as a short video that swings around the scene.

--input is the prefix of the files an example wrote: <prefix>.ply, and <prefix>.npz for the cameras (optional), e.g.
outputs/offline/solve or outputs/streaming/stream. A .gif or .webp is written with Pillow, an .mp4 with ffmpeg (on the
PATH, or from pip install imageio-ffmpeg).

usage: python examples/render_video.py --input outputs/offline/solve --out outputs/offline/solve.mp4
"""
import argparse
import math
import os
import shutil
import subprocess

import numpy as np
import torch
from PIL import Image, ImageDraw


def read_ply(path):
    """The binary PLY of solv3r.export.write_ply -> xyz (N, 3) float32, rgb (N, 3) uint8."""
    with open(path, "rb") as f:
        n = 0
        while (line := f.readline().decode().strip()) != "end_header":
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
        rec = np.frombuffer(f.read(n * 15), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
    return rec["p"].copy(), rec["c"].copy()


def frame_scene(xyz, c2w, aerial=False):
    """Centre, up, the horizontal direction from the centre towards the cameras, and how far the cameras look down (deg).

    Up is the cameras' average up, or for aerial captures the normal of the dominant plane of the points (towards the
    cameras). Without cameras: the first camera's frame."""
    center = np.median(xyz, 0)
    if c2w is None:
        return center, np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, -1.0]), 0.0
    look, cam_up = c2w[:, :3, 2].mean(0), -c2w[:, :3, 1].mean(0)
    look, cam_up = look / np.linalg.norm(look), cam_up / np.linalg.norm(cam_up)
    up = cam_up
    if aerial:
        sub = xyz[:: max(1, len(xyz) // 50000)]
        up = np.linalg.svd(sub - sub.mean(0), full_matrices=False)[2][-1]
        up = up if up @ (c2w[:, :3, 3].mean(0) - center) > 0 else -up
    back = -look - up * (-look @ up)
    if np.linalg.norm(back) < 0.3:                                  # straight down: stand behind the image's top edge
        back = -cam_up - up * (-cam_up @ up)
    return center, up, back / np.linalg.norm(back), math.degrees(math.asin(float(np.clip(-look @ up, -1, 1))))


def fit_distance(xyz, center, view, up, size, fov, yaws, fill=0.85, q=0.97):
    """Distance from the centre along `view` (rotated by each of `yaws` about up) at which the central q of the points
    stays within `fill` of the frame."""
    sub = xyz[:: max(1, len(xyz) // 50000)] - center
    tx = math.tan(math.radians(fov) / 2) * fill
    ty = tx * size[1] / size[0]
    need = 0.0
    for yaw in yaws:
        fwd = -rotate(view, up, yaw)
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        p = sub @ np.stack([right, np.cross(fwd, right), fwd]).T
        need = max(need, float(np.quantile(np.maximum(np.abs(p[:, 0]) / tx, np.abs(p[:, 1]) / ty) - p[:, 2], q)))
    return need


def rotate(v, axis, deg):
    """Rodrigues rotation of v about a unit axis."""
    t = math.radians(deg)
    return v * math.cos(t) + np.cross(axis, v) * math.sin(t) + axis * (axis @ v) * (1 - math.cos(t))


def point_spacing(xyz, m=200000):
    """Distance between neighbouring points around each point: the median over the cloud (from a subset, rescaled for
    points on surfaces), adjusted by the point count of the surrounding voxel, within 0.5-8 times the median."""
    from scipy.spatial import cKDTree

    sub = xyz[np.random.default_rng(0).choice(len(xyz), min(m, len(xyz)), replace=False)]
    median = float(np.median(cKDTree(sub).query(sub, k=2)[0][:, 1])) * math.sqrt(len(sub) / len(xyz))
    cell = np.floor(xyz / (16 * median)).astype(np.int64)
    cell -= cell.min(0)
    key = (cell[:, 0] << 42) | (cell[:, 1] << 21) | cell[:, 2]
    _, inverse, count = np.unique(key, return_inverse=True, return_counts=True)
    return np.clip(16 * median / np.sqrt(count[inverse]), 0.5 * median, 8 * median).astype(np.float32)


def render(xyz, rgb, spacing, eye, target, up, size, fov, point, edl, background, chunk=2 ** 23):
    """Point splats with a z-buffer at twice the size (the caller averages the frame down).  Each point covers a disc
    `point` times its local point spacing wide at its depth, so surfaces stay closed near and far, and eye-dome lighting
    darkens depth edges.  The points are splatted in chunks, so the memory does not grow with the size of the cloud."""
    W, H = size[0] * 2, size[1] * 2
    f = 0.5 * W / math.tan(math.radians(fov) / 2)
    fwd = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = torch.tensor(np.stack([right, down, fwd]), dtype=torch.float32, device=xyz.device)
    p = (xyz - torch.tensor(eye, dtype=torch.float32, device=xyz.device)) @ R.T
    keep = p[:, 2] > 1e-3
    p, col, spacing = p[keep], rgb[keep], spacing[keep]
    best = torch.full((H * W,), 2 ** 62, dtype=torch.long, device=xyz.device)
    zmax = p[:, 2].max() if len(p) else 1.0
    radius = (0.5 * point * f * spacing / p[:, 2]).round().clamp(1, 12).long()    # splat radius in render pixels
    for r in radius.unique().tolist():
        du, dv = torch.meshgrid(torch.arange(-r, r + 1, device=xyz.device), torch.arange(-r, r + 1, device=xyz.device), indexing="ij")
        disc = (du ** 2 + dv ** 2) <= r * r + 1
        du, dv = du[disc], dv[disc]
        ids = torch.nonzero(radius == r).squeeze(1)
        for s in range(0, len(ids), max(1, chunk // len(du))):
            sub = ids[s:s + max(1, chunk // len(du))]
            q = p[sub]
            u = (f * q[:, 0] / q[:, 2] + W / 2).round().long()
            v = (f * q[:, 1] / q[:, 2] + H / 2).round().long()
            uu, vv = (u[:, None] + du).reshape(-1), (v[:, None] + dv).reshape(-1)
            idx = sub.repeat_interleave(len(du))
            ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            pix, idx = (vv * W + uu)[ok], idx[ok]
            key = (p[idx, 2] / zmax * (2 ** 29 - 1)).long() * 2 ** 32 + idx   # nearest point wins, ties by index
            best.scatter_reduce_(0, pix, key, "amin")
    hit = best < 2 ** 62
    img = torch.tensor(background, dtype=torch.float32, device=xyz.device).repeat(H * W, 1)
    img[hit] = col[best[hit] % 2 ** 32].float()
    if edl > 0:                                                    # eye-dome lighting on the log depth
        logz = torch.full((H * W,), math.log2(float(zmax) * 8), device=xyz.device)
        logz[hit] = torch.log2(p[best[hit] % 2 ** 32, 2])
        logz = logz.reshape(H, W)
        response = sum((logz - torch.roll(logz, sh, dims=(0, 1))).clamp(0, 1) for sh in ((0, 2), (0, -2), (2, 0), (-2, 0)))
        img[hit] *= torch.exp(-2.5 * edl * response.reshape(-1)[hit])[:, None]
    return img.reshape(H, W, 3).round().clamp(0, 255).byte().cpu().numpy(), (R.cpu().numpy(), f)


def draw_cameras(draw, c2w, aspect, eye, R, f, wh, scale, near, path):
    """Frustums (apex at the camera, image rectangle `scale` ahead) of up to 40 evenly spaced cameras, and for a video
    the path through all of them with 20 frustums along it."""
    def project(X):
        q = (np.asarray(X) - eye) @ R.T
        return q[..., 2] > near, np.stack([f * q[..., 0] / q[..., 2] + wh[0] / 2, f * q[..., 1] / q[..., 2] + wh[1] / 2], -1)

    if path:
        ok, uv = project(c2w[:, :3, 3])
        for k in range(len(c2w) - 1):
            if ok[k] and ok[k + 1]:
                draw.line([tuple(uv[k]), tuple(uv[k + 1])], fill=(200, 135, 80), width=2)
    w, h = 0.5 * scale, 0.5 * scale / aspect
    corners = np.array([[-w, -h, scale], [w, -h, scale], [w, h, scale], [-w, h, scale]])
    for k in np.unique(np.linspace(0, len(c2w) - 1, min(len(c2w), 20 if path else 40)).round().astype(int)):
        ok, uv = project(np.vstack([c2w[k, :3, 3], c2w[k, :3, 3] + corners @ c2w[k, :3, :3].T]))
        if ok.all():
            pts = [tuple(x) for x in uv]
            for c in pts[1:]:
                draw.line([pts[0], c], fill=(240, 160, 90), width=4)
            draw.line(pts[1:] + [pts[1]], fill=(240, 160, 90), width=4)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="prefix of the example's output files (<prefix>.ply, <prefix>.npz)")
    ap.add_argument("--out", required=True, help=".mp4, .gif or .webp")
    ap.add_argument("--seconds", type=float, default=8)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--swing", type=float, default=50, help="degrees the view swings around the scene (360: a full turn)")
    ap.add_argument("--elevation", type=float, default=None,
                    help="degrees above the horizon (default: 15 above the cameras' own, between 15 and 60)")
    ap.add_argument("--distance", type=float, default=1.0, help="scales the distance from the scene")
    ap.add_argument("--fov", type=float, default=50)
    ap.add_argument("--point", type=float, default=1.2, help="splat size relative to the spacing of the points")
    ap.add_argument("--edl", type=float, default=1.0, help="eye-dome lighting strength (0: off)")
    ap.add_argument("--background", default="0b0d10")
    ap.add_argument("--cameras", action="store_true", help="draw the cameras (and the path of a video)")
    ap.add_argument("--aerial", action="store_true", help="the cameras look down on the scene (drone photos)")
    a = ap.parse_args()

    ffmpeg = None
    if not a.out.endswith((".gif", ".webp")):                                     # before rendering, not after it
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            try:
                import imageio_ffmpeg
                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                raise SystemExit("an .mp4 needs ffmpeg: install it, or pip install imageio-ffmpeg, or write a .gif")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    xyz, rgb = read_ply(f"{a.input}.ply")
    c2w, aspect, video = None, 4 / 3, False
    if os.path.isfile(f"{a.input}.npz"):
        npz = np.load(f"{a.input}.npz")
        c2w, video = npz["c2w"], "c2w_exit" in npz.files                       # streaming results keep c2w_exit
        if "depth" in npz.files:
            aspect = npz["depth"].shape[2] / npz["depth"].shape[1]
    center, up, back, cam_elevation = frame_scene(xyz, c2w, a.aerial)
    size = tuple(int(s) // 2 * 2 for s in a.size.split("x"))                     # even sides for the video encoder
    elevation = math.radians(a.elevation if a.elevation is not None else min(60.0, max(15.0, cam_elevation + 15.0)))
    view = math.cos(elevation) * back + math.sin(elevation) * up                  # unit, from the centre to the eye
    yaws = np.linspace(0, 360, 9) if a.swing >= 360 else np.linspace(-a.swing / 2, a.swing / 2, 5)
    dist = a.distance * fit_distance(xyz, center, view, up, size, a.fov, yaws)
    spacing = point_spacing(xyz)
    frustum = 0.04 * dist
    if c2w is not None and len(c2w) > 1:                                         # frustums about the camera spacing
        from scipy.spatial import cKDTree

        gap = float(np.median(cKDTree(c2w[:, :3, 3]).query(c2w[:, :3, 3], k=2)[0][:, 1]))
        frustum = min(0.04 * dist, max(0.012 * dist, 0.8 * gap))
    hexa = a.background.lstrip("#")
    bg = [int(hexa[i:i + 2], 16) for i in (0, 2, 4)]
    X, C, spacing = torch.tensor(xyz, device=dev), torch.tensor(rgb, device=dev), torch.tensor(spacing, device=dev)
    n = int(a.seconds * a.fps)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    frames, enc = [], None
    if ffmpeg is not None:                                                        # frames go to the encoder as they are made
        enc = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                                f"{size[0]}x{size[1]}", "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                "-crf", "18", a.out], stdin=subprocess.PIPE)
    for i in range(n):
        t = i / n
        yaw = 360 * t if a.swing >= 360 else 0.5 * a.swing * math.sin(2 * math.pi * t)
        eye = center + rotate(view, up, yaw) * dist
        img, (R, f) = render(X, C, spacing, eye, center, up, size, a.fov, a.point, a.edl, bg)
        frame = Image.fromarray(img)
        if a.cameras and c2w is not None:
            draw_cameras(ImageDraw.Draw(frame), c2w, aspect, eye, R, f, frame.size, frustum, 0.05 * dist, video)
        frame = frame.resize(size, Image.Resampling.BOX)
        if enc is not None:
            enc.stdin.write(frame.tobytes())
        else:
            frames.append(frame)
    if enc is not None:
        enc.stdin.close()
        if enc.wait() != 0:
            raise SystemExit(f"ffmpeg failed writing {a.out}")
    else:
        frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=round(1000 / a.fps), loop=0,
                       **({"quality": 85} if a.out.endswith(".webp") else {}))
    print(f"{a.out}: {n} frames, {len(xyz)} points")


if __name__ == "__main__":
    main()
