"""How far VGGT-Omega's depth at the input views is from the scan.

Compares each pixel with the depth rendered from the scan at the view's true camera. Pixels without scan
surface and anonymized pixels are left out. VGGT-Omega's depth has no absolute scale, so the error depends
on the alignment:

    scene   one scale per scene, from fitting VGGT-Omega's camera centres to the true ones
    view    one scale per view, the median of scan depth / predicted depth: the error of the depth's shape
    points  the depth in 3D with VGGT-Omega's own cameras, moved by the same fit: distance to the scan, so
            the error of the depth and of the cameras together

Usage:
    uv run experiments/vggt_omega_depth/evaluate.py 8 16 32 64 256

writes outputs/vggt_omega_depth/<N>views/<scene>.json (a scene that already has one is not evaluated again)
and prints the mean over the scenes.
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_scannetpp as ev  # noqa: E402

OUT = ROOT / "outputs/vggt_omega_depth"
NUM_POINTS = 200_000  # points per scene for the distance to the scan


def umeyama(x: np.ndarray, y: np.ndarray) -> tuple:
    """Scale s, rotation R and translation t minimising sum |s R x_i + t - y_i|^2."""
    mx, my = x.mean(axis=0), y.mean(axis=0)
    U, D, Vt = np.linalg.svd((y - my).T @ (x - mx) / len(x))
    S = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / ((x - mx) ** 2).sum(axis=1).mean()
    return s, R, my - s * R @ mx


def fit_cameras(x: np.ndarray, y: np.ndarray) -> tuple:
    """umeyama, refitted without cameras far off the fit: one camera can be metres off."""
    keep = np.ones(len(x), dtype=bool)
    for _ in range(3):
        s, R, t = umeyama(x[keep], y[keep])
        error = np.linalg.norm(s * x @ R.T + t - y, axis=1)
        keep = error <= max(3 * np.median(error), 0.05)
    return s, R, t, error


def summary(name: str, errors: np.ndarray) -> dict:
    return {f"{name}_median_cm": np.median(errors) * 100, f"{name}_within_2.5cm": (errors < 0.025).mean(),
            f"{name}_within_5cm": (errors < 0.05).mean(), f"{name}_within_10cm": (errors < 0.10).mean()}


def evaluate(path: Path) -> dict:
    scene_dir = ev.DATA / path.stem
    pred = np.load(path)
    depth, Rs, ts, Ks = pred["depth"], pred["extrinsics"][:, :3, :3], pred["extrinsics"][:, :3, 3], pred["intrinsics"]
    height, width = depth.shape[1:]

    meta = json.loads((scene_dir / "dslr/nerfstudio/transforms_undistorted.json").read_text())
    frames = {f["file_path"]: f for f in meta["frames"]}
    frames = [frames[name] for name in pred["names"]]
    scale_x = width / meta["w"]
    K = np.array([[meta["fl_x"] * scale_x, 0, meta["cx"] * scale_x], [0, meta["fl_y"] * scale_x, meta["cy"] * scale_x], [0, 0, 1]])
    cam_to_world = [ev.NERFSTUDIO_TO_SCAN @ np.array(f["transform_matrix"]) @ ev.OPENGL_TO_OPENCV for f in frames]

    s, R_fit, t_fit, camera_error = fit_cameras(np.einsum("nji,nj->ni", Rs, -ts).astype(np.float64),
                                                np.array([c[:3, 3] for c in cam_to_world]))

    scan = ev.raycasting_scene(ev.load_scan(scene_dir))
    rng = np.random.default_rng(0)
    scene_errors, scene_relative, view_errors, view_relative, view_scales, points = [], [], [], [], [], []
    for i, frame in enumerate(frames):
        truth = ev.render_depth(scan, K, cam_to_world[i], width, height)
        mask = Image.open(scene_dir / "dslr/resized_undistorted_masks" / frame["mask_path"])
        valid = (np.array(mask.resize((width, height), Image.NEAREST)) > 0) & np.isfinite(truth) & (depth[i] > 0)
        if not valid.any():
            continue
        v, u = np.nonzero(valid)
        d, t = s * depth[i][v, u], truth[v, u]
        view_scale = np.median(t / d)
        view_scales.append(view_scale)
        scene_errors.append(np.abs(d - t))
        scene_relative.append(np.abs(d - t) / t)
        view_errors.append(np.abs(view_scale * d - t))
        view_relative.append(np.abs(view_scale * d - t) / t)

        pick = rng.choice(len(u), min(NUM_POINTS // len(frames), len(u)), replace=False)
        u, v, z = u[pick], v[pick], depth[i][v[pick], u[pick]]
        cam = np.stack([(u + 0.5 - Ks[i, 0, 2]) / Ks[i, 0, 0] * z, (v + 0.5 - Ks[i, 1, 2]) / Ks[i, 1, 1] * z, z], axis=1)
        points.append(s * (cam - ts[i]) @ Rs[i] @ R_fit.T + t_fit)

    distance = scan.compute_distance(o3d.core.Tensor(np.concatenate(points).astype(np.float32))).numpy()
    scene_errors, view_errors = np.concatenate(scene_errors), np.concatenate(view_errors)
    metrics = {
        "views": len(frames),
        **summary("scene", scene_errors), "scene_absrel": np.concatenate(scene_relative).mean(),
        **summary("view", view_errors), "view_absrel": np.concatenate(view_relative).mean(),
        **summary("points", distance),
        "view_scale_mean": np.mean(view_scales), "view_scale_std": np.std(view_scales),
        "camera_error_mean_m": camera_error.mean(), "camera_error_max_m": camera_error.max(),
    }
    return {name: float(value) for name, value in metrics.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("views", type=int, nargs="+", help="numbers of input views")
    args = parser.parse_args()

    for n in args.views:
        paths = sorted((OUT / f"{n}views").glob("*.npz"))
        todo = [path for path in paths if not path.with_suffix(".json").exists()]
        # Each scene in a new process: one process does not release all memory between scenes.
        with ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1) as pool:
            for path, metrics in zip(todo, pool.map(evaluate, todo)):
                print(f"evaluated {path}", flush=True)
                path.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")

        results = [json.loads(path.with_suffix(".json").read_text()) for path in paths]
        mean = {name: float(np.mean([r[name] for r in results])) for name in results[0]}
        print(f"{n} views, mean over {len(results)} scenes:")
        print(json.dumps(mean, indent=2))


if __name__ == "__main__":
    main()
