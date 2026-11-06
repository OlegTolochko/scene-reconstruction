"""Solv3R on the ScanNet++ evaluation inputs: depth, confidence and cameras of the input views.

Runs in Solv3R's environment, on the scenes that have a GenRecon baseline run. Writes
outputs/solv3r_depth/<N>views/<scene>.npz in the format of experiments/vggt_omega_depth/infer.py, and skips scenes
that already have one:

    names            (S,)         image file names, sorted
    depth            (S, H, W)    depth per pixel along the optical axis, arbitrary scale
    depth_conf       (S, H, W)    confidence per pixel
    extrinsics       (S, 3, 4)    world-to-camera [R | t], OpenCV, fitted to the predicted rays (the paper's protocol)
    intrinsics       (S, 3, 3)    K, in pixels of the depth map, fitted to the predicted rays
    extrinsics_head  (S, 3, 4)    the same from the camera head
    intrinsics_head  (S, 3, 3)

Usage:
    papers/Solv3R/.venv/bin/python experiments/solv3r_depth/infer.py 8 16 32 64 256
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from solv3r import Solv3R, cameras_from_pose_enc, cameras_from_rays, load_images

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "papers/Solv3R/checkpoints/solv3r.safetensors"
OUT = ROOT / "outputs/solv3r_depth"


def world_to_camera(c2w: torch.Tensor) -> np.ndarray:
    return torch.linalg.inv(c2w.double().cpu())[:, :3].float().numpy()


def predict(model: Solv3R, paths: list[Path], calls: int) -> dict:
    images = load_images([str(p) for p in paths])
    ctx = model.encode(images)
    pred = model.decode(model.solve(ctx, calls), ctx)
    c2w, K = cameras_from_rays(pred["rays"], pred["depth_conf"])
    c2w_head, K_head = cameras_from_pose_enc(pred["pose_enc"], (ctx.H, ctx.W))
    to_np = lambda t: t.float().cpu().numpy()  # noqa: E731
    return {"depth": to_np(pred["depth"][..., 0]), "depth_conf": to_np(pred["depth_conf"]),
            "extrinsics": world_to_camera(c2w), "intrinsics": to_np(K),
            "extrinsics_head": world_to_camera(c2w_head), "intrinsics_head": to_np(K_head)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("views", type=int, nargs="+", help="numbers of input views")
    parser.add_argument("--scenes", nargs="*", help="default: every scene with a baseline run")
    parser.add_argument("--calls", type=int, default=16, help="solver calls, 16 in the paper's Table 1")
    args = parser.parse_args()

    # Less memory fragmentation, as in experiments/vggt_omega_depth/infer.py.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    scenes = args.scenes or sorted(p.parent.name for p in (ROOT / "outputs/genrecon_baseline/32views").glob("*/mesh.ply.zst"))
    model = Solv3R.from_pretrained(str(CHECKPOINT))

    for n in args.views:
        (OUT / f"{n}views").mkdir(parents=True, exist_ok=True)
        for scene in scenes:
            path = OUT / f"{n}views/{scene}.npz"
            if path.exists():
                continue
            print(f"{scene}, {n} views", flush=True)
            dslr = ROOT / f"data/scannetpp/eval_inputs/{n}views/{scene}/dslr"
            frames = json.loads((dslr / "nerfstudio/transforms_undistorted.json").read_text())["frames"]
            names = sorted(f["file_path"] for f in frames)
            result = predict(model, [dslr / "resized_undistorted_images" / name for name in names], args.calls)
            np.savez(path, names=np.array(names), **result)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
