"""Solv3R's camera poses for the ScanNet++ evaluation inputs, the same views VGGT-Omega got.

Runs in Solv3R's environment. For every scene in data/scannetpp/eval_inputs/<N>views/, writes
outputs/solv3r_poses/<N>views/<scene>.npz and skips scenes that already have one:

    names            (S,)         image file names, sorted (the order VGGT-Omega saw them in)
    extrinsics       (S, 3, 4)    world-to-camera [R | t], OpenCV, fitted to the predicted rays (the paper's protocol)
    intrinsics       (S, 3, 3)    K, in pixels of Solv3R's depth map, fitted to the predicted rays
    extrinsics_head  (S, 3, 4)    the same from the camera head
    intrinsics_head  (S, 3, 3)

A scene that runs out of GPU memory is skipped and listed in outputs/solv3r_poses/<N>views/oom.txt; a rerun tries
it again.

Usage:
    papers/Solv3R/.venv/bin/python experiments/solv3r_poses/infer.py 32
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
OUT = ROOT / "outputs/solv3r_poses"


def world_to_camera(c2w: torch.Tensor) -> np.ndarray:
    return torch.linalg.inv(c2w.double().cpu())[:, :3].float().numpy()


def predict(model: Solv3R, paths: list[Path], calls: int) -> dict:
    images = load_images([str(p) for p in paths])
    ctx = model.encode(images)  # the first image defines the world frame, as for VGGT-Omega
    pred = model.decode(model.solve(ctx, calls), ctx)
    c2w, K = cameras_from_rays(pred["rays"], pred["depth_conf"])
    c2w_head, K_head = cameras_from_pose_enc(pred["pose_enc"], (ctx.H, ctx.W))
    return {"extrinsics": world_to_camera(c2w), "intrinsics": K.float().cpu().numpy(),
            "extrinsics_head": world_to_camera(c2w_head), "intrinsics_head": K_head.float().cpu().numpy()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("views", type=int, nargs="+", help="numbers of input views")
    parser.add_argument("--scenes", nargs="*", help="default: every scene in data/scannetpp/eval_inputs/<N>views")
    parser.add_argument("--calls", type=int, default=16, help="solver calls, 16 in the paper's Table 1")
    args = parser.parse_args()

    # Less memory fragmentation, as in experiments/vggt_omega_depth/infer.py.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    model = Solv3R.from_pretrained(str(CHECKPOINT))

    for n in args.views:
        inputs = ROOT / f"data/scannetpp/eval_inputs/{n}views"
        out = OUT / f"{n}views"
        out.mkdir(parents=True, exist_ok=True)
        oom = []
        for scene in args.scenes or sorted(p.name for p in inputs.iterdir() if p.is_dir()):
            path = out / f"{scene}.npz"
            if path.exists():
                continue
            print(f"{scene}, {n} views", flush=True)
            dslr = inputs / scene / "dslr"
            frames = json.loads((dslr / "nerfstudio/transforms_undistorted.json").read_text())["frames"]
            names = sorted(Path(f["file_path"]).name for f in frames)
            try:
                result = predict(model, [dslr / "resized_undistorted_images" / name for name in names], args.calls)
            except torch.OutOfMemoryError:
                print("  out of GPU memory, skipped", flush=True)
                oom.append(scene)
                torch.cuda.empty_cache()
                continue
            np.savez(path, names=np.array(names), **result)
            torch.cuda.empty_cache()
        (out / "oom.txt").write_text("".join(f"{s}\n" for s in oom))
        print(f"{n} views: {len(oom)} scenes out of memory", flush=True)


if __name__ == "__main__":
    main()
