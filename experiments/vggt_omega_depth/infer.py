"""VGGT-Omega on the ScanNet++ evaluation inputs: depth, confidence and cameras of the input views.

Runs in VGGT-Omega's environment, on the scenes that have a GenRecon baseline run. Writes
outputs/vggt_omega_depth/<N>views/<scene>.npz and skips scenes that already have one:

    names       (S,)         image file names, sorted
    depth       (S, H, W)    depth per pixel, arbitrary scale
    depth_conf  (S, H, W)    confidence per pixel
    extrinsics  (S, 3, 4)    world-to-camera [R | t], OpenCV, in VGGT-Omega's own world frame
    intrinsics  (S, 3, 3)    K, in pixels of the depth map

Usage:
    papers/VGGT-Omega/.venv/bin/python experiments/vggt_omega_depth/infer.py 8 16 32 64 256
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "papers/VGGT-Omega/checkpoints/VGGT-Omega-1B-512/model.pt"
OUT = ROOT / "outputs/vggt_omega_depth"


def predict(model: VGGTOmega, paths: list[Path]) -> dict:
    images = load_and_preprocess_images([str(p) for p in paths], image_resolution=512).to("cuda")
    with torch.inference_mode():
        predictions = model(images)
    extrinsics, intrinsics = encoding_to_camera(predictions["pose_enc"], predictions["images"].shape[-2:])
    to_np = lambda t: t[0].float().cpu().numpy()  # drop the batch dimension
    return {"depth": to_np(predictions["depth"])[..., 0], "depth_conf": to_np(predictions["depth_conf"]),
            "extrinsics": to_np(extrinsics), "intrinsics": to_np(intrinsics)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("views", type=int, nargs="+", help="numbers of input views")
    parser.add_argument("--scenes", nargs="*", help="default: every scene with a baseline run")
    args = parser.parse_args()

    # Less memory fragmentation: without this, 256 views do not fit in 32 GB.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    scenes = args.scenes or sorted(p.parent.name for p in (ROOT / "outputs/genrecon_baseline/32views").glob("*/mesh.ply.zst"))
    model = VGGTOmega().eval()
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    model = model.to("cuda")

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
            result = predict(model, [dslr / "resized_undistorted_images" / name for name in names])
            np.savez(path, names=np.array(names), **result)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
