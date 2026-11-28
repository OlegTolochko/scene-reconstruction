"""VGGT-Omega's cameras and depth for a list of scenes. Runs in VGGT-Omega's env; experiment.py starts it.

    papers/VGGT-Omega/.venv/bin/python experiments/vggt_omega/infer.py JOBS.json

JOBS.json lists {"images": [...], "names": [...], "out": scene folder}. For each scene this writes
cameras.json (frame arbitrary, the first image's camera defines it) and depth.npz with VGGT-Omega's
confidence, both at the network's resolution (512 pixels on the long side of the input). The model is
loaded once for all scenes; a scene that fails is reported and skipped.
"""

import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluation.formats import Cameras, Depth  # noqa: E402

CHECKPOINT = ROOT / "papers/VGGT-Omega/checkpoints/VGGT-Omega-1B-512/model.pt"


def main():
    jobs = json.loads(Path(sys.argv[1]).read_text())
    model = VGGTOmega().eval()
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    model = model.to("cuda")

    for job in jobs:
        out = Path(job["out"])
        print(f"{out.name}, {len(job['names'])} views", flush=True)
        try:
            images = load_and_preprocess_images(job["images"], image_resolution=512).to("cuda")
            with torch.inference_mode():
                pred = model(images)
            height, width = pred["images"].shape[-2:]
            extrinsics, intrinsics = encoding_to_camera(pred["pose_enc"], (height, width))
            to_np = lambda t: t[0].float().cpu().numpy()  # noqa: E731 (drop the batch dimension)
            S = len(job["names"])
            Depth(job["names"], to_np(pred["depth"])[..., 0], to_np(pred["depth_conf"])).save(out / "depth.npz")
            Cameras(job["names"], to_np(intrinsics).astype(float), to_np(extrinsics).astype(float),
                    np.full(S, width), np.full(S, height), "arbitrary").save(out / "cameras.json")
        except Exception:
            (out / "intermediate").mkdir(parents=True, exist_ok=True)
            (out / "intermediate/error.txt").write_text(traceback.format_exc())
            print(f"  failed, see {out / 'intermediate/error.txt'}", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
