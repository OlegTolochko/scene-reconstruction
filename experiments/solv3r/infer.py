"""Solv3R's cameras and depth for a list of scenes. Runs in Solv3R's env; experiment.py starts it.

    papers/Solv3R/.venv/bin/python experiments/solv3r/infer.py JOBS.json CALLS

JOBS.json lists {"images": [...], "names": [...], "out": scene folder}. For each scene this writes
cameras.json (frame arbitrary, the first image's camera defines it) and depth.npz with Solv3R's confidence,
both at Solv3R's resolution (504 pixels on the long side). The cameras are fitted to the predicted rays, as
in Solv3R's paper; CALLS is the number of solver calls (16 in the paper's Table 1). The model is loaded once
for all scenes; a scene that fails is reported and skipped.
"""

import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from solv3r import Solv3R, cameras_from_rays, load_images

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluation.formats import Cameras, Depth  # noqa: E402

CHECKPOINT = ROOT / "papers/Solv3R/checkpoints/solv3r.safetensors"


def main():
    jobs, calls = json.loads(Path(sys.argv[1]).read_text()), int(sys.argv[2])
    model = Solv3R.from_pretrained(str(CHECKPOINT))

    for job in jobs:
        out = Path(job["out"])
        print(f"{out.name}, {len(job['names'])} views", flush=True)
        try:
            ctx = model.encode(load_images(job["images"]))
            pred = model.decode(model.solve(ctx, calls), ctx)
            cam_to_world, K = cameras_from_rays(pred["rays"], pred["depth_conf"])
            world_to_camera = torch.linalg.inv(cam_to_world.double().cpu())[:, :3].numpy()
            to_np = lambda t: t.float().cpu().numpy()  # noqa: E731
            S = len(job["names"])
            Depth(job["names"], to_np(pred["depth"][..., 0]), to_np(pred["depth_conf"])).save(out / "depth.npz")
            Cameras(job["names"], to_np(K).astype(float), world_to_camera, np.full(S, ctx.W), np.full(S, ctx.H),
                    "arbitrary").save(out / "cameras.json")
        except Exception:
            (out / "intermediate").mkdir(parents=True, exist_ok=True)
            (out / "intermediate/error.txt").write_text(traceback.format_exc())
            print(f"  failed, see {out / 'intermediate/error.txt'}", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
