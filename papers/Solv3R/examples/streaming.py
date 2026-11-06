"""Streaming reconstruction of a video: every frame is answered when it arrives.

Frames are read in file-name order. Writes <out>/stream.npz (camera-to-world of every frame on arrival and when it left
the recent window, intrinsics, depth on arrival), <out>/stream.ply (points of the depth maps on arrival) and
<out>/stream.html (the same points in a browser viewer).

usage: python examples/streaming.py --frames path/to/frames --out outputs/streaming [--config L8 | L4 | S4]
"""
import argparse
import dataclasses
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solv3r import PRESETS, Solv3R, StreamingSolver, list_images, load_images  # noqa: E402
from solv3r.export import confident_points, sky_mask, unproject, write_html, write_ply  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help="a folder of video frames or a video file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default=None, help="a .safetensors file or a Hugging Face repo id (default: checkpoints/ if present, else the released weights)")
    ap.add_argument("--config", default="L8", choices=("L8", "L4", "S4"))
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--keep", type=float, default=0.3, help="fraction of the most confident points per frame")
    ap.add_argument("--sky-mask", action="store_true", help="leave the sky out of the point cloud (outdoor captures)")
    a = ap.parse_args()

    model = Solv3R.from_pretrained(a.checkpoint)
    paths = list_images(a.frames, a.stride, a.max_frames)
    size = load_images(paths[:1]).shape[-2:]
    frames = (load_images([p], size)[0] for p in paths)   # read one at a time: long videos do not fill the memory
    cfg = PRESETS[a.config]                             # memory frames spread over the whole sequence
    cfg = dataclasses.replace(cfg, memory_stride=max(cfg.memory_stride, -(-len(paths) // cfg.memory)))
    solver = StreamingSolver(model, cfg, depth=True)
    for i, frame in enumerate(frames):
        answered = solver.push(frame)
        if answered and i % 50 == 0:
            print(f"frame {i}: answered {answered}")
    res = solver.result()
    print(f"{len(paths)} frames, {res.ms_per_frame:.1f} ms per frame (solver), memory refreshed at "
          f"{sum(res.refreshed)} frames")
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(f"{a.out}/stream.npz", c2w=res.c2w, c2w_exit=res.c2w_exit, K=res.intrinsics,
                        depth=res.depth.astype(np.float16), depth_conf=res.depth_conf.astype(np.float16),
                        names=np.array([os.path.basename(p) for p in paths]))
    every = max(1, -(-len(paths) * size[0] * size[1] * a.keep // 20e6))   # at most ~20M points in the exported cloud
    idx = list(range(0, len(paths), int(every)))
    frames = load_images([paths[i] for i in idx], size)
    points = unproject(res.depth[idx], res.c2w[idx], res.intrinsics[idx])
    valid = ~sky_mask(frames) if a.sky_mask else None
    xyz, rgb = confident_points(points, res.depth_conf[idx], frames, a.keep, valid)
    write_ply(f"{a.out}/stream.ply", xyz, rgb)
    write_html(f"{a.out}/stream.html", xyz, rgb, res.c2w, title="stream")


if __name__ == "__main__":
    main()
