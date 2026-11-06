"""Offline reconstruction: all images are solved together.

Writes <out>/solve.npz (camera-to-world in the frame of the first image, intrinsics, depth, confidence at the model
resolution), <out>/solve.ply and <out>/solve.html (the point cloud in a browser viewer). With --adaptive, calls stop once a call changes the state by less than that fraction.

usage: python examples/offline.py --images path/to/images --out outputs/offline [--calls 16 | --adaptive 0.02]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solv3r import Solv3R, list_images, load_images  # noqa: E402
from solv3r.export import save_solve  # noqa: E402
from solv3r.model import autocast  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="a folder of images or a video file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default=None, help="a .safetensors file or a Hugging Face repo id (default: checkpoints/ if present, else the released weights)")
    ap.add_argument("--calls", type=int, default=16)
    ap.add_argument("--adaptive", type=float, default=None)
    ap.add_argument("--max-calls", type=int, default=64)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--sky-mask", action="store_true", help="leave the sky out of the point cloud (outdoor captures)")
    a = ap.parse_args()

    model = Solv3R.from_pretrained(a.checkpoint)
    paths = list_images(a.images, a.stride)
    images = load_images(paths)
    ctx = model.encode(images)
    z = model.start(ctx)
    with autocast():
        for call in range(1, (a.max_calls if a.adaptive else a.calls) + 1):
            z_next = model.step(z, ctx)
            change = float((z_next.float() - z.float()).norm() / z.float().norm())
            z = z_next
            if a.adaptive and call >= 4 and change < a.adaptive:
                break
    print(f"{len(paths)} images, {call} calls, last relative change {change:.4f}")
    os.makedirs(a.out, exist_ok=True)
    save_solve(f"{a.out}/solve", images, model.decode(z, ctx), [os.path.basename(p) for p in paths], sky=a.sky_mask, html=True)


if __name__ == "__main__":
    main()
