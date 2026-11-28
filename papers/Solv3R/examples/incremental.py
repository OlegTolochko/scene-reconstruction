"""Incremental reconstruction: new images join an ongoing solve.

The first --initial images are solved for --calls calls. The other images then join in groups of --group, in file
order: their encodings are appended to the state and the solve continues for --calls calls, which also updates the
images solved before. The final solve is saved as <out>/solve.{npz,ply,html}, and with --every-stage each stage also
as <out>/stage_NN.{npz,ply}.

usage: python examples/incremental.py --images path/to/images --out outputs/incremental --initial 8 --group 4
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solv3r import Solv3R, list_images, load_images  # noqa: E402
from solv3r.export import save_solve  # noqa: E402


def solve_in_stages(model, images, stages, calls, out, names, sky=False, every_stage=False):
    """stages: lists of image indices; the first image of the first stage is the reference."""
    os.makedirs(out, exist_ok=True)
    order = []
    for i, idx in enumerate(stages):
        if i == 0:
            ctx = model.encode(images[idx])
            z = model.start(ctx)
        else:
            z, ctx = model.add_views(z, ctx, images[idx])
        order += list(idx)
        z = model.solve(ctx, calls, z)
        if every_stage:
            save_solve(f"{out}/stage_{i:02d}", images[order], model.decode(z, ctx), [names[j] for j in order], sky=sky)
        print(f"stage {i}: {len(order)} images solved")
    save_solve(f"{out}/solve", images[order], model.decode(z, ctx), [names[j] for j in order], sky=sky, html=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="a folder of images or a video file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default=None, help="a .safetensors file or a Hugging Face repo id (default: checkpoints/ if present, else the released weights)")
    ap.add_argument("--initial", type=int, default=8)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--calls", type=int, default=8)
    ap.add_argument("--sky-mask", action="store_true", help="leave the sky out of the point cloud (outdoor captures)")
    ap.add_argument("--every-stage", action="store_true", help="also save the solve after every stage")
    a = ap.parse_args()

    model = Solv3R.from_pretrained(a.checkpoint)
    paths = list_images(a.images)
    n = len(paths)
    stages = [list(range(min(a.initial, n)))] + [list(range(s, min(s + a.group, n))) for s in range(a.initial, n, a.group)]
    solve_in_stages(model, load_images(paths), stages, a.calls, a.out, [os.path.basename(p) for p in paths], a.sky_mask,
                    a.every_stage)


if __name__ == "__main__":
    main()
