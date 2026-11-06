"""Any-order reconstruction: images arrive in a random order and join the ongoing solve.

--initial images spread over the capture are solved first, then the others join in groups of --group in a random
order. The final solve is saved as <out>/solve.{npz,ply,html}, and with --every-stage each stage also as
<out>/stage_NN.{npz,ply}. The views of a solve are in their arrival order (names).

usage: python examples/any_order.py --images path/to/images --out outputs/any_order --initial 8 --group 4 --seed 0
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solv3r import Solv3R, list_images, load_images  # noqa: E402
from incremental import solve_in_stages  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="a folder of images or a video file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default=None, help="a .safetensors file or a Hugging Face repo id (default: checkpoints/ if present, else the released weights)")
    ap.add_argument("--initial", type=int, default=8)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--calls", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sky-mask", action="store_true", help="leave the sky out of the point cloud (outdoor captures)")
    ap.add_argument("--every-stage", action="store_true", help="also save the solve after every stage")
    a = ap.parse_args()

    model = Solv3R.from_pretrained(a.checkpoint)
    paths = list_images(a.images)
    n = len(paths)
    first = sorted(set(np.linspace(0, n - 1, min(a.initial, n)).round().astype(int).tolist()))
    rest = np.random.default_rng(a.seed).permutation([i for i in range(n) if i not in first]).tolist()
    stages = [first] + [rest[s:s + a.group] for s in range(0, len(rest), a.group)]
    solve_in_stages(model, load_images(paths), stages, a.calls, a.out, [os.path.basename(p) for p in paths], a.sky_mask,
                    a.every_stage)


if __name__ == "__main__":
    main()
