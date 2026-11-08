"""Unposed GenRecon with visibility-weighted aggregation (inject.py) on the ScanNet++ evaluation inputs.

Same inputs, VGGT-Omega output and GenRecon settings as experiments/unposed_genrecon/run_eval.py, whose
process() runs every step (convert.py, GenRecon, alignment of the mesh with the scan); only GenRecon's
entry point is inject.py. Variants:
    vggt      v from VGGT-Omega's depth and confidence
    gt        v from the scan's depth at the same cameras, c = 1 (oracle: perfect cue)
    identity  v = valid, must reproduce unposed GenRecon (checks the patched code path)

Writes outputs/genrecon_visibility/<variant>[_matched][_<tag>]/<N>views/<scene>/mesh.ply.zst and skips
scenes that already have one. The cues (cues.py) are built first if missing.

Usage:
    uv run experiments/genrecon_visibility/run.py 32 --variant vggt [--matched] [scene ...]
    uv run scripts/evaluate_scannetpp.py outputs/genrecon_visibility/vggt/32views
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "experiments/unposed_genrecon"))
sys.path.insert(0, str(HERE))
import cues  # noqa: E402
import run_eval  # noqa: E402

OPTIONS = ("stages", "tau_vox", "rho", "softness", "lam", "eps", "kappa")  # passed on as inject.py's --vis_*


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("views", type=int, help="number of input views")
    parser.add_argument("scenes", nargs="*", help="scene IDs (default: all evaluation scenes)")
    parser.add_argument("--variant", choices=("vggt", "gt", "identity"), required=True)
    parser.add_argument("--matched", action="store_true", help="GenRecon settings matched to the baseline")
    parser.add_argument("--tag", help="suffix for the output folder, e.g. for a parameter sweep")
    parser.add_argument("--stages", help="subset of ss,shape,tex to weight (default: all)")
    for name in OPTIONS[1:]:
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=float)
    args = parser.parse_args()

    name = args.variant + ("_matched" if args.matched else "") + (f"_{args.tag}" if args.tag else "")
    genrecon_args = run_eval.MATCHED if args.matched else run_eval.SMARTPHONE
    options = [x for o in OPTIONS if getattr(args, o) is not None for x in (f"--vis_{o}", getattr(args, o))]
    inputs = ROOT / f"data/scannetpp/eval_inputs/{args.views}views"
    scenes = args.scenes or sorted(p.name for p in inputs.iterdir())

    for scene in scenes:
        out = ROOT / f"outputs/genrecon_visibility/{name}/{args.views}views/{scene}"
        shared = ROOT / f"outputs/unposed_genrecon/{args.views}views/{scene}"  # VGGT-Omega output
        if (out / "mesh.ply.zst").exists():
            continue
        print(f"{scene}, {args.views} views, {name}", flush=True)
        out.mkdir(parents=True, exist_ok=True)
        try:
            if args.variant == "identity":
                script = [HERE / "inject.py", "--vis_identity"]
            else:
                cue = cues.build(args.views, scene) / f"{args.variant}.npz"
                script = [HERE / "inject.py", "--vis_cues", cue]
            views, rmse = run_eval.process(inputs / scene, out, shared, genrecon_args, script=[*script, *options])
            print(f"  done, {views} views, camera alignment error {rmse:.3f} m", flush=True)
        except Exception as e:  # one bad scene should not stop the rest
            print(f"  failed ({e}), see {out / 'run.log'}", flush=True)


if __name__ == "__main__":
    main()
