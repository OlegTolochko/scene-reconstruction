"""Unposed GenRecon with visibility-weighted aggregation (inject.py) on the ScanNet++ evaluation inputs.

Same inputs, cameras, depth and GenRecon settings as experiments/unposed_genrecon/run_eval.py [--model],
whose process() runs every step (the model, convert.py, GenRecon, alignment of the mesh with the scan); only
GenRecon's entry point is inject.py. --model picks whose cameras GenRecon uses (VGGT-Omega or Solv3R).
Variants:
    depth     v from the model's own depth and confidence
    gt        v from the scan's depth at the same cameras, c = 1 (oracle: perfect cue)
    identity  v = valid, must reproduce unposed GenRecon (checks the patched code path)

Writes <out_root>/<name>[_matched][_<tag>]/<N>views/<scene>/mesh.ply.zst and skips scenes that already have
one. out_root is outputs/genrecon_occlusion_weighting (VGGT-Omega) or ..._solv3r, and name is the cue
(vggt / solv3r / gt) or identity. The cues (cues.py) are built first if missing.

Usage:
    uv run experiments/genrecon_occlusion_weighting/run.py 32 --variant depth [--model solv3r] [--matched] [scene ...]
    uv run scripts/evaluate_scannetpp.py outputs/genrecon_occlusion_weighting/vggt/32views
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

OPTIONS = ("stages", "var_mode", "tau_vox", "rho", "softness", "lam", "eps", "kappa")  # passed on as inject.py's --vis_*


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("views", type=int, help="number of input views")
    parser.add_argument("scenes", nargs="*", help="scene IDs (default: all evaluation scenes)")
    parser.add_argument("--variant", choices=("depth", "gt", "identity"), required=True)
    parser.add_argument("--model", choices=run_eval.MODELS, default="vggt_omega", help="whose cameras and depth")
    parser.add_argument("--matched", action="store_true", help="GenRecon settings matched to the baseline")
    parser.add_argument("--tag", help="suffix for the output folder, e.g. for a parameter sweep")
    parser.add_argument("--stages", help="subset of ss,shape,tex to weight (default: all)")
    parser.add_argument("--var-mode", dest="var_mode", choices=("weighted", "original", "around_vmean"),
                        help="variance the aggregator gets (visibility.py, default: weighted)")
    for name in OPTIONS[2:]:
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=float)
    args = parser.parse_args()

    cue = cues.depth_cue(args.model) if args.variant == "depth" else args.variant
    name = cue + ("_matched" if args.matched else "") + (f"_{args.tag}" if args.tag else "")
    genrecon_args = run_eval.MATCHED if args.matched else run_eval.SMARTPHONE
    options = [x for o in OPTIONS if getattr(args, o) is not None for x in (f"--vis_{o}", getattr(args, o))]
    inputs = ROOT / f"data/scannetpp/eval_inputs/{args.views}views"
    scenes = args.scenes or sorted(p.name for p in inputs.iterdir())

    for scene in scenes:
        out = cues.out_root(args.model) / f"{name}/{args.views}views/{scene}"
        shared = ROOT / f"outputs/unposed_genrecon/{args.views}views/{scene}"  # the model's output
        if (out / "mesh.ply.zst").exists():
            continue
        print(f"{scene}, {args.views} views, {args.model}, {name}", flush=True)
        out.mkdir(parents=True, exist_ok=True)
        try:
            if args.variant == "identity":
                script = [HERE / "inject.py", "--vis_identity"]
            else:
                script = [HERE / "inject.py", "--vis_cues", cues.build(args.views, scene, args.model) / f"{cue}.npz"]
            views, rmse = run_eval.process(inputs / scene, out, shared, genrecon_args, args.model,
                                           script=[*script, *options])
            print(f"  done, {views} views, camera alignment error {rmse:.3f} m", flush=True)
        except Exception as e:  # one bad scene should not stop the rest
            print(f"  failed ({e}), see {out / 'run.log'}", flush=True)


if __name__ == "__main__":
    main()
