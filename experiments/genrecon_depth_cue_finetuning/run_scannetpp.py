"""Unposed GenRecon on the ScanNet++ eval inputs with a depth-cue aggregator (inject.py) in stage 1.

Reuses unposed_genrecon/run_eval.process() (VGGT-Omega or Solv3R, convert.py, GenRecon, alignment with the
scan) with inject.py as GenRecon's entry point. The cue depth is the same model's depth and raw confidence,
in the units of its cameras, written per scene to <out_root>/depth/<N>views/<scene>/<model>.npz.

    uv run experiments/genrecon_depth_cue_finetuning/run_scannetpp.py 32 --name zero_init --zero-init [scene ...]
    uv run experiments/genrecon_depth_cue_finetuning/run_scannetpp.py 32 --name overfit_cues --run RUN_DIR
    uv run scripts/evaluate_scannetpp.py outputs/genrecon_depth_cue_finetuning/scannetpp/<model>/<name>/32views

--model solv3r (default) uses Solv3R's cameras and depth, the setting of the training data; the baseline
without cues is outputs/unposed_genrecon_solv3r. --zero-init must reproduce that baseline exactly.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "experiments/unposed_genrecon"))
import run_eval  # noqa: E402

OUT = ROOT / "outputs/genrecon_depth_cue_finetuning/scannetpp"


def depth_file(n: int, scene: str, model: str) -> Path:
    out = OUT / f"depth/{n}views/{scene}/{model}.npz"
    if out.exists():
        return out
    src = ROOT / f"outputs/unposed_genrecon/{n}views/{scene}" / run_eval.MODELS[model][2]
    if not src.exists():
        raise FileNotFoundError(f"{src} missing: run unposed_genrecon/run_eval.py {n} --model {model} first")
    d = np.load(src)
    S = d["depth"].shape[0]
    depth = d["depth"].reshape(d["depth"].shape[:3]).astype(np.float32)
    H, W = depth.shape[1:]
    K = d["intrinsics"].astype(np.float32).copy()  # pixels of the depth map -> normalised
    K[:, 0] /= W
    K[:, 1] /= H
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, names=[f"{i:04d}.png" for i in range(S)], depth=depth,  # convert.py's names
                        conf=d["depth_conf"].reshape(depth.shape).astype(np.float32), K=K)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("views", type=int)
    ap.add_argument("scenes", nargs="*", help="default: all eval scenes")
    ap.add_argument("--name", required=True, help="output folder name")
    ap.add_argument("--model", choices=run_eval.MODELS, default="solv3r")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", type=Path, help="train.py output_dir")
    src.add_argument("--zero-init", action="store_true", help="GenRecon's aggregator + zero cue layers")
    ap.add_argument("--step", type=int, help="checkpoint step (default: the last)")
    args = ap.parse_args()

    inputs = ROOT / f"data/scannetpp/eval_inputs/{args.views}views"
    scenes = args.scenes or sorted(p.name for p in inputs.iterdir())
    source = ["--cue_zero_init"] if args.zero_init else ["--cue_run", args.run.resolve()]
    if args.step:
        source += ["--cue_step", args.step]
    for scene in scenes:
        out = OUT / f"{args.model}/{args.name}/{args.views}views/{scene}"
        if (out / "mesh.ply.zst").exists():
            continue
        print(f"{scene}, {args.views} views, {args.model}, {args.name}", flush=True)
        out.mkdir(parents=True, exist_ok=True)
        shared = ROOT / f"outputs/unposed_genrecon/{args.views}views/{scene}"
        try:
            script = [HERE / "inject.py", "--cue_depth", depth_file(args.views, scene, args.model), *source]
            views, rmse = run_eval.process(inputs / scene, out, shared, run_eval.SMARTPHONE, args.model, script=script)
            print(f"  done, {views} views, camera alignment error {rmse:.3f} m", flush=True)
        except Exception as e:  # one bad scene should not stop the rest
            print(f"  failed ({e}), see {out / 'run.log'}", flush=True)


if __name__ == "__main__":
    main()
