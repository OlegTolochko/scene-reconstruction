"""Which variance separates stage 1's false cells from its true ones? A check before running --var-mode.

For each scene, runs GenRecon up to stage 1's conditioning (inject.py --vis_dump, no sampling) and takes,
for every cell of the global sparse-structure grid, the variance the aggregator would get in each mode of
visibility.py (mean over channels):

    original      GenRecon's (what the baseline's MLPs saw)
    weighted      the current method's, over the v-weighted views
    around_vmean  over the valid views, around the v-weighted mean

The baseline's stage 1 cells (experiments/unposed_genrecon) are labelled with the scan, which align.json
moves into GenRecon's frame: true if a cell contains scan surface, false if the nearest scan cell is 2 or more
cells away (cells 1 away are left out, they are neither clearly right nor clearly wrong). A high variance
should mean empty, so

    auroc         P(variance of a false cell > variance of a true cell), 0.5 = no separation
    auroc_hidden  the same over the cells where the cue down-weights at least one view's worth of weight
                  (sum of valid - v >= 1). Elsewhere the three modes are almost the same

If around_vmean separates better than original, a full run with --var-mode around_vmean is worth doing. If
weighted separates worse than original, weighting the variance pushed false cells towards "surface".

Writes outputs/genrecon_occlusion_weighting/variance_check/<cue>[_matched]/<N>views/<scene>/{variances.npz,
check.json} (skips scenes that have them) and prints the median over the scenes.

Usage:
    uv run experiments/genrecon_occlusion_weighting/variance_check.py 32 [scene ...] [--cue gt] [--matched]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_cdt
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "experiments/unposed_genrecon"),
                str(ROOT / "experiments/genrecon_oracle_occupancy"), str(HERE)]
import cues  # noqa: E402
import evaluate_scannetpp as ev  # noqa: E402
import run_eval  # noqa: E402
from occupancy import POINTS_PER_M2, R, layout, predicted_cells  # noqa: E402

OUT = ROOT / "outputs/genrecon_occlusion_weighting/variance_check"
MODES = ("original", "weighted", "around_vmean")  # visibility.VAR_MODES, GenRecon's first


def auroc(neg: np.ndarray, pos: np.ndarray) -> float:
    """P(pos > neg), ties counted half (Mann-Whitney U)."""
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    ranks = rankdata(np.concatenate([neg, pos]))
    return float((ranks[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def scan_distance(scene: str, base: Path, lay: dict, cells: np.ndarray) -> np.ndarray:
    """Chessboard distance in cells from each cell to the nearest cell with scan surface."""
    scan = ev.load_scan(ev.DATA / scene)
    points, _ = ev.sample_points(scan, int(scan.area * POINTS_PER_M2))
    a = json.loads((base / "align.json").read_text())  # GenRecon frame -> scan: s R x + t
    points = (points - np.array(a["translation"])) @ np.array(a["rotation"]) / a["scale"]
    occupied = np.floor((points - lay["origin"]) / lay["cell"]).astype(int)
    occupied = occupied[((occupied >= 0) & (occupied < lay["shape"])).all(axis=1)]
    grid = np.ones(lay["shape"], dtype=bool)
    grid[tuple(occupied.T)] = False
    return distance_transform_cdt(grid, metric="chessboard")[tuple(cells.T)]


def at_cells(values: np.ndarray, dump, cells: np.ndarray) -> np.ndarray:
    """values on the dumped grid (R_dump per chunk), read at stage 1's cells (R per chunk)."""
    grid, r = tuple(dump["grid"]), int(dump["R"])
    values = values.reshape(grid)
    if r <= R:
        return values[tuple((cells * r // R).T)]
    f = r // R  # each cell covers f^3 cells of the dumped grid: average them
    return np.mean([values[tuple((cells * f + np.array([i, j, k])).T)]
                    for i in range(f) for j in range(f) for k in range(f)], axis=0)


def check(scene: str, base: Path, recon: Path, dump) -> dict:
    lay = layout(recon)
    if not np.allclose(lay["centres"], layout(base / "genrecon")["centres"]):
        raise RuntimeError("the chunk layout differs from the baseline run's")
    if int(dump["R"]) not in (R // 2, R, 2 * R):
        raise RuntimeError(f"dumped grid has {int(dump['R'])} cells per chunk, stage 1 {R}")
    cells = predicted_cells(base / "genrecon", lay)
    dist = scan_distance(scene, base, lay, cells)
    true, false = dist == 0, dist >= 2
    hidden = at_cells(dump["sum_valid"] - dump["sum_v"], dump, cells) >= 1
    result = {"cells": len(cells), "true": int(true.sum()), "false": int(false.sum()),
              "hidden_fraction": float(hidden[true | false].mean())}
    for mode in MODES:
        var = at_cells(dump[mode], dump, cells)
        result[mode] = {"auroc": auroc(var[true], var[false]),
                        "auroc_hidden": auroc(var[true & hidden], var[false & hidden]),
                        "median_true": float(np.median(var[true])), "median_false": float(np.median(var[false]))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("views", type=int, help="number of input views")
    parser.add_argument("scenes", nargs="*", help="scene IDs (default: every scene with a baseline run)")
    parser.add_argument("--cue", choices=("gt", "vggt"), default="gt", help="depth for v (default: the scan)")
    parser.add_argument("--matched", action="store_true", help="GenRecon settings matched to the baseline")
    args = parser.parse_args()

    suffix = "_matched" if args.matched else ""
    baselines = ROOT / f"outputs/unposed_genrecon{suffix}/{args.views}views"
    scenes = args.scenes or sorted(p.parent.name for p in baselines.glob("*/mesh.ply.zst"))
    genrecon_args = run_eval.MATCHED if args.matched else run_eval.SMARTPHONE
    results = {}
    for scene in scenes:
        base, out = baselines / scene, OUT / f"{args.cue}{suffix}/{args.views}views/{scene}"
        if (out / "check.json").exists():
            results[scene] = json.loads((out / "check.json").read_text())
            continue
        print(scene, flush=True)
        out.mkdir(parents=True, exist_ok=True)
        try:
            if not (out / "variances.npz").exists():
                cue = cues.build(args.views, scene) / f"{args.cue}.npz"
                script = [HERE / "inject.py", "--vis_cues", cue, "--vis_stages", "ss", "--vis_dump", out / "variances.npz"]
                with open(out / "run.log", "a") as log:
                    npz = ROOT / f"outputs/unposed_genrecon/{args.views}views/{scene}/vggt.npz"
                    run_eval.genrecon(npz, out / "scene", out / "genrecon", genrecon_args, log, script)
                shutil.rmtree(out / "scene/rgb", ignore_errors=True)
            results[scene] = check(scene, base, out / "genrecon", np.load(out / "variances.npz"))
            (out / "check.json").write_text(json.dumps(results[scene], indent=2))
        except Exception as e:  # one bad scene should not stop the rest
            print(f"  failed ({e}), see {out / 'run.log'}", flush=True)

    if not results:
        return
    print(f"\n{args.cue} cue, {len(results)} scenes, median over scenes "
          f"(cells with a hidden view: {np.median([r['hidden_fraction'] for r in results.values()]):.2f})")
    print(f"{'variance':<14}{'auroc':>8}{'auroc_hidden':>14}")
    for mode in MODES:
        a = [r[mode]["auroc"] for r in results.values()]
        h = [r[mode]["auroc_hidden"] for r in results.values()]
        print(f"{mode:<14}{np.nanmedian(a):>8.3f}{np.nanmedian(h):>14.3f}")


if __name__ == "__main__":
    main()
