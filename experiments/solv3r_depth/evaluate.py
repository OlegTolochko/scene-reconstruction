"""Solv3R's depth at the input views against the scan, next to VGGT-Omega's on the same scenes.

The metrics are those of experiments/vggt_omega_depth/evaluate.py, computed by its evaluate() on Solv3R's
predictions; see there for the three alignments (scene, view, points). VGGT-Omega's results are read from
outputs/vggt_omega_depth.

Usage:
    uv run experiments/solv3r_depth/evaluate.py 8 16 32 64 256

writes outputs/solv3r_depth/<N>views/<scene>.json (a scene that already has one is not evaluated again),
outputs/solv3r_depth/summary.csv with one row per method, view count and scene, and prints the means.
"""

import argparse
import csv
import importlib.util
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/solv3r_depth"
VGGT = ROOT / "outputs/vggt_omega_depth"

spec = importlib.util.spec_from_file_location("vggt_omega_depth_evaluate", ROOT / "experiments/vggt_omega_depth/evaluate.py")
depth_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(depth_eval)

ROWS = [  # (metric, label, lower is better)
    ("scene_absrel", "depth AbsRel, scale per scene", True),
    ("scene_median_cm", "depth median error cm, scale per scene", True),
    ("view_absrel", "depth AbsRel, scale per view", True),
    ("view_median_cm", "depth median error cm, scale per view", True),
    ("points_median_cm", "points median distance to scan cm", True),
    ("points_within_5cm", "points within 5 cm of scan", False),
    ("camera_error_mean_m", "camera centre error m", True),
]


def evaluate(path: Path) -> dict:
    return depth_eval.evaluate(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("views", type=int, nargs="+", help="numbers of input views")
    args = parser.parse_args()

    rows = []
    for n in args.views:
        paths = sorted((OUT / f"{n}views").glob("*.npz"))
        todo = [path for path in paths if not path.with_suffix(".json").exists()]
        # Each scene in a new process: one process does not release all memory between scenes.
        with ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1) as pool:
            for path, metrics in zip(todo, pool.map(evaluate, todo)):
                print(f"evaluated {path}", flush=True)
                path.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")

        results = {"VGGT-Omega": [], "Solv3R": []}
        for path in paths:
            scene = path.stem
            vggt = json.loads((VGGT / f"{n}views/{scene}.json").read_text())
            solv3r = json.loads(path.with_suffix(".json").read_text())
            for method, result in (("VGGT-Omega", vggt), ("Solv3R", solv3r)):
                results[method].append(result)
                rows.append({"method": method, "views": n, "scene": scene,
                             **{name: result[name] for name, _, _ in ROWS}})

        print(f"\n{n} views, mean over {len(paths)} scenes")
        print(f"{'':42s}{'VGGT-Omega':>12s}{'Solv3R':>12s}   Solv3R better")
        for name, label, lower in ROWS:
            vggt, solv3r = (np.array([r[name] for r in results[m]]) for m in ("VGGT-Omega", "Solv3R"))
            wins = (solv3r < vggt) if lower else (solv3r > vggt)
            print(f"{label:42s}{np.mean(vggt):12.4g}{np.mean(solv3r):12.4g}   {wins.sum()}/{len(paths)} scenes")

    with open(OUT / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
