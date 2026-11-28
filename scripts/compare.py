"""Runs side by side, over the scenes they all have.

Usage:
    uv run scripts/compare.py [RUN ...] [--dataset scannetpp] [--views 32] [--metrics GROUP.NAME ...]

RUN is <stage>/<experiment>/<tag> (default: every run with results for the dataset and view count). The
means are taken over the scenes that every listed run has, so the rows always compare the same scenes.
A metric is GROUP.NAME as in metrics.json, e.g. mesh.fscore@5cm.
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["cameras.registered", "cameras.auc30", "cameras.ate_cm", "depth.scene_absrel", "depth.points_median_cm",
           "cells.precision", "cells.recall", "mesh.completeness", "mesh.precision@5cm", "mesh.recall@5cm",
           "mesh.fscore@5cm", "mesh.chamfer"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*")
    parser.add_argument("--dataset", default="scannetpp")
    parser.add_argument("--views", type=int, default=32)
    parser.add_argument("--metrics", nargs="+", default=METRICS)
    args = parser.parse_args()

    name = f"{args.dataset}_{args.views}views.json"
    paths = [ROOT / "results" / run / name for run in args.runs] or sorted((ROOT / "results").glob(f"*/*/*/{name}"))
    results = {}
    for path in paths:
        if not path.exists():
            raise SystemExit(f"No results at {path.relative_to(ROOT)}; run scripts/evaluate.py first.")
        results[str(path.parent.relative_to(ROOT / "results"))] = json.loads(path.read_text())["per_scene"]
    if not results:
        raise SystemExit(f"No results for {args.dataset}, {args.views} views.")

    scenes = sorted(set.intersection(*(set(per_scene) for per_scene in results.values())))

    def mean(per_scene: dict, metric: str) -> float:
        group, name = metric.split(".", 1)
        values = [v for s in scenes if np.isfinite(v := per_scene[s].get(group, {}).get(name, np.nan))]
        return float(np.mean(values)) if values else np.nan

    table = {run: [mean(per_scene, metric) for metric in args.metrics] for run, per_scene in results.items()}
    columns = [i for i, _ in enumerate(args.metrics) if any(np.isfinite(row[i]) for row in table.values())]

    print(f"{args.dataset}, {args.views} views, {len(scenes)} scenes in common\n")
    widths = [max(len(args.metrics[i]), 8) + 2 for i in columns]
    run_width = max(map(len, table)) + 2
    print("run".ljust(run_width) + "".join(args.metrics[i].rjust(w) for i, w in zip(columns, widths)))
    for run, row in table.items():
        print(run.ljust(run_width) + "".join((f"{row[i]:.4g}" if np.isfinite(row[i]) else "-").rjust(w)
                                             for i, w in zip(columns, widths)))


if __name__ == "__main__":
    main()
