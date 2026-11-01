"""Precision, recall and F-score at 2, 5 and 10 cm.

With the scan's cells, all surface stage 2 generates lies in a cell that contains scan surface, so within
about one cell (10 cm) of the scan, and the precision at 10 cm is close to 1 by construction. The tighter
thresholds show how well stage 2 places the surface inside the cells. Same point sampling and crop as
scripts/evaluate_scannetpp.py.

Usage:
    uv run experiments/genrecon_oracle_occupancy/fscores.py PRED_DIR

writes fscores.json next to each mesh (a scene that already has one is not evaluated again) and prints
the mean over the scenes.
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import evaluate_scannetpp as ev  # noqa: E402

THRESHOLDS = [0.02, 0.05, 0.10]  # metres


def fscores(mesh: Path) -> dict:
    pred, scan = ev.load_prediction(mesh), ev.load_scan(ev.DATA / mesh.parent.name)
    pred_points, _ = ev.sample_points(pred, ev.NUM_POINTS)
    scan_points, _ = ev.sample_points(scan, ev.NUM_POINTS)
    pred_to_scan, _ = cKDTree(scan_points).query(pred_points)
    keep = pred_to_scan <= ev.CROP_DISTANCE  # as in evaluate_scannetpp.py, far predicted points are not scored
    pred_to_scan = pred_to_scan[keep]
    scan_to_pred, _ = cKDTree(pred_points[keep]).query(scan_points)
    out = {}
    for t in THRESHOLDS:
        precision, recall = (pred_to_scan < t).mean(), (scan_to_pred < t).mean()
        out[f"precision@{t}"] = float(precision)
        out[f"recall@{t}"] = float(recall)
        out[f"fscore@{t}"] = float(2 * precision * recall / (precision + recall))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pred_dir", type=Path)
    args = parser.parse_args()

    meshes = sorted(args.pred_dir.glob("*/mesh.ply*"))
    todo = [mesh for mesh in meshes if not (mesh.parent / "fscores.json").exists()]
    # Each scene in a new process: one process does not release all memory between scenes.
    with ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1) as pool:
        for mesh, result in zip(todo, pool.map(fscores, todo)):
            print(f"evaluated {mesh}", flush=True)
            (mesh.parent / "fscores.json").write_text(json.dumps(result, indent=2) + "\n")

    results = [json.loads((mesh.parent / "fscores.json").read_text()) for mesh in meshes]
    mean = {name: float(np.mean([r[name] for r in results])) for name in results[0]}
    print(f"mean over {len(results)} scenes:")
    print(json.dumps(mean, indent=2))


if __name__ == "__main__":
    main()
