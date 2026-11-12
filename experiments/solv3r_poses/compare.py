"""Solv3R's camera poses against VGGT-Omega's, on the same ScanNet++ eval scenes and views.

Scores both with pose_metrics() of experiments/camera_poses/eval_cameras.py, so the metrics are the same as there
(auc30, rre/rte medians, ate_cm, rot_deg_median). The poses are read straight from the models' outputs; the
similarity transform convert.py applies does not change any of these metrics.

    VGGT-Omega         outputs/unposed_genrecon/<N>views/<scene>/vggt.npz   (experiments/unposed_genrecon/run_eval.py)
    Solv3R (rays)      outputs/solv3r_poses/<N>views/<scene>.npz            (infer.py, cameras fitted to the rays)
    Solv3R (head)      the same file, cameras from the camera head

Prints one line per scene (worst Solv3R AUC first) and the means over the scenes all three have, and writes
outputs/solv3r_poses/<N>views/comparison.csv.

Usage:
    uv run experiments/solv3r_poses/compare.py 32
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/camera_poses"))
from eval_cameras import pose_metrics, true_poses  # noqa: E402

METRICS = ["auc30", "rre_deg_median", "rte_deg_median", "ate_cm", "rot_deg_median"]


def poses(names, extrinsics):
    """World-to-camera rotations and camera centres by image name, from (S, 3, 4) world-to-camera [R | t]."""
    R, t = extrinsics[:, :3, :3].astype(float), extrinsics[:, :3, 3].astype(float)
    C = -np.einsum("sji,sj->si", R, t)
    return {n: (R[k], C[k]) for k, n in enumerate(names)}


def score(pred, gt):
    names = sorted(gt)
    R_p, C_p = np.stack([pred[n][0] for n in names]), np.stack([pred[n][1] for n in names])
    R_g, C_g = np.stack([gt[n][0] for n in names]), np.stack([gt[n][1] for n in names])
    return pose_metrics(R_p, C_p, R_g, C_g)


def predictions(scene, n):
    """Poses by method name for the scene, for the methods that have an output."""
    out = {}
    vggt = ROOT / f"outputs/unposed_genrecon/{n}views/{scene}/vggt.npz"
    if vggt.exists():
        d = np.load(vggt)
        out["vggt_omega"] = poses([Path(p).name for p in d["image_paths"]], d["extrinsics"])
    solv3r = ROOT / f"outputs/solv3r_poses/{n}views/{scene}.npz"
    if solv3r.exists():
        d = np.load(solv3r)
        names = [str(x) for x in d["names"]]
        out["solv3r_rays"] = poses(names, d["extrinsics"])
        out["solv3r_head"] = poses(names, d["extrinsics_head"])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("views", type=int, nargs="+", help="numbers of input views")
    args = parser.parse_args()
    methods = ["vggt_omega", "solv3r_rays", "solv3r_head"]

    for n in args.views:
        inputs = ROOT / f"data/scannetpp/eval_inputs/{n}views"
        rows = []
        for scene_in in sorted(p for p in inputs.iterdir() if p.is_dir()):
            gt = true_poses(scene_in / "dslr/nerfstudio/transforms_undistorted.json")
            for method, pred in predictions(scene_in.name, n).items():
                try:
                    rows.append({"scene": scene_in.name, "method": method, **score(pred, gt)})
                except Exception as e:  # one bad scene should not stop the rest
                    print(f"{scene_in.name} {method}: failed ({e})")

        out = ROOT / f"outputs/solv3r_poses/{n}views"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "comparison.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["scene", "method", "views", *METRICS, "scale"])
            writer.writeheader()
            writer.writerows(rows)

        by = {(r["scene"], r["method"]): r for r in rows}
        scenes = sorted({r["scene"] for r in rows})
        common = [s for s in scenes if all((s, m) in by for m in methods)]
        oom = out / "oom.txt"
        print(f"\n===== {n} views: {len(scenes)} scenes with any poses, {len(common)} with all three"
              + (f", Solv3R out of memory on {len(oom.read_text().split())}" if oom.exists() else ""))

        cols = [("vggt_omega", "auc30"), ("solv3r_rays", "auc30"), ("vggt_omega", "ate_cm"), ("solv3r_rays", "ate_cm"),
                ("vggt_omega", "rot_deg_median"), ("solv3r_rays", "rot_deg_median")]
        head = {"vggt_omega": "vggt", "solv3r_rays": "solv3r"}
        print(f"{'scene':<12}" + "".join(f"{head[m] + ' ' + c.split('_')[0]:>14}" for m, c in cols))
        for s in sorted(scenes, key=lambda s: by.get((s, "solv3r_rays"), {}).get("auc30", -1)):
            cells = [by[(s, m)][c] if (s, m) in by else float("nan") for m, c in cols]
            print(f"{s:<12}" + "".join(f"{v:>14.3f}" for v in cells))

        print(f"\nmean over the {len(common)} scenes all three have (auc30 higher is better, the rest lower):")
        print(f"  {'':<16}" + "".join(f"{m:>14}" for m in methods))
        for c in METRICS:
            print(f"  {c:<16}" + "".join(f"{np.mean([by[(s, m)][c] for s in common]):>14.3f}" for m in methods))
        if common:
            wins = sum(by[(s, "solv3r_rays")]["auc30"] > by[(s, "vggt_omega")]["auc30"] for s in common)
            print(f"  Solv3R (rays) has the higher auc30 on {wins} of {len(common)} scenes")
        print(f"Wrote {out / 'comparison.csv'}")


if __name__ == "__main__":
    main()
