"""Compare VGGT-Omega's camera poses with the ScanNet++ ones, per scene and on average.

Reads the poses run_eval.py already wrote (<scene>/scene/colmap/images.txt), so no GPU and no rerun.
Both GenRecon runs (smartphone and --matched) share these poses, so one evaluation covers both.

Metrics:
    Pairwise, independent of the world frame and scale (standard for VGGT, CO3D, RealEstate10K):
        rre_deg      relative rotation error between two cameras, in degrees
        rte_deg      angle between the predicted and true direction from one camera to the other
        auc30        area under the accuracy curve of max(rre, rte) up to 30 degrees (higher is better)
    After aligning all cameras to the ScanNet++ ones (scale, rotation, translation, as for the mesh):
        ate_cm       RMS distance between predicted and true camera centres, in cm
        rot_deg      angle between each aligned camera's orientation and the true one, in degrees

Writes cameras.json into every scene folder and prints one line per scene (worst AUC first), the
mean over scenes, and how the pose metrics correlate with the mesh F-score.

Usage:
    uv run experiments/camera_poses/eval_cameras.py                     # 32 views
    uv run experiments/camera_poses/eval_cameras.py --views 8
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]

# ScanNet++ nerfstudio world -> scan frame (as in run_eval.py and scripts/evaluate_scannetpp.py)
NERFSTUDIO_TO_SCAN = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=float)
# nerfstudio cameras look down -z with y up (OpenGL); COLMAP/VGGT cameras look down +z with y down (OpenCV)
GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def fit_similarity(src, dst):
    """Scale s, rotation R, translation t minimising |dst - (s R src + t)|^2 over the point pairs (Umeyama)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    x, y = src - mu_s, dst - mu_d
    R = Rotation.align_vectors(y, x)[0].as_matrix()  # best rotation of x onto y (Kabsch)
    s = (y * (x @ R.T)).sum() / (x**2).sum()  # best scale for that rotation
    return s, R, mu_d - s * R @ mu_s


def angle_deg(R):
    """Rotation angle of each 3x3 matrix in R (N, 3, 3), in degrees."""
    return np.degrees(Rotation.from_matrix(R).magnitude())


def predicted_poses(images_txt):
    """World-to-camera rotations and camera centres by image name, from COLMAP images.txt."""
    poses = {}
    for line in images_txt.read_text().splitlines():
        parts = line.split()
        if len(parts) != 10:
            continue
        qw, qx, qy, qz, tx, ty, tz = map(float, parts[1:8])
        R = Rotation.from_quat([qw, qx, qy, qz], scalar_first=True).as_matrix()
        poses[parts[9]] = (R, -R.T @ np.array([tx, ty, tz]))
    return poses


def true_poses(transforms_json):
    """World-to-camera rotations (OpenCV axes, scan frame) and camera centres by image file name."""
    poses = {}
    for f in json.loads(transforms_json.read_text())["frames"]:
        c2w = NERFSTUDIO_TO_SCAN @ np.array(f["transform_matrix"]) @ GL_TO_CV
        poses[Path(f["file_path"]).name] = (c2w[:3, :3].T, c2w[:3, 3])
    return poses


def evaluate(scene_out, scene_in):
    gt = true_poses(scene_in / "dslr/nerfstudio/transforms_undistorted.json")
    pred_by_index = predicted_poses(scene_out / "scene/colmap/images.txt")
    # convert.py names image i "{i:04d}.png"; i is the i-th entry of the npz's image_paths, which
    # infer.py took in sorted order from the image folder. Fall back to that order without the npz.
    npz = scene_out / "vggt.npz"
    names = [Path(p).name for p in np.load(npz)["image_paths"]] if npz.exists() else sorted(gt)
    R_p = np.stack([pred_by_index[f"{i:04d}.png"][0] for i in range(len(names))])
    C_p = np.stack([pred_by_index[f"{i:04d}.png"][1] for i in range(len(names))])
    R_g = np.stack([gt[n][0] for n in names])
    C_g = np.stack([gt[n][1] for n in names])
    return pose_metrics(R_p, C_p, R_g, C_g)


def pose_metrics(R_p, C_p, R_g, C_g):
    """Metrics of predicted world-to-camera rotations R_p (N, 3, 3) and camera centres C_p (N, 3) against the true ones."""
    # Pairwise errors over all ordered pairs i != j
    i, j = np.where(~np.eye(len(R_p), dtype=bool))
    rel_p = R_p[j] @ R_p[i].transpose(0, 2, 1)
    rel_g = R_g[j] @ R_g[i].transpose(0, 2, 1)
    rre = angle_deg(rel_p @ rel_g.transpose(0, 2, 1))
    dir_p = np.einsum("nab,nb->na", R_p[i], C_p[j] - C_p[i])  # direction to camera j, in camera i's frame
    dir_g = np.einsum("nab,nb->na", R_g[i], C_g[j] - C_g[i])
    cos = (dir_p * dir_g).sum(1) / (np.linalg.norm(dir_p, axis=1) * np.linalg.norm(dir_g, axis=1) + 1e-12)
    rte = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    err = np.maximum(rre, rte)
    auc30 = float(np.mean([(err < t).mean() for t in range(1, 31)]))

    # Absolute errors after the same similarity alignment as the mesh
    s, R, t = fit_similarity(C_p, C_g)
    ate = float(np.sqrt(((s * C_p @ R.T + t - C_g) ** 2).sum(1).mean()))
    # Aligned camera-to-world rotation R @ R_p^T against the true R_g^T
    rot = angle_deg(R_g @ R @ R_p.transpose(0, 2, 1))

    return {
        "views": len(R_p),
        "auc30": auc30,
        "rre_deg_median": float(np.median(rre)),
        "rte_deg_median": float(np.median(rte)),
        "ate_cm": 100 * ate,
        "rot_deg_median": float(np.median(rot)),
        "scale": float(s),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--views", type=int, default=32)
    args = parser.parse_args()
    pred_dir = ROOT / f"outputs/unposed_genrecon/{args.views}views"
    gt_dir = ROOT / f"data/scannetpp/eval_inputs/{args.views}views"

    results = {}
    for scene_out in sorted(p for p in pred_dir.iterdir() if (p / "scene/colmap/images.txt").exists()):
        try:
            r = evaluate(scene_out, gt_dir / scene_out.name)
        except Exception as e:  # one bad scene should not stop the rest
            print(f"{scene_out.name}: failed ({e})")
            continue
        (scene_out / "cameras.json").write_text(json.dumps(r, indent=2) + "\n")
        metrics = scene_out / "metrics.json"
        r["fscore"] = json.loads(metrics.read_text())["fscore"] if metrics.exists() else float("nan")
        results[scene_out.name] = r
    if not results:
        raise SystemExit(f"No scenes with scene/colmap/images.txt in {pred_dir}")

    cols = ["auc30", "rre_deg_median", "rte_deg_median", "ate_cm", "rot_deg_median", "fscore"]
    print(f"{'scene':<12}" + "".join(f"{c:>16}" for c in cols))
    for name, r in sorted(results.items(), key=lambda kv: kv[1]["auc30"]):
        print(f"{name:<12}" + "".join(f"{r[c]:>16.3f}" for c in cols))

    print(f"\nmean over {len(results)} scenes:")
    for c in cols:
        print(f"  {c:<16}{np.nanmean([r[c] for r in results.values()]):.3f}")
    print(f"  {'(median ate_cm)':<16}{np.median([r['ate_cm'] for r in results.values()]):.3f}")

    f = np.array([r["fscore"] for r in results.values()])
    if np.isfinite(f).sum() > 2:
        print("\nSpearman correlation with F-score (negative for errors means bigger error, worse mesh):")
        for c in cols[:-1]:
            rho, p = spearmanr([r[c] for r in results.values()], f, nan_policy="omit")
            print(f"  {c:<16}{rho:+.2f}  (p = {p:.3f})")


if __name__ == "__main__":
    main()
