"""Streaming camera-pose benchmark (paper Table 3, pose columns): Sintel, TUM-dynamics and ScanNet.

Frames and ground truth follow the relpose evaluation of CUT3R / TTT3R (sintel, tum_s1_1000, scannet_s3_1000):
  <data-root>/sintel/training/final/<seq>/*.png, camdata_left/<seq>/*.cam       (14 sequences)
  <data-root>/long_tum_s1/<seq>/rgb_1000/*.png, groundtruth_1000.txt              (8 sequences, one GT row per image)
  <data-root>/long_scannet_s3/<scene>/color_1000/*.jpg, pose_1000.txt             (scans with a complete GT trajectory)
Every frame is answered when it arrives. Metrics as in those evaluations: ATE (RMSE of camera centres after a Sim(3)
alignment), RPE translation and rotation (consecutive frames, after the same alignment), averaged over sequences.

usage:
  python eval/streaming.py --data-root /path/to/data --task tumdyn --config L8 --out results/streaming
  python eval/streaming.py --data-root /path/to/data --task all --config S4 --out results/streaming
  python eval/streaming.py --data-root /path/to/data --task sintel --out results/streaming --score-only
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solv3r.images import load_images  # noqa: E402

SINTEL = ["alley_2", "ambush_4", "ambush_5", "ambush_6", "cave_2", "cave_4", "market_2", "market_5", "market_6",
          "shaman_3", "sleeping_1", "sleeping_2", "temple_2", "temple_3"]
TASKS = ("sintel", "tumdyn", "scannet")


# ---------------------------------------------------------------------- data
def _sintel_cam(path):
    with open(path, "rb") as f:
        assert np.fromfile(f, np.float32, 1)[0] == 202021.25, path
        np.fromfile(f, np.float64, 9)
        w2c = np.fromfile(f, np.float64, 12).reshape(3, 4)
    return np.linalg.inv(np.vstack([w2c, [0, 0, 0, 1]]))


def sequences(root, task):
    """-> list of (name, frame paths, ground-truth camera-to-world (N, 4, 4))."""
    out = []
    if task == "sintel":
        for s in SINTEL:
            frames = sorted(glob.glob(f"{root}/sintel/training/final/{s}/*.png"))
            cams = sorted(glob.glob(f"{root}/sintel/training/camdata_left/{s}/*.cam"))
            if not frames or not cams:
                raise FileNotFoundError(f"Sintel sequence {s} not found in {root}/sintel/training/{{final,camdata_left}}")
            out.append((s, frames, np.stack([_sintel_cam(p) for p in cams])))
    elif task == "tumdyn":
        for d in sorted(glob.glob(f"{root}/long_tum_s1/*/")):
            frames = sorted(glob.glob(f"{d}rgb_1000/*.png"))
            rows = [l.split() for l in open(f"{d}groundtruth_1000.txt") if l.strip() and not l.startswith("#")]
            gt = np.tile(np.eye(4), (len(rows), 1, 1))
            for i, r in enumerate(rows):
                gt[i, :3, :3] = Rotation.from_quat([float(x) for x in r[4:8]]).as_matrix()
                gt[i, :3, 3] = [float(x) for x in r[1:4]]
            out.append((os.path.basename(d.rstrip("/")).replace("rgbd_dataset_", ""), frames, gt))
    elif task == "scannet":
        for d in sorted(glob.glob(f"{root}/long_scannet_s3/*/")):
            if not os.path.isfile(f"{d}pose_1000.txt"):
                continue
            gt = np.loadtxt(f"{d}pose_1000.txt").reshape(-1, 4, 4)
            if not np.isfinite(gt).all():
                continue                              # incomplete ground truth: skipped, as in the reference evaluation
            out.append((os.path.basename(d.rstrip("/")), sorted(glob.glob(f"{d}color_1000/*.jpg")), gt))
    else:
        raise ValueError(task)
    if not out:
        raise FileNotFoundError(f"no {task} sequences found in {root}")
    for name, frames, gt in out:
        assert len(frames) == len(gt), (name, len(frames), len(gt))
    return out


# ---------------------------------------------------------------------- metrics
def umeyama(src, dst):
    """Sim(3) (s, R, t) minimising |s R src + t - dst|^2 over points (N, 3)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / xs.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


def _orthonormal(R):
    return Rotation.from_matrix(R).as_matrix()


def pose_metrics(pred, gt):
    """pred, gt: camera-to-world (N, 4, 4). -> ATE, RPE translation, RPE rotation (deg)."""
    s, R, t = umeyama(pred[:, :3, 3], gt[:, :3, 3])
    est = np.tile(np.eye(4), (len(pred), 1, 1))
    est[:, :3, :3] = R[None] @ _orthonormal(pred[:, :3, :3])
    est[:, :3, 3] = s * pred[:, :3, 3] @ R.T + t
    ref = gt.copy()
    ref[:, :3, :3] = _orthonormal(gt[:, :3, :3])
    ate = np.sqrt(np.mean(np.sum((est[:, :3, 3] - ref[:, :3, 3]) ** 2, 1)))
    rel = lambda P: np.linalg.inv(P[:-1]) @ P[1:]
    E = np.linalg.inv(rel(ref)) @ rel(est)
    rpe_t = np.sqrt(np.mean(np.sum(E[:, :3, 3] ** 2, 1)))
    cos = np.clip((np.trace(E[:, :3, :3], axis1=1, axis2=2) - 1) / 2, -1, 1)
    rpe_r = np.sqrt(np.mean(np.degrees(np.arccos(cos)) ** 2))
    return dict(ate=float(ate), rpe_trans=float(rpe_t), rpe_rot=float(rpe_r))


def score(out_dir, task, seqs):
    rows = {}
    for name, _, gt in seqs:
        f = f"{out_dir}/{task}/{name}/pred.npz"
        if os.path.isfile(f):
            rows[name] = pose_metrics(np.load(f)["c2w"], gt)
    if not rows:
        print(f"{task}: no predictions in {out_dir}/{task}")
        return None
    mean = {k: float(np.mean([r[k] for r in rows.values()])) for k in ("ate", "rpe_trans", "rpe_rot")}
    print(f"{task}: {len(rows)}/{len(seqs)} sequences  ATE {mean['ate']:.4f}  RPE trans {mean['rpe_trans']:.4f}  "
          f"RPE rot {mean['rpe_rot']:.3f} deg")
    json.dump(dict(mean=mean, sequences=rows), open(f"{out_dir}/{task}/scores.json", "w"), indent=1)
    return mean


# ---------------------------------------------------------------------- run
def run(model, seqs, out_dir, task, config, depth=False):
    from solv3r.streaming import run_stream

    for name, frames, _ in seqs:
        d = f"{out_dir}/{task}/{name}"
        if os.path.isfile(f"{d}/pred.npz"):
            continue
        torch.cuda.reset_peak_memory_stats()
        res = run_stream(model, load_images(frames), config, depth=depth)
        os.makedirs(d, exist_ok=True)
        extra = dict(depth=res.depth, depth_conf=res.depth_conf) if depth else {}
        np.savez(f"{d}/pred.npz", c2w=res.c2w, c2w_exit=res.c2w_exit, intrinsics=res.intrinsics, **extra)
        json.dump(dict(frames=len(frames), ms_per_frame=res.ms_per_frame, refresh_rate=float(np.mean(res.refreshed)),
                       peak_gpu_gb=torch.cuda.max_memory_allocated() / 2 ** 30), open(f"{d}/stats.json", "w"))
        print(f"{task}/{name}: {len(frames)} frames, {res.ms_per_frame:.1f} ms/frame (solver)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--task", default="all", choices=TASKS + ("all",))
    ap.add_argument("--config", default="L8", choices=("L8", "L4", "S4"))
    ap.add_argument("--checkpoint", default=None, help="a .safetensors file or a Hugging Face repo id (default: checkpoints/ if present, else the released weights)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", action="store_true", help="also save the depth of every frame on arrival")
    ap.add_argument("--score-only", action="store_true")
    a = ap.parse_args()
    out = f"{a.out}/{a.config}"
    model = None
    for task in TASKS if a.task == "all" else (a.task,):
        seqs = sequences(a.data_root, task)
        if not a.score_only:
            if model is None:
                from solv3r.model import Solv3R
                model = Solv3R.from_pretrained(a.checkpoint)
            run(model, seqs, out, task, a.config, a.depth)
        score(out, task, seqs)


if __name__ == "__main__":
    main()
