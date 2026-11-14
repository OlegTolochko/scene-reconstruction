"""Solv3R's output on a room's renders -> the depth-cue files train.py reads, for one or more SAGE rooms.

Input per room: <room>_solv3r.npz from experiments/unposed_genrecon/infer_solv3r.py run on
<base>/renders_room/<room>/ (all frames), with image_paths, depth [S, H, W(, 1)], depth_conf [S, H, W],
extrinsics [S, 3, 4] world-to-camera (OpenCV) and intrinsics [S, 3, 3] in pixels of the depth maps.

Solv3R's world frame is its first camera's, at an unknown scale. The camera centres are aligned to the true
ones of transforms.json with a similarity transform (as unposed_genrecon/run_eval.py does with the scan), which
gives camera-to-room poses and depth in room units. Rooms whose alignment error exceeds --max-rmse (relative to
the spread of the true camera centres) are skipped.

Writes <out>/<room>/cameras.npz (K [F, 3, 3] normalised to the full render, c2w [F, 4, 4] camera-to-room in
Blender's convention, F = transforms.json's frames in its order), <out>/<room>/<i:04d>.npz (depth, conf; float16)
and <out>/metadata.csv (sha256 = the room's chunks from <base>/crops/<room>.json, depth_cues = True), which
GenRecon's dataset reads from every data root. align.json per room records the alignment.

Usage:
    uv run experiments/genrecon_depth_cue_finetuning/prepare_cues.py --base data/SAGE10k \
        --out data/SAGE10k/depth_cues_solv3r ROOM_solv3r.npz [...]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/unposed_genrecon"))
from run_eval import umeyama  # noqa: E402

FLIP = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenCV <-> Blender camera axes


def convert(npz: Path, base: Path, out: Path, max_rmse: float) -> dict:
    d = np.load(npz)
    names = [Path(str(p)).name for p in d["image_paths"]]
    room = npz.name.removesuffix("_solv3r.npz")
    frames = json.loads((base / "renders_room" / room / "transforms.json").read_text())["frames"]
    index = {Path(f["file_path"]).name: i for i, f in enumerate(frames)}
    missing = [f["file_path"] for f in frames if Path(f["file_path"]).name not in names]
    if missing:
        raise RuntimeError(f"{room}: Solv3R has no prediction for frames {missing[:5]}")
    order = [names.index(Path(f["file_path"]).name) for f in frames]  # Solv3R index per frame

    S = len(names)
    depth = d["depth"].reshape(S, *d["depth"].shape[1:3]).astype(np.float64)[order]
    H, W = depth.shape[1:]
    conf = d["depth_conf"].reshape(S, H, W).astype(np.float32)[order]
    w2c = np.tile(np.eye(4), (S, 1, 1))
    w2c[:, :3] = d["extrinsics"][:, :3, :4]
    c2w_cv = np.linalg.inv(w2c)[order]
    K = d["intrinsics"].astype(np.float64)[order].copy()
    K[:, 0] /= W
    K[:, 1] /= H

    true_c2w = np.array([f["transform_matrix"] for f in frames], dtype=np.float64)
    s, R, t = umeyama(c2w_cv[:, :3, 3], true_c2w[:, :3, 3])
    centres = s * c2w_cv[:, :3, 3] @ R.T + t
    rmse = float(np.sqrt(((centres - true_c2w[:, :3, 3]) ** 2).sum(1).mean()))
    spread = float(np.sqrt(((true_c2w[:, :3, 3] - true_c2w[:, :3, 3].mean(0)) ** 2).sum(1).mean()))

    c2w = np.tile(np.eye(4), (len(frames), 1, 1))
    c2w[:, :3, :3] = R @ c2w_cv[:, :3, :3]
    c2w[:, :3, 3] = centres
    c2w = c2w @ FLIP  # OpenCV camera axes -> Blender's, as transforms.json
    rot_err = [float(np.degrees(np.arccos(np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1, 1))))
               for a, b in zip(c2w, true_c2w)]
    f_true = 0.5 / np.tan(np.array([f["camera_angle_x"] for f in frames]) / 2)
    info = {"room": room, "frames": len(frames), "scale": float(s), "rmse": rmse, "rmse_rel": rmse / spread,
            "rot_err_deg_median": float(np.median(rot_err)), "rot_err_deg_max": float(np.max(rot_err)),
            "focal_ratio_median": float(np.median(K[:, 0, 0] / f_true)), "depth_hw": [H, W],
            "conf_mean": float(conf.mean()), "conf_std": float(conf.std())}
    if rmse / spread > max_rmse:
        info["skipped"] = True
        return info

    room_out = out / room
    room_out.mkdir(parents=True, exist_ok=True)
    np.savez(room_out / "cameras.npz", K=K.astype(np.float32), c2w=c2w.astype(np.float32))
    for i in range(len(frames)):
        np.savez_compressed(room_out / f"{i:04d}.npz", depth=(s * depth[i]).astype(np.float16),
                            conf=conf[i].astype(np.float16))
    (room_out / "align.json").write_text(json.dumps(info, indent=2))
    chunks = json.loads((base / "crops" / f"{room}.json").read_text())["chunks"]
    info["chunks"] = [f"{room}_{int(c['index']):04d}" for c in chunks]
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("npz", type=Path, nargs="+", help="<room>_solv3r.npz files")
    parser.add_argument("--base", type=Path, required=True, help="SAGE data root (renders_room/, crops/)")
    parser.add_argument("--out", type=Path, required=True, help="depth-cue root to write")
    parser.add_argument("--max-rmse", type=float, default=0.05, help="max camera alignment error / centre spread")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    meta_path = args.out / "metadata.csv"
    meta = {}  # sha256 -> depth_cues
    if meta_path.exists():
        with open(meta_path) as f:
            meta = {r["sha256"]: r["depth_cues"] for r in csv.DictReader(f)}
    for npz in args.npz:
        info = convert(npz, args.base, args.out, args.max_rmse)
        print(json.dumps({k: v for k, v in info.items() if k != "chunks"}), flush=True)
        if not info.get("skipped"):
            meta.update({c: "True" for c in info["chunks"]})
    with open(meta_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sha256", "depth_cues"])
        w.writerows(sorted(meta.items()))

if __name__ == "__main__":
    main()
