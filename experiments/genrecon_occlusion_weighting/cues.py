"""Depth cues for inject.py, one file per scene and source, in the units of GenRecon's (VGGT-Omega) cameras.

    vggt  VGGT-Omega's depth and confidence from experiments/unposed_genrecon (vggt.npz). These cameras are
          what GenRecon uses in the unposed setting, so depth needs no alignment. c ramps linearly from 0
          at conf = --conf-lo to 1 at conf = --conf-hi.
    gt    The laser scan rendered at the same cameras, c = 1 (c = 0 where a ray misses). The cameras are
          placed in the scan with the similarity transform from their centres to the ScanNet++ ones, as in
          unposed_genrecon/run_eval.py, and the depth divided by its scale. This keeps the VGGT-Omega
          cameras, so only the depth differs from the vggt cue.

Each file holds names (convert.py's image names), depth [S, H, W], conf [S, H, W] in [0, 1] and K [S, 3, 3]
normalised to the full image. check.json compares the two depths where both are known: a median ratio far
from 1 or a large AbsRel means a frame, scale or sign error, not a weak cue.

Writes outputs/genrecon_visibility/cues/<N>views/<scene>/{vggt,gt}.npz and check.json; skips existing ones.
Needs experiments/unposed_genrecon/run_eval.py to have produced vggt.npz.

Usage:
    uv run experiments/genrecon_visibility/cues.py 32 [scene ...]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments/unposed_genrecon"))
import evaluate_scannetpp as ev  # noqa: E402
from run_eval import NERFSTUDIO_TO_SCAN, umeyama  # noqa: E402

OUT = ROOT / "outputs/genrecon_visibility/cues"
CONF_LO, CONF_HI = 2.0, 5.0  # convert.py keeps VGGT-Omega points with conf >= 2


def vggt_npz(n: int, scene: str) -> Path:
    return ROOT / f"outputs/unposed_genrecon/{n}views/{scene}/vggt.npz"


def load_vggt(path: Path):
    d = np.load(path)
    S = d["depth"].shape[0]
    depth = d["depth"].reshape(d["depth"].shape[:3]).astype(np.float32)
    conf = d["depth_conf"].reshape(depth.shape).astype(np.float32)
    extr = d["extrinsics"][:, :3, :4].astype(np.float64)  # world-to-camera, OpenCV
    K_pix = d["intrinsics"].astype(np.float64)
    names = [f"{i:04d}.png" for i in range(S)]  # convert.py's names for the GenRecon scene
    return d["image_paths"], names, depth, conf, extr, K_pix


def normalised(K_pix: np.ndarray, H: int, W: int) -> np.ndarray:
    K = K_pix.copy()
    K[:, 0] /= W
    K[:, 1] /= H
    return K.astype(np.float32)


def scan_depth(scene: str, n: int, image_paths, extr, K_pix, H: int, W: int):
    """The scan rendered at the VGGT-Omega cameras, in VGGT-Omega units. Returns depth and the transform."""
    frames = json.loads((ROOT / f"data/scannetpp/eval_inputs/{n}views/{scene}/dslr/nerfstudio/"
                                "transforms_undistorted.json").read_text())["frames"]
    gt = {Path(f["file_path"]).name: (NERFSTUDIO_TO_SCAN @ np.array(f["transform_matrix"]))[:3, 3] for f in frames}
    R_c2w = extr[:, :, :3].transpose(0, 2, 1)
    centres = -np.einsum("sij,sj->si", R_c2w, extr[:, :, 3])
    # Rotation-equivariant fit, so the same scale as run_eval.py's fit in convert.py's rotated frame.
    s, R, t = umeyama(centres, np.array([gt[Path(p).name] for p in image_paths]))

    rays = ev.raycasting_scene(ev.load_scan(ev.DATA / scene))
    depth = np.empty((len(extr), H, W), dtype=np.float32)
    for i in range(len(extr)):
        cam_to_scan = np.eye(4)
        cam_to_scan[:3, :3] = R @ R_c2w[i]
        cam_to_scan[:3, 3] = s * R @ centres[i] + t
        depth[i] = ev.render_depth(rays, K_pix[i], cam_to_scan, W, H) / s
    return depth, {"scale": float(s), "rotation": R.tolist(), "translation": t.tolist()}


def check(vggt_depth, vggt_conf, gt_depth) -> dict:
    """VGGT-Omega depth against the scan, where the scan is hit and VGGT-Omega is confident (c > 0)."""
    per_view = []
    for dv, cv, dg in zip(vggt_depth, vggt_conf, gt_depth):
        m = np.isfinite(dg) & (dg > 0) & (cv > 0) & (dv > 0)
        if m.sum() < 100:
            per_view.append(None)
            continue
        r = dg[m] / dv[m]
        per_view.append({"pixels": int(m.sum()), "median_gt_over_vggt": float(np.median(r)),
                         "absrel": float(np.mean(np.abs(dv[m] - dg[m]) / dg[m])),
                         "delta_1.25": float(np.mean(np.maximum(r, 1 / r) < 1.25))})
    ok = [v for v in per_view if v]
    mean = {k: float(np.mean([v[k] for v in ok])) for k in ("median_gt_over_vggt", "absrel", "delta_1.25")} if ok else {}
    return {"mean": mean, "views": per_view}


def build(n: int, scene: str, conf_lo: float = CONF_LO, conf_hi: float = CONF_HI) -> Path:
    out = OUT / f"{n}views/{scene}"
    if (out / "vggt.npz").exists() and (out / "gt.npz").exists():
        return out
    src = vggt_npz(n, scene)
    if not src.exists():
        raise FileNotFoundError(f"{src} missing, run experiments/unposed_genrecon/run_eval.py {n} first")
    out.mkdir(parents=True, exist_ok=True)
    image_paths, names, depth, conf, extr, K_pix = load_vggt(src)
    H, W = depth.shape[1:]
    K = normalised(K_pix, H, W)

    c = np.clip((conf - conf_lo) / (conf_hi - conf_lo), 0, 1).astype(np.float32)
    np.savez_compressed(out / "vggt.npz", names=names, depth=depth, conf=c, K=K)

    gt_depth, transform = scan_depth(scene, n, image_paths, extr, K_pix, H, W)
    hit = np.isfinite(gt_depth)
    np.savez_compressed(out / "gt.npz", names=names, depth=np.where(hit, gt_depth, 0).astype(np.float32),
                        conf=hit.astype(np.float32), K=K)

    report = {"conf_lo": conf_lo, "conf_hi": conf_hi, "vggt_to_scan": transform,
              "scan_hit_fraction": float(hit.mean()), "vggt_c_mean": float(c.mean()), **check(depth, c, gt_depth)}
    (out / "check.json").write_text(json.dumps(report, indent=2))
    print(f"{scene}: VGGT-Omega vs scan {report['mean']}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("views", type=int, help="number of input views")
    parser.add_argument("scenes", nargs="*", help="scene IDs (default: every scene with a vggt.npz)")
    parser.add_argument("--conf-lo", type=float, default=CONF_LO)
    parser.add_argument("--conf-hi", type=float, default=CONF_HI)
    args = parser.parse_args()
    scenes = args.scenes or sorted(p.parent.name for p in
                                   (ROOT / f"outputs/unposed_genrecon/{args.views}views").glob("*/vggt.npz"))
    for scene in scenes:
        build(args.views, scene, args.conf_lo, args.conf_hi)


if __name__ == "__main__":
    main()
