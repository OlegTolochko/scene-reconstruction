"""Depth cues for inject.py, one file per scene and source, in the units of GenRecon's cameras.

GenRecon's cameras in the unposed setting are those of --model (VGGT-Omega or Solv3R, as in
experiments/unposed_genrecon/run_eval.py), so both cues are built for that model's cameras:

    depth  the model's own depth and confidence (vggt.npz / solv3r.npz from experiments/unposed_genrecon),
           saved as vggt.npz / solv3r.npz. They come with the cameras, so depth needs no alignment. c ramps
           linearly from 0 at conf = --conf-lo to 1 at conf = --conf-hi.
    gt     The laser scan rendered at the same cameras, c = 1 (c = 0 where a ray misses). The cameras are
           placed in the scan with the similarity transform from their centres to the ScanNet++ ones, as in
           unposed_genrecon/run_eval.py, and the depth divided by its scale. This keeps the model's cameras,
           so only the depth differs from the depth cue.

Each file holds names (convert.py's image names), depth [S, H, W], conf [S, H, W] in [0, 1] and K [S, 3, 3]
normalised to the full image. check.json compares the two depths where both are known: a median ratio far
from 1 or a large AbsRel means a frame, scale or sign error, not a weak cue.

Writes <out_root>/cues/<N>views/<scene>/{vggt|solv3r,gt}.npz and check.json, where out_root is
outputs/genrecon_occlusion_weighting (VGGT-Omega) or outputs/genrecon_occlusion_weighting_solv3r; skips
existing ones. Needs experiments/unposed_genrecon/run_eval.py [--model solv3r] to have produced the npz.

Usage:
    uv run experiments/genrecon_occlusion_weighting/cues.py 32 [scene ...] [--model solv3r]
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
from run_eval import MODELS, NERFSTUDIO_TO_SCAN, umeyama  # noqa: E402

# convert.py keeps the points with conf >= 2, for both models. Check c_mean in check.json for Solv3R.
CONF_LO, CONF_HI = 2.0, 5.0


def out_root(model: str) -> Path:
    """This experiment's outputs for a model's cameras, named like run_eval.py's (unposed_genrecon_solv3r)."""
    return ROOT / ("outputs/genrecon_occlusion_weighting" + ("" if model == "vggt_omega" else f"_{model}"))


def depth_cue(model: str) -> str:
    """Name of the model's depth cue, the stem of its npz: vggt / solv3r."""
    return Path(MODELS[model][2]).stem


def prediction_npz(n: int, scene: str, model: str) -> Path:
    return ROOT / f"outputs/unposed_genrecon/{n}views/{scene}" / MODELS[model][2]


def load_prediction(path: Path):
    d = np.load(path)
    S = d["depth"].shape[0]
    depth = d["depth"].reshape(d["depth"].shape[:3]).astype(np.float32)
    conf = d["depth_conf"].reshape(depth.shape).astype(np.float32)
    extr = d["extrinsics"][:, :3, :4].astype(np.float64)  # world-to-camera, OpenCV
    K_pix = d["intrinsics"].astype(np.float64)  # at the depth's resolution
    names = [f"{i:04d}.png" for i in range(S)]  # convert.py's names for the GenRecon scene
    return d["image_paths"], names, depth, conf, extr, K_pix


def normalised(K_pix: np.ndarray, H: int, W: int) -> np.ndarray:
    K = K_pix.copy()
    K[:, 0] /= W
    K[:, 1] /= H
    return K.astype(np.float32)


def scan_depth(scene: str, n: int, image_paths, extr, K_pix, H: int, W: int):
    """The scan rendered at the predicted cameras, in their units. Returns depth and the transform."""
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


def check(pred_depth, pred_conf, gt_depth) -> dict:
    """The predicted depth against the scan, where the scan is hit and the prediction is confident (c > 0)."""
    per_view = []
    for dp, cp, dg in zip(pred_depth, pred_conf, gt_depth):
        m = np.isfinite(dg) & (dg > 0) & (cp > 0) & (dp > 0)
        if m.sum() < 100:
            per_view.append(None)
            continue
        r = dg[m] / dp[m]
        per_view.append({"pixels": int(m.sum()), "median_gt_over_pred": float(np.median(r)),
                         "absrel": float(np.mean(np.abs(dp[m] - dg[m]) / dg[m])),
                         "delta_1.25": float(np.mean(np.maximum(r, 1 / r) < 1.25))})
    ok = [v for v in per_view if v]
    mean = {k: float(np.mean([v[k] for v in ok])) for k in ("median_gt_over_pred", "absrel", "delta_1.25")} if ok else {}
    return {"mean": mean, "views": per_view}


def build(n: int, scene: str, model: str = "vggt_omega", conf_lo: float = CONF_LO, conf_hi: float = CONF_HI) -> Path:
    out, name = out_root(model) / f"cues/{n}views/{scene}", depth_cue(model)
    if (out / f"{name}.npz").exists() and (out / "gt.npz").exists():
        return out
    src = prediction_npz(n, scene, model)
    if not src.exists():
        raise FileNotFoundError(f"{src} missing, run experiments/unposed_genrecon/run_eval.py {n} --model {model} first")
    out.mkdir(parents=True, exist_ok=True)
    image_paths, names, depth, conf, extr, K_pix = load_prediction(src)
    H, W = depth.shape[1:]
    K = normalised(K_pix, H, W)

    c = np.clip((conf - conf_lo) / (conf_hi - conf_lo), 0, 1).astype(np.float32)
    np.savez_compressed(out / f"{name}.npz", names=names, depth=depth, conf=c, K=K)

    gt_depth, transform = scan_depth(scene, n, image_paths, extr, K_pix, H, W)
    hit = np.isfinite(gt_depth)
    np.savez_compressed(out / "gt.npz", names=names, depth=np.where(hit, gt_depth, 0).astype(np.float32),
                        conf=hit.astype(np.float32), K=K)

    report = {"model": model, "conf_lo": conf_lo, "conf_hi": conf_hi, "cameras_to_scan": transform,
              "scan_hit_fraction": float(hit.mean()), "c_mean": float(c.mean()), **check(depth, c, gt_depth)}
    (out / "check.json").write_text(json.dumps(report, indent=2))
    print(f"{scene}: {model} depth vs scan {report['mean']}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("views", type=int, help="number of input views")
    parser.add_argument("scenes", nargs="*", help="scene IDs (default: every scene with the model's npz)")
    parser.add_argument("--model", choices=MODELS, default="vggt_omega", help="whose cameras and depth")
    parser.add_argument("--conf-lo", type=float, default=CONF_LO)
    parser.add_argument("--conf-hi", type=float, default=CONF_HI)
    args = parser.parse_args()
    scenes = args.scenes or sorted(p.parent.name for p in (ROOT / f"outputs/unposed_genrecon/{args.views}views")
                                   .glob(f"*/{MODELS[args.model][2]}"))
    for scene in scenes:
        build(args.views, scene, args.model, args.conf_lo, args.conf_hi)


if __name__ == "__main__":
    main()
