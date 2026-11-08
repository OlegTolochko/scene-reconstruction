"""Run VGGT-Omega + GenRecon on the ScanNet++ evaluation inputs written by scripts/select_views_scannetpp.py.

For each scene, using the same N DSLR views as the GenRecon baseline but none of their poses:
    1. VGGT-Omega predicts cameras and depth       -> vggt.npz
    2. convert.py writes them as a COLMAP scene     -> scene/
    3. GenRecon (Iphone mode) generates the mesh    -> genrecon/
    4. The mesh is moved into the scan's frame with the similarity transform (rotation, translation,
       scale) that best maps the predicted camera centres onto the ScanNet++ ones -> mesh.ply.zst

The ScanNet++ poses are used only in step 4, to make the mesh comparable with the scan.

Writes outputs/unposed_genrecon/<N>views/<scene>/ and skips scenes that already have mesh.ply.zst.

With --matched, GenRecon uses the GenRecon baseline's settings wherever Iphone mode allows (one centre
crop per image, default chunk size, statistical outlier ratio 2.0) and effectively no radius outlier filter, whose
fixed distance means nothing at VGGT-Omega's arbitrary scale. Results go to
outputs/unposed_genrecon_matched/<N>views/<scene>/. Both runs share the VGGT-Omega output (vggt.npz),
which stays in outputs/unposed_genrecon/<N>views/<scene>/.

Usage:
    uv run experiments/unposed_genrecon/run_eval.py 32 [--matched]
    uv run scripts/evaluate_scannetpp.py outputs/unposed_genrecon/32views   # or unposed_genrecon_matched
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
VGGT_PYTHON = ROOT / "papers/VGGT-Omega/.venv/bin/python"
GENRECON = ROOT / "papers/GenRecon"
GENRECON_PYTHON = GENRECON / ".venv/bin/python"
CKPT = GENRECON / "checkpoints"

# GenRecon settings that differ between the two runs
SMARTPHONE = ["--chunk_size_factor", "1.08", "--stat_std_ratio", "3.0", "--radius_nb_points", "7", "--radius_m", "0.2"]
# The radius filter is effectively off: it only drops points with no other point within 0.05 units, which
# the dense VGGT-Omega cloud hardly has. Open3D rejects 0 neighbours, and a large radius would be very slow.
MATCHED = ["--center_crop", "--stat_std_ratio", "2.0", "--radius_nb_points", "1", "--radius_m", "0.05"]

# ScanNet++ nerfstudio world -> scan frame (same matrix as scripts/evaluate_scannetpp.py)
NERFSTUDIO_TO_SCAN = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=float)


def umeyama(src, dst):
    """Scale s, rotation R, translation t minimising |dst - (s R src + t)|^2 over the point pairs."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    x, y = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(y.T @ x / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / (x**2).sum(1).mean()
    return s, R, mu_d - s * R @ mu_s


def predicted_centres(images_txt):
    """Camera centres by image name from the COLMAP images.txt that convert.py wrote."""
    centres = {}
    for line in images_txt.read_text().splitlines():
        parts = line.split()
        if len(parts) != 10:
            continue
        qw, qx, qy, qz, tx, ty, tz = map(float, parts[1:8])
        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        centres[parts[9]] = -R.T @ np.array([tx, ty, tz])
    return centres


def run(cmd, log, cwd=ROOT, env=None):
    log.write(f"\n$ {' '.join(map(str, cmd))}\n")
    log.flush()
    subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def process(scene_in, out, shared, genrecon_args, script=("reconstruct_scene.py",)):
    """shared: folder holding the VGGT-Omega input and output, the same for both runs.
    script: GenRecon's entry point and its leading arguments, run from papers/GenRecon."""
    frames = json.loads((scene_in / "dslr/nerfstudio/transforms_undistorted.json").read_text())["frames"]
    npz, scene, recon = shared / "vggt.npz", out / "scene", out / "genrecon"

    # The selected views only, as an image folder for infer.py
    images = shared / "images"
    images.mkdir(parents=True, exist_ok=True)
    for f in frames:
        link = images / Path(f["file_path"]).name
        if not link.exists():
            link.symlink_to((scene_in / "dslr/resized_undistorted_images" / f["file_path"]).resolve())

    env = {**os.environ, "ATTN_BACKEND": "flash_attn", "SPARSE_ATTN_BACKEND": "flash_attn",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    with open(out / "run.log", "a") as log:
        if not npz.exists():
            run([VGGT_PYTHON, HERE / "infer.py", images, "--num-images", len(frames), "--output", npz], log)
        if not (recon / "mesh.ply").exists():
            run([GENRECON_PYTHON, HERE / "convert.py", npz, "--out", scene], log)
            # Geometry only (as the baseline). 999 = every image.
            run([GENRECON_PYTHON, *script, "--mode", "Iphone", "--path", scene,
                 "--colmap_subdir", "colmap", "--output_path", recon,
                 "--ss_ckpt", CKPT / "sparse_structure/ckpts/sparse_structure.pt",
                 "--shape_ckpt", CKPT / "shape_slat/ckpts/shape_slat.pt",
                 "--tex_ckpt", CKPT / "texture_slat/ckpts/texture_slat.pt",
                 "--num_imgs_per_scene", "999", *genrecon_args,
                 "--pipeline_config", "configs/pipelines/original.json", "--proj_batch_voxels", "1024"],
                log, cwd=GENRECON, env=env)

    # Predicted image i (convert.py's {i:04d}.png) is the i-th image path in the npz
    names = [Path(p).name for p in np.load(npz)["image_paths"]]
    pred = predicted_centres(scene / "colmap/images.txt")
    gt = {Path(f["file_path"]).name: (NERFSTUDIO_TO_SCAN @ np.array(f["transform_matrix"]))[:3, 3] for f in frames}
    src = np.array([pred[f"{i:04d}.png"] for i in range(len(names))])
    dst = np.array([gt[name] for name in names])
    s, R, t = umeyama(src, dst)
    rmse = float(np.sqrt(((s * src @ R.T + t - dst) ** 2).sum(1).mean()))

    mesh = trimesh.load(recon / "mesh.ply", process=False)
    mesh.vertices = s * mesh.vertices @ R.T + t
    mesh.export(out / "mesh.ply")
    (out / "align.json").write_text(json.dumps(
        {"scale": float(s), "rotation": R.tolist(), "translation": t.tolist(), "camera_rmse_m": rmse}, indent=2))

    # Keep only what the eval needs: compress the mesh, drop the large GenRecon and image files.
    subprocess.run(["zstd", "-q", "--rm", "-f", out / "mesh.ply"], check=True)
    for big in (recon / "mesh.ply", recon / "to_glb_inputs.pt", scene / "rgb"):
        if big.is_dir():
            shutil.rmtree(big)
        elif big.exists():
            big.unlink()
    return len(frames), rmse


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("views", type=int, nargs="+", help="numbers of input views")
    parser.add_argument("--matched", action="store_true", help="GenRecon settings matched to the baseline")
    args = parser.parse_args()
    out_root = ROOT / ("outputs/unposed_genrecon_matched" if args.matched else "outputs/unposed_genrecon")
    genrecon_args = MATCHED if args.matched else SMARTPHONE

    for n in args.views:
        for scene_in in sorted((ROOT / f"data/scannetpp/eval_inputs/{n}views").iterdir()):
            out = out_root / f"{n}views" / scene_in.name
            shared = ROOT / f"outputs/unposed_genrecon/{n}views" / scene_in.name
            if (out / "mesh.ply.zst").exists():
                continue
            print(f"{scene_in.name}, {n} views", flush=True)
            out.mkdir(parents=True, exist_ok=True)
            shared.mkdir(parents=True, exist_ok=True)
            try:
                views, rmse = process(scene_in, out, shared, genrecon_args)
                print(f"  done, {views} views, camera alignment error {rmse:.3f} m", flush=True)
            except Exception as e:  # one bad scene should not stop the rest
                print(f"  failed ({e}), see {out / 'run.log'}", flush=True)


if __name__ == "__main__":
    main()
