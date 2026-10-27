"""Unposed images or video -> VGGT-Omega -> GenRecon -> scene.glb

Usage (from anywhere; each step uses its own paper's uv env):
    python experiments/unposed_genrecon/run.py path/to/images_or_video --name my_scene

Steps, all writing to outputs/unposed_genrecon/<name>/:
    1. infer.py              VGGT-Omega env  -> vggt.npz
    2. convert.py            GenRecon env    -> scene/ (rgb/ + colmap/ for GenRecon's Iphone mode)
    3. reconstruct_scene.py  GenRecon env    -> genrecon/to_glb_inputs.pt
    4. chunked_to_glb.py     GenRecon env    -> genrecon/scene.glb

A step is skipped if its output already exists, so a failed run can be rerun
and continues where it stopped. Delete a step's output to redo it.
"""

import argparse
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs/unposed_genrecon"

VGGT_PYTHON = ROOT / "papers/VGGT-Omega/.venv/bin/python"
GENRECON = ROOT / "papers/GenRecon"
GENRECON_PYTHON = GENRECON / ".venv/bin/python"
CKPT = GENRECON / "checkpoints"


def run(cmd, cwd=ROOT, env=None):
    print("\n$ " + " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="Image folder or video file.")
    p.add_argument("--name", default=None, help="Run name, e.g. the scene ID. Default: the input's name.")
    p.add_argument("--num-images", type=int, default=32, help="Evenly spaced images (or video frames) to use.")
    p.add_argument("--room-height", type=float, default=2.7, help="Floor-to-ceiling height in metres.")
    args = p.parse_args()

    name = args.name or Path(args.input).stem
    out = OUTPUT_DIR / name
    npz, scene, recon = out / "vggt.npz", out / "scene", out / "genrecon"

    env = {**os.environ, "ATTN_BACKEND": "flash_attn", "SPARSE_ATTN_BACKEND": "flash_attn"}

    if not npz.exists():
        run([VGGT_PYTHON, HERE / "infer.py", Path(args.input).resolve(),
             "--num-images", args.num_images, "--name", name])

    if not (scene / "colmap/points3D.txt").exists():
        run([GENRECON_PYTHON, HERE / "convert.py", npz, scene, "--room-height", args.room_height])

    if not (recon / "to_glb_inputs.pt").exists():
        # Settings from GenRecon's README example for smartphone captures. 999 = use every image.
        run([GENRECON_PYTHON, "reconstruct_scene.py", "--mode", "Iphone", "--path", scene, "--colmap_subdir", "colmap",
             "--output_path", recon,
             "--ss_ckpt", CKPT / "sparse_structure/ckpts/sparse_structure.pt",
             "--shape_ckpt", CKPT / "shape_slat/ckpts/shape_slat.pt",
             "--tex_ckpt", CKPT / "texture_slat/ckpts/texture_slat.pt",
             "--num_imgs_per_scene", "999", "--chunk_size_factor", "1.08",
             "--stat_std_ratio", "3.0", "--radius_nb_points", "7", "--radius_m", "0.2",
             "--pipeline_config", "configs/pipelines/texture.json", "--proj_batch_voxels", "2048"],
            cwd=GENRECON, env=env)

    if not (recon / "scene.glb").exists():
        # Without this, CuMesh runs out of GPU memory (see papers/GenRecon/README_scene-reconstruction.md).
        glb_env = {**env, "PYTORCH_NO_CUDA_MEMORY_CACHING": "1"}
        run([GENRECON_PYTHON, "chunked_to_glb.py", "--inputs", recon / "to_glb_inputs.pt",
             "--chunk_inputs", recon / "chunk_inputs.pt", "--output_dir", recon], cwd=GENRECON, env=glb_env)

    if not (recon / "scene.glb").exists():
        raise SystemExit(f"chunked_to_glb.py did not write {recon / 'scene.glb'}; check its log for FAILED chunks.")
    print(f"\nDone: {recon / 'scene.glb'}")


if __name__ == "__main__":
    main()
