"""Images or video -> VGGT-Omega -> GenRecon -> scene.glb

Usage (from the repo root, in the genrecon env with VGGT-Omega installed):
    python pipeline/run.py path/to/images_or_video --out outputs/my_room

Each step runs as its own process (frees GPU memory in between) and is skipped
if its output already exists, so a failed step can be rerun on its own.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENRECON = ROOT / "papers/GenRecon"
CKPT = GENRECON / "checkpoints"


def run(cmd, cwd=ROOT):
    print("\n$ " + " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="Image folder or video file.")
    p.add_argument("--out", required=True)
    p.add_argument("--num-frames", type=int, default=32, help="Frames to sample from a video.")
    p.add_argument("--room-height", type=float, default=2.7, help="Floor-to-ceiling height in metres.")
    args = p.parse_args()

    out = Path(args.out).resolve()
    npz, scene, recon = out / "vggt/predictions.npz", out / "scene", out / "genrecon"
    py = sys.executable
    os.environ.setdefault("ATTN_BACKEND", "flash_attn")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "flash_attn")

    if not npz.exists():
        run([py, ROOT / "experiments/vggt_omega_inference/infer.py", Path(args.input).resolve(),
             "--num-frames", args.num_frames, "--output", npz])

    if not (scene / "colmap/points3D.txt").exists():
        run([py, ROOT / "pipeline/convert.py", npz, scene, "--room-height", args.room_height])

    if not (recon / "to_glb_inputs.pt").exists():
        run([py, "reconstruct_scene.py", "--mode", "Iphone", "--path", scene, "--colmap_subdir", "colmap",
             "--output_path", recon,
             "--ss_ckpt", CKPT / "ss/ckpts/sparse_structure.pt",
             "--shape_ckpt", CKPT / "shape/ckpts/shape_slat.pt",
             "--tex_ckpt", CKPT / "tex/ckpts/texture_slat.pt",
             "--num_imgs_per_scene", "999", "--chunk_size_factor", "1.08",
             "--stat_std_ratio", "3.0", "--radius_nb_points", "7", "--radius_m", "0.2",
             "--pipeline_config", "configs/pipelines/texture.json", "--proj_batch_voxels", "2048"],
            cwd=GENRECON)

    if not (recon / "scene.glb").exists():
        run([py, "chunked_to_glb.py", "--inputs", recon / "to_glb_inputs.pt",
             "--chunk_inputs", recon / "chunk_inputs.pt", "--output_dir", recon], cwd=GENRECON)

    print(f"\nDone: {recon / 'scene.glb'}")


if __name__ == "__main__":
    main()
