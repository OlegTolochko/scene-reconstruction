"""Run GenRecon on the ScanNet++ evaluation inputs written by scripts/select_views_scannetpp.py.

Writes outputs/genrecon_baseline/<N>views/<scene>/mesh.ply.zst (zstd -d restores mesh.ply) and
skips scenes that already have one.

Usage:
    uv run experiments/genrecon_baseline/run.py 32
"""

import argparse
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GENRECON = ROOT / "papers/GenRecon"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("views", type=int, nargs="+", help="numbers of input views")
    args = parser.parse_args()

    for n in args.views:
        for scene in sorted((ROOT / f"data/scannetpp/eval_inputs/{n}views").iterdir()):
            out = ROOT / f"outputs/genrecon_baseline/{n}views" / scene.name
            if (out / "mesh.ply.zst").exists():
                continue
            print(f"{scene.name}, {n} views", flush=True)
            out.mkdir(parents=True, exist_ok=True)
            with open(out / "run.log", "w") as log:
                result = subprocess.run(
                    [GENRECON / ".venv/bin/python", "reconstruct_scene.py", "--mode", "Scannet_colmap",
                     "--path", scene, "--output_path", out,
                     "--ss_ckpt", "checkpoints/sparse_structure/ckpts/sparse_structure.pt",
                     "--shape_ckpt", "checkpoints/shape_slat/ckpts/shape_slat.pt",
                     "--tex_ckpt", "checkpoints/texture_slat/ckpts/texture_slat.pt",
                     "--pipeline_config", "configs/pipelines/original.json",
                     "--num_imgs_per_scene", str(n),
                     # Fewer voxels at a time in the multi-view feature aggregation: same result, less
                     # GPU memory. With the default (8192), 256 views do not fit in 32 GB.
                     "--proj_batch_voxels", "1024"],
                    cwd=GENRECON, stdout=log, stderr=subprocess.STDOUT,
                    # The largest scenes peak close to 32 GB; this reduces memory fragmentation.
                    env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
                )
            if result.returncode != 0:
                print(f"  failed, see {out / 'run.log'}", flush=True)
                continue
            # Keep only the geometry: compress the mesh and delete the input for texture baking (0.5 GB+).
            subprocess.run(["zstd", "-q", "--rm", out / "mesh.ply"], check=True)
            (out / "to_glb_inputs.pt").unlink()


if __name__ == "__main__":
    main()
