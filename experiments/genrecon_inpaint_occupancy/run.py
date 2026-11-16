"""Run GenRecon with stage 1 inpainted around the depth's cells (cells.py) on the ScanNet++ evaluation inputs.

Same inputs and settings as experiments/genrecon_baseline/run.py, through inject.py. The variants are the sources
of cells.py, and the same with _fix, which also adds the occupied cells and removes the free ones afterwards.
Writes outputs/genrecon_inpaint_occupancy/<variant>/<N>views/<scene>/mesh.ply.zst and skips scenes that already
have one.

Usage:
    uv run experiments/genrecon_inpaint_occupancy/run.py {scan,solv3r,scan_fix,solv3r_fix} 32
"""

import argparse
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GENRECON = ROOT / "papers/GenRecon"
INJECT = ROOT / "experiments/genrecon_inpaint_occupancy/inject.py"
OUT = ROOT / "outputs/genrecon_inpaint_occupancy"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variant", choices=["scan", "solv3r", "scan_fix", "solv3r_fix"])
    parser.add_argument("views", type=int, nargs="+", help="numbers of input views")
    parser.add_argument("--scenes", nargs="*", help="default: every scene with cells from cells.py")
    args = parser.parse_args()
    source = args.variant.removesuffix("_fix")
    fix = ["--fix"] if args.variant.endswith("_fix") else []

    for n in args.views:
        cells_dir = OUT / source / f"cells/{n}views"
        for scene in args.scenes or sorted(p.stem for p in cells_dir.glob("*.npz")):
            out = OUT / args.variant / f"{n}views" / scene
            if (out / "mesh.ply.zst").exists():
                continue
            print(f"{scene}, {n} views", flush=True)
            out.mkdir(parents=True, exist_ok=True)
            with open(out / "run.log", "w") as log:
                result = subprocess.run(
                    [GENRECON / ".venv/bin/python", INJECT, cells_dir / f"{scene}.npz", *fix, "--mode", "Scannet_colmap",
                     "--path", ROOT / f"data/scannetpp/eval_inputs/{n}views" / scene, "--output_path", out,
                     "--ss_ckpt", "checkpoints/sparse_structure/ckpts/sparse_structure.pt",
                     "--shape_ckpt", "checkpoints/shape_slat/ckpts/shape_slat.pt",
                     "--tex_ckpt", "checkpoints/texture_slat/ckpts/texture_slat.pt",
                     "--pipeline_config", "configs/pipelines/original.json",
                     "--num_imgs_per_scene", str(n),
                     "--proj_batch_voxels", "1024"],
                    cwd=GENRECON, stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
                )
            if result.returncode != 0:
                print(f"  failed, see {out / 'run.log'}", flush=True)
                continue
            # Keep only the geometry, as in the baseline.
            subprocess.run(["zstd", "-q", "--rm", out / "mesh.ply"], check=True)
            (out / "to_glb_inputs.pt").unlink()


if __name__ == "__main__":
    main()
