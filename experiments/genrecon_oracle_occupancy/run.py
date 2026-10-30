"""Run GenRecon with the scan's visible cells (inject.py) on the ScanNet++ evaluation inputs.

Same inputs and settings as experiments/genrecon_baseline/run.py. Needs the cells from occupancy.py.
Writes outputs/genrecon_oracle_occupancy/<N>views/<scene>/mesh.ply.zst and skips scenes that already
have one.

Usage:
    uv run experiments/genrecon_oracle_occupancy/run.py 32
"""

import argparse
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GENRECON = ROOT / "papers/GenRecon"
OUT = ROOT / "outputs/genrecon_oracle_occupancy"


def reconstruct(scene: str, n: int, out: Path, oracle: bool) -> int:
    """Run GenRecon as in the baseline, with the scan's cells if oracle. Returns the exit code."""
    script = [Path(__file__).parent / "inject.py", OUT / f"gt/{n}views/{scene}.npz"] if oracle else ["reconstruct_scene.py"]
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "run.log", "w") as log:
        return subprocess.run(
            [GENRECON / ".venv/bin/python", *script, "--mode", "Scannet_colmap",
             "--path", ROOT / f"data/scannetpp/eval_inputs/{n}views" / scene, "--output_path", out,
             "--ss_ckpt", "checkpoints/sparse_structure/ckpts/sparse_structure.pt",
             "--shape_ckpt", "checkpoints/shape_slat/ckpts/shape_slat.pt",
             "--tex_ckpt", "checkpoints/texture_slat/ckpts/texture_slat.pt",
             "--pipeline_config", "configs/pipelines/original.json",
             "--num_imgs_per_scene", str(n),
             "--proj_batch_voxels", "1024"],
            cwd=GENRECON, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        ).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("views", type=int, nargs="+", help="numbers of input views")
    args = parser.parse_args()

    for n in args.views:
        for gt in sorted((OUT / f"gt/{n}views").glob("*.npz")):
            out = OUT / f"{n}views" / gt.stem
            if (out / "mesh.ply.zst").exists():
                continue
            print(f"{gt.stem}, {n} views", flush=True)
            if reconstruct(gt.stem, n, out, oracle=True) != 0:
                print(f"  failed, see {out / 'run.log'}", flush=True)
                continue
            # Keep only the geometry, as in the baseline.
            subprocess.run(["zstd", "-q", "--rm", out / "mesh.ply"], check=True)
            (out / "to_glb_inputs.pt").unlink()


if __name__ == "__main__":
    main()
