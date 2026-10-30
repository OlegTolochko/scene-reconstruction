"""Textured (baked) meshes of the baseline and the oracle, for viewer.py.

The runs of genrecon_baseline/run.py and run.py delete GenRecon's texture-baking inputs, so this reruns
both on a few scenes with 32 views and bakes the result with GenRecon's chunked_to_glb.py. For the browser,
each chunk is simplified to 400k faces (GenRecon's default: 2M) with 2048^2 textures (default: 4096^2).

Writes outputs/genrecon_oracle_occupancy/baked/<baseline|oracle>/<scene>/scene.glb.

Usage:
    uv run experiments/genrecon_oracle_occupancy/bake.py 5f99900f09 cc5237fd77 27dd4da69e
"""

import argparse
import os
import subprocess

from run import GENRECON, OUT, reconstruct


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenes", nargs="+")
    args = parser.parse_args()

    for scene in args.scenes:
        for run in ("baseline", "oracle"):
            out = OUT / "baked" / run / scene
            if (out / "scene.glb").exists():
                continue
            print(f"{scene}, {run}", flush=True)
            if reconstruct(scene, 32, out, oracle=run == "oracle") != 0:
                print(f"  failed, see {out / 'run.log'}", flush=True)
                continue
            with open(out / "glb.log", "w") as log:
                subprocess.run(
                    [GENRECON / ".venv/bin/python", "chunked_to_glb.py", "--inputs", out / "to_glb_inputs.pt",
                     "--chunk_inputs", out / "chunk_inputs.pt", "--output_dir", out,
                     "--simplify_threshold", "400000", "--texture_size", "2048"],
                    cwd=GENRECON, stdout=log, stderr=subprocess.STDOUT,
                    # CuMesh allocates GPU memory itself (see papers/GenRecon/README_scene-reconstruction.md)
                    env={**os.environ, "PYTORCH_NO_CUDA_MEMORY_CACHING": "1"},
                )
            if not (out / "scene.glb").exists():  # chunked_to_glb.py exits with 0 on failure
                print(f"  baking failed, see {out / 'glb.log'}", flush=True)
                continue
            subprocess.run(["zstd", "-q", "--rm", out / "mesh.ply"], check=True)
            (out / "to_glb_inputs.pt").unlink()


if __name__ == "__main__":
    main()
