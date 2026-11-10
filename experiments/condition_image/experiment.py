"""How strongly does GenRecon follow the per-chunk 2D conditioning image?

Same run as experiments/genrecon_baseline (same views, seed 42), with only the 2D image condition
changed. The 3D condition (features projected from all views) is left as is.

    none  normal GenRecon (reference)
    zero  the 2D condition is all zeros, which equals the negative condition, so guidance on the
          image is also gone

Usage (--scene runs one scene only; 286b55a2bf is the smallest eval scene):
    uv run experiments/condition_image/experiment.py none 32 --scene 286b55a2bf
    uv run experiments/condition_image/experiment.py zero 32 --scene 286b55a2bf
    uv run scripts/evaluate_scannetpp.py outputs/condition_image/zero/32views
"""

import argparse
import os
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GENRECON = ROOT / "papers/GenRecon"
MODES = ["none", "zero"]


def patch(mode):
    """Runs inside the GenRecon env: change the 2D condition, then run reconstruct_scene.py."""
    sys.path.insert(0, str(GENRECON))
    import torch
    from genrecon.pipelines.full_scene_images_to_3d import FullSceneImagesTo3DPipeline as Pipeline

    if mode == "zero":
        assemble = Pipeline._assemble_cond
        Pipeline._assemble_cond = staticmethod(lambda cond_2D, cond_3D: assemble(torch.zeros_like(cond_2D), cond_3D))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("views", type=int, nargs="+", help="numbers of input views")
    parser.add_argument("--scene", help="run only this scene ID")
    args = parser.parse_args()

    for n in args.views:
        for scene in sorted((ROOT / f"data/scannetpp/eval_inputs/{n}views").iterdir()):
            if args.scene and scene.name != args.scene:
                continue
            out = ROOT / f"outputs/condition_image/{args.mode}/{n}views" / scene.name
            if (out / "mesh.ply.zst").exists():
                continue
            print(f"{scene.name}, {n} views, {args.mode}", flush=True)
            out.mkdir(parents=True, exist_ok=True)
            with open(out / "run.log", "w") as log:
                result = subprocess.run(
                    [GENRECON / ".venv/bin/python", __file__, "--patch", args.mode,
                     "--mode", "Scannet_colmap", "--path", scene, "--output_path", out,
                     "--ss_ckpt", "checkpoints/sparse_structure/ckpts/sparse_structure.pt",
                     "--shape_ckpt", "checkpoints/shape_slat/ckpts/shape_slat.pt",
                     "--tex_ckpt", "checkpoints/texture_slat/ckpts/texture_slat.pt",
                     "--pipeline_config", "configs/pipelines/original.json",
                     "--num_imgs_per_scene", str(n), "--proj_batch_voxels", "1024", "--save_imgs"],
                    cwd=GENRECON, stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
                )
            if result.returncode != 0:
                print(f"  failed, see {out / 'run.log'}", flush=True)
                continue
            subprocess.run(["zstd", "-q", "--rm", out / "mesh.ply"], check=True)
            (out / "to_glb_inputs.pt").unlink()


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--patch":
        patch(sys.argv[2])
        sys.argv = ["reconstruct_scene.py", *sys.argv[3:]]
        runpy.run_path(str(GENRECON / "reconstruct_scene.py"), run_name="__main__")
    else:
        main()
