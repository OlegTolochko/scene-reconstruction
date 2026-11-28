"""GenRecon: a mesh from the input views and their cameras.

With the dataset's cameras (--use_cameras gt), GenRecon runs its ScanNet++ mode on the scene folder with the
N input views, as in GenRecon's paper: its chunks come from ScanNet++'s own COLMAP points (built from all
images of the scene), and the mesh comes out in the scan's frame.

With a cameras run, GenRecon runs its Iphone mode, the mode that reads any COLMAP-format scene: the run's
cameras, with the confident depth points to place the chunks (scene.py). Its settings are GenRecon's
defaults, which are the ScanNet++ mode's too, plus --center_crop: the ScanNet++ mode takes one square crop
per image, the Iphone mode two by default.

--cells chooses the cells stage 2 gets: genrecon (stage 1's own), or, with the dataset's cameras, scan or
depth (cells.py, through inject.py). Stage 1 runs only when its cells are used.

Writes cameras.json (the cameras GenRecon used; its mesh and cells are in their frame), cells.npz (the cells
stage 2 got) and mesh.ply.zst (geometry only), and keeps GenRecon's other files in intermediate/.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import cells as sources  # experiments/genrecon/cells.py
from evaluation.datasets import true_cameras
from evaluation.formats import save_mesh
from evaluation.interface import Experiment, Job
from scene import write_scene  # experiments/genrecon/scene.py

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GENRECON = ROOT / "papers/GenRecon"


class GenRecon(Experiment):
    stage = "recon"
    defaults = {
        "cells": "genrecon",  # genrecon, scan or depth
        "depth": None,  # depth cells: "scan", or the cameras run whose depth is used, cameras/<experiment>/<tag>
        "min_views": 1,  # depth cells: views whose depth points a cell must hold
        "keep_unseen": False,  # depth cells: also keep stage 1's cells outside every input image
    }
    outputs = ("cells.npz", "mesh.ply.zst")

    def check(self, settings: dict, use_cameras: str | None):
        if settings["cells"] not in ("genrecon", "scan", "depth"):
            raise ValueError(f"--cells must be genrecon, scan or depth, not {settings['cells']}")
        if settings["cells"] != "genrecon" and use_cameras != "gt":
            raise ValueError("--cells scan and depth need --use_cameras gt")
        if settings["cells"] == "depth" and not (settings["depth"] == "scan" or str(settings["depth"]).startswith("cameras/")):
            raise ValueError("--cells depth needs --depth scan or --depth cameras/<experiment>/<tag>")

    def run_scene(self, job: Job):
        s, work = job.settings, job.intermediate
        recon = work / "genrecon"
        if job.cameras is None:
            cameras = true_cameras(job.views)
            mode = ["--mode", "Scannet_colmap", "--path", job.dataset.input_dir(job.scene, job.n),
                    "--num_imgs_per_scene", job.n]
        else:
            cameras = write_scene(work / "scene", job.cameras, job.views)
            mode = ["--mode", "Iphone", "--path", work / "scene", "--colmap_subdir", "colmap",
                    "--num_imgs_per_scene", 999, "--center_crop"]
        cameras.save(job.out / "cameras.json")

        script = ["reconstruct_scene.py"]
        if s["cells"] != "genrecon":
            depth = s["depth"]
            if depth and depth != "scan":
                depth = str(ROOT / "outputs" / depth / job.dataset.name / f"{job.n}views" / job.scene / "depth.npz")
            spec = {"cells": s["cells"], "dataset": job.dataset.name, "scene": job.scene, "depth": depth,
                    "cameras": str(job.out / "cameras.json"), "min_views": s["min_views"], "keep_unseen": s["keep_unseen"]}
            (work / "cells_source.json").write_text(json.dumps(spec, indent=2) + "\n")
            script = [HERE / "inject.py", work / "cells_source.json"]

        env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}  # large scenes fit in 32 GB
        cmd = [GENRECON / ".venv/bin/python", *script, *mode, "--output_path", recon,
               "--ss_ckpt", "checkpoints/sparse_structure/ckpts/sparse_structure.pt",
               "--shape_ckpt", "checkpoints/shape_slat/ckpts/shape_slat.pt",
               "--tex_ckpt", "checkpoints/texture_slat/ckpts/texture_slat.pt",
               "--pipeline_config", "configs/pipelines/original.json",  # geometry only
               # fewer voxels at a time in the feature aggregation: same result, less GPU memory
               "--proj_batch_voxels", 1024]
        with open(work / "genrecon.log", "w") as log:
            subprocess.run([str(c) for c in cmd], cwd=GENRECON, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

        grid = sources.layout(recon)
        grid.cells = sources.stage2_cells(recon, grid)
        grid.save(job.out / "cells.npz")
        save_mesh(recon / "mesh.ply", job.out / "mesh.ply.zst")
        # The large files: the uncompressed mesh, GenRecon's GLB inputs (texture), the image links.
        (recon / "mesh.ply").unlink()
        (recon / "to_glb_inputs.pt").unlink(missing_ok=True)
        shutil.rmtree(work / "scene" / "rgb", ignore_errors=True)


EXPERIMENT = GenRecon()
