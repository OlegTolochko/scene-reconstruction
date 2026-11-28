"""COLMAP structure from motion on the input views, as a camera baseline.

COLMAP gets the N input images and none of their poses or intrinsics: one shared PINHOLE camera (the images
are undistorted and from one camera), SIFT features, exhaustive matching, incremental mapping. Models of 2
images and up are kept (COLMAP's default drops those under 10, so 8 views could never give one), and the
model with the most registered images is used.

Writes cameras.json (the registered images only, frame arbitrary), and in intermediate/ the database, every
model COLMAP built, the used model as text and summary.json.
Runs in the root env (pycolmap), on every core of the machine; SIFT needs about 1 GB of RAM per core.
"""

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pycolmap

from evaluation.formats import Cameras
from evaluation.interface import Experiment, Job

pycolmap.logging.minloglevel = 1  # warnings and errors only; COLMAP logs every step otherwise


class Colmap(Experiment):
    stage = "cameras"
    outputs = ("cameras.json",)

    def run_scene(self, job: Job):
        work = job.intermediate
        images = work / "images"
        images.mkdir(exist_ok=True)
        for view in job.views:
            if not (images / view.name).exists():
                (images / view.name).symlink_to(view.image.resolve())

        # From scratch: a partial database of a failed run would be reused otherwise.
        database, sparse = work / "database.db", work / "sparse"
        database.unlink(missing_ok=True)
        shutil.rmtree(sparse, ignore_errors=True)
        sparse.mkdir()
        start = time.time()
        pycolmap.extract_features(database, images, camera_mode=pycolmap.CameraMode.SINGLE,
                                  reader_options=pycolmap.ImageReaderOptions(camera_model="PINHOLE"))
        pycolmap.match_exhaustive(database)
        models = pycolmap.incremental_mapping(database, images, sparse,
                                              options=pycolmap.IncrementalPipelineOptions(min_model_size=2))
        summary = {"views": len(job.views), "models": len(models), "registered": 0, "seconds": time.time() - start}

        names, K, world_to_camera, width, height = [], [], [], [], []
        if models:
            model = max(models.values(), key=lambda m: m.num_reg_images())
            (work / "colmap").mkdir(exist_ok=True)
            model.write_text(work / "colmap")
            for image in sorted((im for im in model.images.values() if im.has_pose), key=lambda im: im.name):
                camera = model.cameras[image.camera_id]
                names.append(image.name)
                K.append(camera.calibration_matrix())
                world_to_camera.append(image.cam_from_world().matrix())
                width.append(camera.width)
                height.append(camera.height)
            summary.update(registered=len(names), points=model.num_points3D(),
                           reprojection_error_px=model.compute_mean_reprojection_error())
        (work / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"  {summary['registered']}/{summary['views']} registered, {summary['seconds']:.0f} s", flush=True)
        Cameras(names, np.array(K).reshape(-1, 3, 3), np.array(world_to_camera).reshape(-1, 3, 4), np.array(width, dtype=int),
                np.array(height, dtype=int), "arbitrary").save(job.out / "cameras.json")


EXPERIMENT = Colmap()
