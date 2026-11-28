"""What an experiment implements, and the job it gets for each scene.

An experiment is experiments/<name>/experiment.py with a module-level EXPERIMENT, an instance of a subclass
of Experiment. scripts/run.py runs it in the root env; the experiment starts its paper's env itself, as a
subprocess, for the parts that need it.

Stages:
    cameras  images -> cameras.json, plus depth.npz or points.ply if the method gives them
    recon    images + the cameras of a cameras run (--use_cameras) or the dataset's -> mesh.ply.zst, ...

Every file in a scene folder is in the frame of that folder's cameras.json (see evaluation/formats.py).
"""

import json
import os
import subprocess
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path

from evaluation.datasets import Dataset, View


@dataclass
class Job:
    dataset: Dataset
    scene: str
    n: int  # number of input views
    views: list[View]  # the input views, with the dataset's cameras
    out: Path  # the scene folder of this run
    settings: dict
    cameras: Path | None = None  # recon: the scene folder of the cameras run, None for the dataset's cameras

    @property
    def intermediate(self) -> Path:
        """Files a run needs only while it runs, or for debugging; safe to delete."""
        path = self.out / "intermediate"
        path.mkdir(parents=True, exist_ok=True)
        return path


class Experiment:
    stage: str
    defaults: dict = {}
    outputs: tuple[str, ...] = ()  # the files a finished scene has; the last one written marks it done

    def done(self, job: Job) -> bool:
        return all((job.out / name).exists() for name in self.outputs)

    def check(self, settings: dict, use_cameras: str | None):
        """Raise ValueError if the settings do not go together. Called once per run, before any scene."""

    def run(self, jobs: list[Job]):
        """Run the jobs one by one; one failed scene does not stop the others."""
        for job in jobs:
            print(f"{job.scene}, {job.n} views", flush=True)
            job.out.mkdir(parents=True, exist_ok=True)
            try:
                self.run_scene(job)
            except Exception:
                (job.intermediate / "error.txt").write_text(traceback.format_exc())
                print(f"  failed, see {job.intermediate / 'error.txt'}", flush=True)

    def run_scene(self, job: Job):
        raise NotImplementedError


def run_in_env(python: Path, script: Path, jobs: list[Job], args: list[str] = ()):
    """Run script with another env's python on all jobs at once, e.g. so a model is loaded only once.

    script gets a JSON file listing {"images": [...], "names": [...], "out": scene folder} per job, the views
    sorted by name, then args.
    """
    specs = []
    for job in jobs:
        job.out.mkdir(parents=True, exist_ok=True)
        views = sorted(job.views, key=lambda v: v.name)
        specs.append({"images": [str(v.image) for v in views], "names": [v.name for v in views], "out": str(job.out)})
    with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
        json.dump(specs, f)
        f.flush()
        # Less memory fragmentation: without it, 256 views do not fit in 32 GB.
        env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        subprocess.run([python, script, f.name, *args], env=env, check=True)
