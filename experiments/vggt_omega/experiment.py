"""VGGT-Omega: cameras and depth of the input views, from the images alone.

Writes cameras.json (frame arbitrary) and depth.npz with confidence, at the network's resolution. The images
go in sorted by name, so the first one by name defines the frame. infer.py does the work in VGGT-Omega's env.
"""

from pathlib import Path

from evaluation.interface import Experiment, Job, run_in_env

ROOT = Path(__file__).resolve().parents[2]


class VGGTOmega(Experiment):
    stage = "cameras"
    outputs = ("depth.npz", "cameras.json")

    def run(self, jobs: list[Job]):
        if jobs:
            run_in_env(ROOT / "papers/VGGT-Omega/.venv/bin/python", Path(__file__).parent / "infer.py", jobs)


EXPERIMENT = VGGTOmega()
