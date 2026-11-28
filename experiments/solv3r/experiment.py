"""Solv3R: cameras and depth of the input views, from the images alone.

Writes cameras.json (frame arbitrary) and depth.npz with confidence, at Solv3R's resolution, with the cameras
fitted to the predicted rays. The images go in sorted by name, so the first one by name defines the frame.
infer.py does the work in Solv3R's env.
"""

from pathlib import Path

from evaluation.interface import Experiment, Job, run_in_env

ROOT = Path(__file__).resolve().parents[2]


class Solv3R(Experiment):
    stage = "cameras"
    defaults = {"calls": 16}  # solver calls, as in Solv3R's paper (Table 1)
    outputs = ("depth.npz", "cameras.json")

    def run(self, jobs: list[Job]):
        if jobs:
            run_in_env(ROOT / "papers/Solv3R/.venv/bin/python", Path(__file__).parent / "infer.py", jobs,
                       [str(jobs[0].settings["calls"])])


EXPERIMENT = Solv3R()
