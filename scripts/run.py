"""Run an experiment on the input views of a dataset.

Usage:
    uv run scripts/run.py EXPERIMENT --tag TAG --dataset DATASET --views N [N ...]
                          [--use_cameras gt | cameras/<experiment>/<tag>] [--scenes SCENE ...] [--SETTING VALUE ...]

EXPERIMENT is a folder in experiments/ with an experiment.py. Every other --NAME VALUE sets one of the
experiment's settings (values are read as JSON if they parse, e.g. 2, true or null, else as text); the
experiment's defaults fill in the rest. A recon experiment takes its cameras from a cameras run of the same
dataset and view count (--use_cameras cameras/<experiment>/<tag>), or from the dataset (--use_cameras gt).

Writes outputs/<stage>/<experiment>/<tag>/<dataset>/<N>views/:
    config.json   the experiment, settings, cameras run, date and git commit
    <scene>/      the standard files (evaluation/formats.py) and intermediate/

Scenes that are done are skipped, so running the same command again continues a run. A run refuses to
continue with a different config.json. Examples:

    uv run scripts/run.py solv3r --tag default --dataset scannetpp --views 8 32
    uv run scripts/run.py genrecon --tag solv3r --dataset scannetpp --views 8 32 --use_cameras cameras/solv3r/default
    uv run scripts/run.py genrecon --tag scan_cells --dataset scannetpp --views 32 --use_cameras gt --cells scan
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from evaluation import config as run_config
from evaluation import datasets
from evaluation.interface import Job

ROOT = Path(__file__).resolve().parents[1]


def load_experiment(name: str):
    path = ROOT / "experiments" / name / "experiment.py"
    spec = importlib.util.spec_from_file_location(f"experiments.{name}.experiment", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))  # for the experiment's own helper modules
    spec.loader.exec_module(module)
    return module.EXPERIMENT


def parse_settings(rest: list[str], defaults: dict) -> dict:
    """--NAME VALUE pairs on top of the defaults."""
    settings = dict(defaults)
    if len(rest) % 2 or not all(a.startswith("--") for a in rest[::2]):
        raise SystemExit(f"Expected --SETTING VALUE pairs, got: {' '.join(rest)}")
    for flag, value in zip(rest[::2], rest[1::2]):
        name = flag[2:]
        if name not in defaults:
            raise SystemExit(f"Unknown setting {flag}. Settings: " + ", ".join(f"--{k}" for k in defaults))
        try:
            settings[name] = json.loads(value)
        except json.JSONDecodeError:
            settings[name] = value
    return settings


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment")
    parser.add_argument("--tag", required=True, help="names the run, with the experiment, dataset and view count")
    parser.add_argument("--dataset", required=True, choices=datasets.DATASETS)
    parser.add_argument("--views", type=int, nargs="+", required=True, help="numbers of input views")
    parser.add_argument("--use_cameras", help="recon only: gt, or a cameras run as cameras/<experiment>/<tag>")
    parser.add_argument("--scenes", nargs="+", help="only these scenes (default: all of the dataset)")
    args, rest = parser.parse_known_args()

    experiment = load_experiment(args.experiment)
    settings = parse_settings(rest, experiment.defaults)
    if experiment.stage == "cameras" and args.use_cameras:
        raise SystemExit("--use_cameras is for recon experiments; a cameras experiment computes its own.")
    if experiment.stage == "recon" and not args.use_cameras:
        raise SystemExit("A recon experiment needs --use_cameras gt or --use_cameras cameras/<experiment>/<tag>.")
    try:
        experiment.check(settings, args.use_cameras)
    except ValueError as e:
        raise SystemExit(f"{args.experiment}: {e}")

    dataset = datasets.get(args.dataset)
    scenes = args.scenes or dataset.scenes()
    any_failed = False
    for n in args.views:
        run_dir = ROOT / "outputs" / experiment.stage / args.experiment / args.tag / args.dataset / f"{n}views"
        cameras_dir = None
        if args.use_cameras and args.use_cameras != "gt":
            cameras_dir = ROOT / "outputs" / args.use_cameras / args.dataset / f"{n}views"
            if not (cameras_dir / "config.json").exists():
                raise SystemExit(f"No cameras run at {cameras_dir.relative_to(ROOT)}")
        run_config.check_or_write(run_dir / "config.json", run_config.build(
            experiment=args.experiment, stage=experiment.stage, tag=args.tag, dataset=args.dataset, views=n,
            use_cameras=args.use_cameras, settings=settings))

        jobs, missing = [], []
        for scene in scenes:
            job = Job(dataset, scene, n, dataset.input_views(scene, n), run_dir / scene, settings,
                      cameras_dir / scene if cameras_dir else None)
            if experiment.done(job):
                continue
            if job.cameras and not (job.cameras / "cameras.json").exists():
                missing.append(scene)
                continue
            jobs.append(job)
        print(f"{run_dir.relative_to(ROOT)}: {len(scenes) - len(jobs) - len(missing)} scenes done, "
              f"{len(jobs)} to run" + (f", {len(missing)} without cameras in the cameras run" if missing else ""),
              flush=True)
        experiment.run(jobs)
        failed = [job.scene for job in jobs if not experiment.done(job)]
        if failed:
            any_failed = True
            print(f"{len(failed)} scenes failed: {' '.join(failed)}", flush=True)
    if any_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
