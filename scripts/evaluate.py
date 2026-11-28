"""Score runs: every standard file of every scene against the dataset, then the mean over the scenes.

Usage:
    uv run scripts/evaluate.py [RUN ...] [--workers N]

RUN is <stage>/<experiment>/<tag> (default: every run in outputs/). For each finished scene folder (one with
cameras.json in a cameras run, mesh.ply.zst in a recon run) without metrics.json, this writes metrics.json
with one group per kind:

    alignment  cameras in an arbitrary frame only: the fit of the camera centres onto the dataset's (align.py),
               scale and camera_rmse_m. Depth, cells and mesh are scored after it.
    cameras    cameras runs: metrics/cameras.py, plus registered, the share of input views with a camera
    depth      depth.npz: metrics/depth.py
    cells      cells.npz: metrics/cells.py
    mesh       mesh.ply.zst: metrics/mesh.py

Then results/<stage>/<experiment>/<tag>/<dataset>_<N>views.json: the run's config, the number of scenes, the
mean of every metric over them, and every scene's metrics. Each scene is scored in a new process, since
Open3D does not release all memory between scenes; --workers scenes at a time.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import trimesh

from evaluation import datasets
from evaluation.align import transform_cameras, umeyama
from evaluation.formats import Cameras, Cells, Depth, load_mesh
from evaluation.metrics import cameras as camera_metrics
from evaluation.metrics import cells as cell_metrics
from evaluation.metrics import depth as depth_metrics
from evaluation.metrics import mesh as mesh_metrics

ROOT = Path(__file__).resolve().parents[1]
FINISHED = {"cameras": "cameras.json", "recon": "mesh.ply.zst"}  # the file a finished scene of each stage has


def evaluate_scene(scene_dir: Path, dataset_name: str, n: int, stage: str) -> dict:
    dataset = datasets.get(dataset_name)
    views = dataset.input_views(scene_dir.name, n)
    true = {view.name: view for view in views}
    cameras = Cameras.load(scene_dir / "cameras.json")
    out, fit = {}, None
    if cameras.frame == "arbitrary":
        if len(cameras.names) < 3:  # e.g. COLMAP registered too few images: nothing to align or score
            return {"cameras": {"registered": len(cameras.names) / n}}
        predicted, true_centres = cameras.centres, np.array([true[name].cam_to_world[:3, 3] for name in cameras.names])
        fit = umeyama(predicted, true_centres)
        rmse = np.sqrt(((fit.apply(predicted) - true_centres) ** 2).sum(axis=1).mean())
        out["alignment"] = {"scale": float(fit.scale), "camera_rmse_m": float(rmse)}
        if stage == "cameras":
            out["cameras"] = {**camera_metrics.evaluate(cameras, views), "registered": len(cameras.names) / n}

    surface = dataset.surface(scene_dir.name)
    if (scene_dir / "depth.npz").exists():
        out["depth"] = depth_metrics.evaluate(Depth.load(scene_dir / "depth.npz"), cameras, views, surface)
    if (scene_dir / "cells.npz").exists():
        grid = Cells.load(scene_dir / "cells.npz")
        in_metres = cameras if fit is None else transform_cameras(cameras, fit, "dataset")
        cells = cell_metrics.true_cells(surface, grid, in_metres, None if fit is None else fit.inverse())
        out["cells"] = cell_metrics.compare(grid.cells, cells["all"], grid.shape, cells["visible"])
    if (scene_dir / "mesh.ply.zst").exists():
        mesh = load_mesh(scene_dir / "mesh.ply.zst")
        if fit is not None:
            mesh = trimesh.Trimesh(fit.apply(mesh.vertices), mesh.faces, process=False)
        out["mesh"] = mesh_metrics.evaluate(mesh, dataset, scene_dir.name)
    return out


def score(task: tuple) -> tuple[Path, dict | str]:
    scene_dir = task[0]
    try:
        return scene_dir, evaluate_scene(*task)
    except Exception as e:  # one bad scene should not stop the rest
        return scene_dir, f"{type(e).__name__}: {e}"


def mean(per_scene: list[dict]) -> dict:
    """The mean of every metric over the scenes that have it (NaN, e.g. a recall with nothing to recall, left out)."""
    groups = sorted({g for m in per_scene for g in m})
    names = {g: sorted({k for m in per_scene for k in m.get(g, {})}) for g in groups}
    return {g: {k: float(np.nanmean([m[g][k] for m in per_scene if k in m.get(g, {})])) for k in names[g]}
            for g in groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", help="<stage>/<experiment>/<tag> (default: all)")
    parser.add_argument("--workers", type=int, default=1, help="scenes scored at a time (each needs a few GB of RAM)")
    args = parser.parse_args()

    runs = [ROOT / "outputs" / run for run in args.runs] or sorted(
        p for stage in ("cameras", "recon") for p in (ROOT / "outputs" / stage).glob("*/*") if p.is_dir())
    run_dirs = sorted(p.parent for run in runs for p in run.glob("*/*views/config.json"))
    if not run_dirs:
        raise SystemExit("No runs found.")

    tasks = []
    for run_dir in run_dirs:
        config = json.loads((run_dir / "config.json").read_text())
        for scene_dir in sorted(p for p in run_dir.iterdir() if (p / FINISHED[config["stage"]]).exists()):
            if not (scene_dir / "metrics.json").exists():
                tasks.append((scene_dir, config["dataset"], config["views"], config["stage"]))
    print(f"{len(tasks)} scenes to score in {len(run_dirs)} runs", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1) as pool:
        for scene_dir, result in pool.map(score, tasks):
            if isinstance(result, str):
                print(f"failed {scene_dir.relative_to(ROOT)}: {result}", flush=True)
                continue
            (scene_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
            print(f"scored {scene_dir.relative_to(ROOT)}", flush=True)

    for run_dir in run_dirs:
        config = json.loads((run_dir / "config.json").read_text())
        per_scene = {p.parent.name: json.loads(p.read_text()) for p in sorted(run_dir.glob("*/metrics.json"))}
        if not per_scene:
            continue
        rel = run_dir.relative_to(ROOT / "outputs")  # <stage>/<experiment>/<tag>/<dataset>/<N>views
        path = ROOT / "results" / rel.parent.parent / f"{config['dataset']}_{config['views']}views.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"config": config, "scenes": len(per_scene), "mean": mean(list(per_scene.values())),
                                    "per_scene": per_scene}, indent=1) + "\n")
        print(f"wrote {path.relative_to(ROOT)} ({len(per_scene)} scenes)", flush=True)


if __name__ == "__main__":
    main()
