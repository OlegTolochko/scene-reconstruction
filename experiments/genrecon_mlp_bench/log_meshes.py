"""Adds the ScanNet++ mesh metrics of a bench run to its W&B run, so the runs table compares them too.

The mean over the scenes of every metric that scripts/evaluate_scannetpp.py wrote (metrics.json per scene) goes
to <run>/scannetpp/<N>views/mean_metrics.json and into the run's W&B summary as scannetpp_<N>views/<metric>.

    papers/GenRecon/.venv/bin/python experiments/genrecon_mlp_bench/log_meshes.py RUN_DIR 32
"""

import json
import sys
from pathlib import Path


def mean_metrics(pred_dir: Path) -> dict:
    scenes = [json.loads(p.read_text()) for p in sorted(pred_dir.glob("*/metrics.json"))]
    if not scenes:
        raise SystemExit(f"no */metrics.json in {pred_dir}: run scripts/evaluate_scannetpp.py on it first")
    return {"scenes": len(scenes), **{k: sum(s[k] for s in scenes) / len(scenes) for k in scenes[0]}}


def main():
    run_dir, views = Path(sys.argv[1]).resolve(), int(sys.argv[2])
    pred_dir = run_dir / "scannetpp" / f"{views}views"
    mean = mean_metrics(pred_dir)
    (pred_dir / "mean_metrics.json").write_text(json.dumps(mean, indent=2) + "\n")
    print(json.dumps(mean, indent=2))
    try:
        import wandb

        run = wandb.init(id=(run_dir / "wandb_id.txt").read_text().strip(), resume="allow", name=run_dir.name,
                         group=run_dir.parent.name, dir=str(run_dir))
        run.summary.update({f"scannetpp_{views}views/{k}": v for k, v in mean.items()})
        run.finish()
    except Exception as e:  # the metrics are on disk either way
        print(f"W&B: could not add the metrics to the run ({e})")


if __name__ == "__main__":
    main()
