"""Offline benchmark (paper Table 1): DTU, ETH3D, 7-Scenes, ScanNet++ and nuScenes.

Solv3R runs inside the evaluation pipeline of DVLT (https://github.com/nv-tlabs/dvlt): the same loaders and
preprocessing, the same Sim(3) alignment and the same metrics. Cameras are fitted to the predicted rays and world
points are the depth maps unprojected with these cameras, as for DVLT.
Reported: pose AUC@3 / AUC@30, point-map relative L2 (absrel_world) and inlier ratio (inlier_world@3%).

Data: <data-root>/test/{dtu,eth3d,sevenscenes,scannetpp,nuscenes}, prepared with DVLT's preprocessing scripts.

usage:
  python eval/offline.py --data-root /path/to/dvlt_bench --out results/offline                  # 16 calls (Table 1)
  python eval/offline.py --data-root /path/to/dvlt_bench --out results/offline_adaptive --adaptive 0.02
  python eval/offline.py ... --datasets dtu,eth3d,sevenscenes
"""
import argparse
import json
import os
import sys

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from dvlt.common.constants import DataField
from dvlt.config.schema import register_configs
from dvlt.model.base import Module
from dvlt.model.dvlt.model import DVLT

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solv3r.model import Solv3R, autocast  # noqa: E402

DATASETS = ("dtu", "eth3d", "sevenscenes", "scannetpp", "nuscenes")


class Solv3REval(Module):
    """DVLT benchmark model: encode, fixed or adaptive number of calls, decode, DVLT post-processing."""

    def __init__(self, checkpoint, calls=16, adaptive=None):
        self.checkpoint = checkpoint
        self.calls, self.adaptive = calls, adaptive
        self.calls_used, self.dataset = [], ""
        self.patch_size = 14
        self.use_depth_conf_for_pose = False
        self.world_points_from_rays = False
        super().__init__()

    def build_model(self):
        return Solv3R.from_pretrained(self.checkpoint)

    def load_pretrained(self, *args, **kwargs):
        return [], []

    def train_step(self, *args, **kwargs):
        raise NotImplementedError

    _postprocess_predictions = DVLT._postprocess_predictions

    @torch.no_grad()
    def predict(self, batch, accelerator):
        model = self.model
        images = batch[DataField.IMAGES][0]
        ctx = model.encode(images, chunk=images.shape[0])
        z = model.start(ctx)
        with autocast():
            if self.adaptive is None:
                for _ in range(self.calls):
                    z = model.step(z, ctx)
                used = self.calls
            else:
                tau, lo, hi = self.adaptive
                for used in range(1, hi + 1):
                    zn = model.step(z, ctx)
                    change = float((zn.float() - z.float()).norm() / z.float().norm().clamp_min(1e-9))
                    z = zn
                    if used >= lo and change < tau:
                        break
        pred = {k: v[None] for k, v in model.decode(z, ctx).items()}       # DVLT post-processing takes a batch dim
        self.calls_used.append(dict(dataset=self.dataset, views=ctx.S, calls=used))
        return self._postprocess_predictions(batch, pred)


class _DatasetName:
    """Trainer callback: tells the model which test dataset is running."""

    def __init__(self, model):
        self.model = model

    def on_test_dataset_start(self, trainer, dataset_name, *args, **kwargs):
        self.model.dataset = dataset_name

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *args, **kwargs: {}


def summarize(out_dir):
    rows = {}
    for ds in sorted(os.listdir(out_dir)):
        f = os.path.join(out_dir, ds, "metrics.json")
        if not os.path.isfile(f):
            continue
        m = json.load(open(f))
        pm, po = m["pointmap"], m["pose"]
        rows[ds] = {"AUC@3": 100 * po["Auc_3"], "AUC@30": 100 * po["Auc_30"],
                    "Rel.L2": pm["absrel_world"], "IR": 100 * pm["inlier_world@3%"]}
        r = rows[ds]
        print(f"{ds:12s} AUC@3 {r['AUC@3']:5.1f}  AUC@30 {r['AUC@30']:5.1f}  Rel.L2 {r['Rel.L2']:.3f}  IR {r['IR']:5.1f}")
    json.dump(rows, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None, help="a .safetensors file or a Hugging Face repo id (default: checkpoints/ if present, else the released weights)")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--calls", type=int, default=16)
    ap.add_argument("--adaptive", type=float, default=None,
                    help="stop once the relative state change of a call is below this value (paper: 0.02)")
    ap.add_argument("--min-calls", type=int, default=4)
    ap.add_argument("--max-calls", type=int, default=64)
    a = ap.parse_args()
    unknown = sorted(set(a.datasets.split(",")) - set(DATASETS))
    if unknown:
        ap.error(f"unknown --datasets {','.join(unknown)} (choose from {','.join(DATASETS)})")

    from accelerate import PartialState

    PartialState()
    register_configs()
    datasets = a.datasets.split(",")
    drop = [f"~data.test_datasets.{d}" for d in DATASETS if d not in datasets]
    with initialize_config_dir(config_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config"),   # DVLT's benchmark configs
                               version_base=None):
        cfg = compose(config_name="dvlt", overrides=[
            "data=benchmark", f"user.data_root={a.data_root}", f"trainer.output_dir={a.out}",
            "trainer.experiment_name=solv3r", "trainer.timestamp=eval", "trainer.experiment_logger=[]",
            "data.distributed_eval=false", "model.load_patch_embed_weights=false", *drop])

    adaptive = (a.adaptive, a.min_calls, a.max_calls) if a.adaptive is not None else None
    model = Solv3REval(a.checkpoint, calls=a.calls, adaptive=adaptive)
    trainer = instantiate(cfg.trainer, model=model, data=instantiate(cfg.data))
    trainer.setup(mode="test")
    model.setup_test(trainer.accelerator)
    trainer.callbacks["_dataset_name"] = _DatasetName(model)
    out = os.path.join(str(trainer.output_dir), f"adaptive{a.adaptive}" if adaptive else f"calls{a.calls}")
    trainer._test_loop(mode="test", output_dir=out)
    if trainer.accelerator.is_main_process:
        summarize(out)
        json.dump(model.calls_used, open(os.path.join(out, "calls.json"), "w"))
        if adaptive:
            for ds in datasets:
                c = [r["calls"] for r in model.calls_used if r["dataset"] == ds]
                if c:
                    print(f"{ds:12s} calls: mean {np.mean(c):.1f}, min {min(c)}, max {max(c)}")


if __name__ == "__main__":
    main()
