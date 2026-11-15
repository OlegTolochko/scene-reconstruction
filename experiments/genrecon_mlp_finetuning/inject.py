"""GenRecon's reconstruct_scene.py with the depth-cue aggregator of variant A in the sparse-structure stage.

GenRecon is not modified. This patches FullSceneImagesTo3DPipeline._build_global_dense_cond: in the
sparse-structure stage it adds the cue layers (cues.py) to the flow model's aggregator, loads the finetuned
aggregator, and wraps project_features_on_points to set the aggregator's cues for each batch of voxels, as
train.py's trainer does. The shape and texture stages are GenRecon's.

Runs in GenRecon's environment, from papers/GenRecon (run_scannetpp.py does this):
    .venv/bin/python ../../experiments/genrecon_mlp_finetuning/inject.py --cue_depth DEPTH.npz \
        (--cue_run RUN_DIR [--cue_step N] | --cue_zero_init) <reconstruct_scene.py arguments>

DEPTH.npz (run_scannetpp.py) holds one depth map per image of the scene: names, depth [S, H, W], conf
[S, H, W] (the model's raw confidence) and K [S, 3, 3] normalised to the full image, in the units of
GenRecon's cameras. RUN_DIR is a train.py output_dir: its cues.json gives use_cues and the confidence
standardisation, ckpts/ the aggregator (EMA weights by default). With use_cues false (the control) the cues
are 0. --cue_zero_init uses GenRecon's aggregator with zero cue layers, which must reproduce GenRecon exactly
(--cue_extra footprint similarity adds the extra cues of cues.py to it; with --cue_run they come from cues.json).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.cwd()))  # papers/GenRecon
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cues as cue_lib  # noqa: E402
import reconstruct_scene  # noqa: E402
from genrecon.pipelines import full_scene_images_to_3d as fs  # noqa: E402


def checkpoint(run: Path, step, ema: bool) -> Path:
    pattern = ("denoiser_ema*_step" if ema else "denoiser_step") + (f"{step:07d}.pt" if step else "*.pt")
    found = sorted((run / "ckpts").glob(pattern))
    if not found:
        raise FileNotFoundError(f"no {pattern} in {run / 'ckpts'}")
    return found[-1]


class CueState:
    def __init__(self, args, out: Path):
        self.args, self.out = args, out
        self.model = None  # the sparse-structure flow model, once its aggregator has the cue layers
        self.active = False
        if args.cue_zero_init:
            self.use_cues, self.conf_mean, self.conf_std, self.ckpt = True, 0.0, 1.0, None
            self.extra = tuple(args.cue_extra)
        else:
            info = json.loads((args.cue_run / "cues.json").read_text())
            self.use_cues, self.conf_mean, self.conf_std = info["use_cues"], info["conf_mean"], info["conf_std"]
            self.ckpt = checkpoint(args.cue_run, args.cue_step, not args.cue_no_ema)
            self.extra = tuple(info.get("extra_cues", ()))
            if args.cue_off:  # the trained aggregator with all cues 0: how much does the model rely on them?
                self.use_cues = False

    def setup(self, flow_model, device):
        if self.model is flow_model:
            return
        agg = cue_lib.add_cue_layers(flow_model.aggregator, self.extra)
        agg.cues_enabled = self.use_cues
        if self.ckpt is not None:
            state = torch.load(self.ckpt, map_location="cpu", weights_only=True)
            sub = {k.removeprefix("aggregator."): v for k, v in state.items() if k.startswith("aggregator.")}
            agg.load_state_dict(sub, strict=True)  # every aggregator weight, the cue layers included
        norm = {n: round(float(p.norm()), 4) for n, p in agg.named_parameters() if n.startswith("cue_")}
        print(f"[cues] checkpoint {self.ckpt}, use_cues {self.use_cues}, extra {self.extra}, conf ({self.conf_mean:.2f}, "
              f"{self.conf_std:.2f}), cue layer norms {norm}", flush=True)

        # Scene views in GenRecon's order; both square crops of an image share its img_path, and the depth
        # maps use the full image's intrinsics, so each crop looks up the right pixels.
        views = [Path(v["img_path"]).name for v in json.loads((self.out / "cameras.json").read_text())["scene"]]
        d = np.load(self.args.cue_depth)
        index = {str(n): i for i, n in enumerate(d["names"])}
        missing = [n for n in views if n not in index]
        if missing:
            raise RuntimeError(f"no depth for views {missing}")
        sel = [index[n] for n in views]
        self.maps = {k: torch.from_numpy(d[k][sel]).float().to(device)[None] for k in ("depth", "conf", "K")}
        chunks = json.loads((self.out / "chunk_transforms.json").read_text())["chunks"]
        chunk_size = chunks[0]["M_chunk_to_original"][0][0]  # camera units per chunk-frame unit
        self.voxel = chunk_size / flow_model.resolution
        self.model = flow_model
        self.stats = {"pairs": 0, "known": 0, "occluded": 0}
        print(f"[cues] {len(views)} views, voxel {self.voxel:.4f} camera units", flush=True)

    def set_cues(self, points, extrinsics, intrinsics, patch_res):
        m = self.maps
        cues = cue_lib.compute_cues(points, extrinsics, m["K"], m["depth"], m["conf"], self.voxel,
                                    self.conf_mean, self.conf_std)  # [1, N, V, 4]
        known = cues[..., 3] > 0
        self.stats["pairs"] += known.numel()
        self.stats["known"] += int(known.sum())
        self.stats["occluded"] += int((known & (cues[..., 0] > 1)).sum())  # > 1 voxel behind the observed depth
        self.model.aggregator.cues = cues if self.use_cues else torch.zeros_like(cues)
        if "footprint" in self.extra:
            # intrinsics: of the crops the features come from, unlike the depth maps' m["K"]
            self.model.aggregator.footprint = cue_lib.footprint_cue(points, extrinsics, intrinsics, self.voxel,
                                                                    patch_res)


def install(state: CueState):
    P = fs.FullSceneImagesTo3DPipeline
    build = P._build_global_dense_cond

    def build_with_cues(self, flow_model, scene_feats, cond2d_feats, scene_ext_c0, scene_intr, *args, **kwargs):
        if flow_model is not self.models["sparse_structure_flow_model"] or not hasattr(flow_model, "aggregator"):
            return build(self, flow_model, scene_feats, cond2d_feats, scene_ext_c0, scene_intr, *args, **kwargs)
        state.setup(flow_model, scene_feats.device)
        state.active = True
        try:
            return build(self, flow_model, scene_feats, cond2d_feats, scene_ext_c0, scene_intr, *args, **kwargs)
        finally:
            state.active = False
            flow_model.aggregator.cues = None
            flow_model.aggregator.footprint = None
            (state.out / "cue_stats.json").write_text(json.dumps(state.stats, indent=2))

    project = fs.project_features_on_points

    def project_with_cues(proj, points, cond, extrinsics, intrinsics, eps=1e-6):
        out = project(proj, points, cond, extrinsics, intrinsics, eps=eps)
        if state.active:
            state.set_cues(points, extrinsics, intrinsics, proj.img_patch_res)
        return out

    P._build_global_dense_cond = build_with_cues
    fs.project_features_on_points = project_with_cues


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)  # the rest goes to reconstruct_scene.py
    p.add_argument("--cue_depth", type=Path, required=True, help="depth maps of the scene (run_scannetpp.py)")
    p.add_argument("--cue_run", type=Path, help="train.py output_dir to load the aggregator from")
    p.add_argument("--cue_step", type=int, help="checkpoint step (default: the last)")
    p.add_argument("--cue_no_ema", action="store_true", help="the raw weights instead of the EMA")
    p.add_argument("--cue_off", action="store_true", help="with --cue_run: set all cues to 0 at test time")
    p.add_argument("--cue_zero_init", action="store_true", help="GenRecon's aggregator + zero cue layers")
    p.add_argument("--cue_extra", nargs="*", default=[], choices=cue_lib.EXTRA_CUES,
                   help="with --cue_zero_init: extra cues for the zero cue layers")
    return p


def main():
    args, rest = build_parser().parse_known_args(sys.argv[1:])
    if args.cue_zero_init == (args.cue_run is not None):
        raise SystemExit("give exactly one of --cue_run and --cue_zero_init")
    if args.cue_extra and not args.cue_zero_init:
        raise SystemExit("--cue_extra only with --cue_zero_init (a run's extra cues come from its cues.json)")
    sys.argv = [sys.argv[0], *rest]
    out = Path(reconstruct_scene.build_parser().parse_args(rest).output_path)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cue_args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    install(CueState(args, out))
    reconstruct_scene.main()


if __name__ == "__main__":
    main()
