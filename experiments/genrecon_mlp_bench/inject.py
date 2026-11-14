"""GenRecon's reconstruct_scene.py with a bench variant as the aggregator of the sparse-structure stage.

GenRecon is not modified. This patches FullSceneImagesTo3DPipeline._build_global_dense_cond: in the
sparse-structure stage it puts the variant (variants.py) in place of the flow model's aggregator, loads the run's
finetuned weights, and wraps project_features_on_points to give the aggregator its depth cues for each batch of
voxels, as train.py's trainer does. The shape and texture stages are GenRecon's.
(Adapted from genrecon_depth_cue_finetuning/inject.py.)

Runs in GenRecon's environment, from papers/GenRecon (run_scannetpp.py does this):
    .venv/bin/python ../../experiments/genrecon_mlp_bench/inject.py --cue_depth DEPTH.npz \
        (--bench_run RUN_DIR [--bench_step N] [--bench_no_ema] | --bench_variant NAME) <reconstruct_scene.py arguments>

DEPTH.npz (run_scannetpp.py) holds one depth map per image of the scene: names, depth [S, H, W], conf
[S, H, W] (the model's raw confidence) and K [S, 3, 3] normalised to the full image, in the units of
GenRecon's cameras. RUN_DIR is a run folder of train.py: run.json names the variant and the confidence
standardisation, ckpts/ holds the aggregator (the EMA weights by default). --bench_variant uses a variant
untrained: one that starts at GenRecon then has to reproduce GenRecon's mesh.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.cwd()))  # papers/GenRecon
sys.path.insert(0, str(Path(__file__).resolve().parent))
import aggregator as agg_lib  # noqa: E402
import variants  # noqa: E402, F401  (registers the variants)


def checkpoint(run: Path, step, ema: bool) -> Path:
    pattern = ("denoiser_ema*_step" if ema else "denoiser_step") + (f"{step:07d}.pt" if step else "*.pt")
    found = sorted((run / "ckpts").glob(pattern))
    if not found:
        raise FileNotFoundError(f"no {pattern} in {run / 'ckpts'}")
    return found[-1]


def load_variant(base, variant: str, ckpt):
    """The variant around GenRecon's aggregator `base` (on its device, in its dtype), with a checkpoint's weights."""
    agg = agg_lib.build(variant, base)
    if ckpt is not None:
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        sub = {k.removeprefix("aggregator."): v for k, v in state.items() if k.startswith("aggregator.")}
        agg.load_state_dict(sub, strict=True)  # every aggregator weight, GenRecon's and the variant's
    return agg.eval()


class CueState:
    def __init__(self, args, out: Path):
        self.args, self.out = args, out
        self.model = None  # the sparse-structure flow model, once the variant is in place
        self.active = False
        if args.bench_run:
            info = json.loads((args.bench_run / "run.json").read_text())
            self.variant, self.conf_mean, self.conf_std = info["variant"], info["conf_mean"], info["conf_std"]
            if info["code_hash"] != agg_lib.source_hash(self.variant):
                raise SystemExit(f"The code of variant {self.variant!r} changed since {args.bench_run} was trained: "
                                 "its weights belong to the old code.")
            self.ckpt = checkpoint(args.bench_run, args.bench_step, not args.bench_no_ema)
        else:
            self.variant, self.conf_mean, self.conf_std, self.ckpt = args.bench_variant, 0.0, 1.0, None

    def setup(self, flow_model, device):
        if self.model is flow_model:
            return
        flow_model.aggregator = load_variant(flow_model.aggregator, self.variant, self.ckpt)
        new = {n: round(float(p.float().norm()), 4) for n, p in flow_model.aggregator.named_parameters()
               if not n.startswith(("feature_mlp.", "weight_mlp."))}
        print(f"[bench] variant {self.variant}, checkpoint {self.ckpt}, conf ({self.conf_mean:.2f}, "
              f"{self.conf_std:.2f}), norms of the new layers {new}", flush=True)

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
        print(f"[bench] {len(views)} views, voxel {self.voxel:.4f} camera units", flush=True)

    def set_cues(self, points, extrinsics):
        m = self.maps
        cues = agg_lib.compute_cues(points, extrinsics, m["K"], m["depth"], m["conf"], self.voxel,
                                    self.conf_mean, self.conf_std)  # [1, N, V, 4]
        known = cues[..., agg_lib.KNOWN] > 0
        self.stats["pairs"] += known.numel()
        self.stats["known"] += int(known.sum())
        self.stats["occluded"] += int((known & (cues[..., agg_lib.BEHIND] > 1)).sum())  # > 1 voxel behind the depth
        self.model.aggregator.cues = cues


def install(state: CueState):
    from genrecon.pipelines import full_scene_images_to_3d as fs

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
            (state.out / "cue_stats.json").write_text(json.dumps(state.stats, indent=2))

    project = fs.project_features_on_points

    def project_with_cues(proj, points, cond, extrinsics, intrinsics, eps=1e-6):
        out = project(proj, points, cond, extrinsics, intrinsics, eps=eps)
        if state.active:
            state.set_cues(points, extrinsics)
        return out

    P._build_global_dense_cond = build_with_cues
    fs.project_features_on_points = project_with_cues


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)  # the rest goes to reconstruct_scene.py
    p.add_argument("--cue_depth", type=Path, required=True, help="depth maps of the scene (run_scannetpp.py)")
    p.add_argument("--bench_run", type=Path, help="run folder of train.py to load the aggregator from")
    p.add_argument("--bench_step", type=int, help="checkpoint step (default: the last)")
    p.add_argument("--bench_no_ema", action="store_true", help="the raw weights instead of the EMA")
    p.add_argument("--bench_variant", help="a variant of variants.py, untrained")
    return p


def main():
    import reconstruct_scene

    args, rest = build_parser().parse_known_args(sys.argv[1:])
    if (args.bench_run is None) == (args.bench_variant is None):
        raise SystemExit("give exactly one of --bench_run and --bench_variant")
    sys.argv = [sys.argv[0], *rest]
    out = Path(reconstruct_scene.build_parser().parse_args(rest).output_path)
    out.mkdir(parents=True, exist_ok=True)
    (out / "bench_args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    install(CueState(args, out))
    reconstruct_scene.main()


if __name__ == "__main__":
    main()
