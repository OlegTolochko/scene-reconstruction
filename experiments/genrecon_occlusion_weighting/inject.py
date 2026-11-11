"""GenRecon's reconstruct_scene.py, with the multi-view aggregation weighted by per-view visibility.

GenRecon itself is not modified. This patches, in FullSceneImagesTo3DPipeline:
  _build_global_dense_cond / _build_global_sparse_cond   to know the stage and its voxel size,
  project_features_on_points                              to compute v for each batch of voxels,
and the forward of both MultiViewFeatAggregator classes, which use v (visibility.py) instead of the mask.
The trained weights, the DiTs, chunking and voting are untouched.

Runs in GenRecon's environment, from papers/GenRecon (run.py does this):
    .venv/bin/python ../../experiments/genrecon_occlusion_weighting/inject.py --vis_cues CUES.npz [--vis_* options] \
        <reconstruct_scene.py arguments>

CUES.npz (cues.py) holds one depth map per image of the scene: names, depth [S, H, W], conf [S, H, W]
in [0, 1] and K [S, 3, 3] normalised to the full image. Depth is in the units of GenRecon's cameras.
--vis_identity sets v = valid (no cues needed), which must reproduce plain GenRecon.

--vis_dump OUT.npz stops after stage 1's conditioning, before any sampling, and writes for every cell of
stage 1's global grid (C order, shape `grid`, R cells per chunk side) the variance each --vis_var_mode
would give the aggregator, as the mean over channels (weighted, original, around_vmean), and the
visibility of the views (sum_valid, sum_v). variance_check.py uses it.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.cwd()))  # papers/GenRecon
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reconstruct_scene  # noqa: E402
import visibility as vis  # noqa: E402
from genrecon.modules.cond_3D import aggregation_net, sparse_aggregation_net  # noqa: E402
from genrecon.pipelines import full_scene_images_to_3d as fs  # noqa: E402

STAGES = ("ss", "shape", "tex")
DELTA_BINS = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]  # Delta / tau


class Visibility:
    """Holds the cues, the current stage and the v of the last projected batch."""

    def __init__(self, args, out: Path):
        self.args, self.out = args, out
        self.p = vis.Params(args.vis_tau_vox, args.vis_rho, args.vis_softness, args.vis_lam, args.vis_eps,
                            args.vis_kappa, args.vis_var_mode)
        self.stages = set(args.vis_stages.split(","))
        self.cues = None  # loaded on first use, after get_images has written cameras.json
        self.stage = None
        self.pending = None
        self.stats = {}
        self.dump = {"sum_valid": [], "sum_v": [], **{m: [] for m in vis.VAR_MODES}}

    def load(self, device):
        if self.cues is not None or self.args.vis_identity:
            return
        # Scene views in GenRecon's order. Both square crops of an image share its img_path, and the cues
        # use the full image's intrinsics, so each crop looks up the right pixels.
        views = [Path(v["img_path"]).name for v in json.loads((self.out / "cameras.json").read_text())["scene"]]
        cues = np.load(self.args.vis_cues)
        index = {str(n): i for i, n in enumerate(cues["names"])}
        missing = [n for n in views if n not in index]
        if missing:
            raise RuntimeError(f"no cue for views {missing}")
        sel = [index[n] for n in views]
        self.cues = {k: torch.from_numpy(cues[k][sel]).float().to(device) for k in ("depth", "conf", "K")}
        chunks = json.loads((self.out / "chunk_transforms.json").read_text())["chunks"]
        self.chunk_size = chunks[0]["M_chunk_to_original"][0][0]  # camera units per chunk-frame unit
        print(f"[visibility] {len(views)} views, chunk size {self.chunk_size:.3f}, {self.p}", flush=True)

    def begin(self, stage: str, resolution: int):
        if stage in self.stages:
            self.stage, self.resolution = stage, resolution
            self.stats.setdefault(stage, {"pairs": 0, "pairs_v_below_half": 0, "voxels": 0,
                                          "voxels_fallback": 0, "sum_v": 0.0, "low_conf_pairs": 0,
                                          "delta_over_tau_hist": [0] * (len(DELTA_BINS) + 1)})

    def end(self):
        if self.stage is not None:
            s = self.stats[self.stage]
            s["mean_sum_v"] = s["sum_v"] / max(s["voxels"], 1)
            s["delta_over_tau_bins"] = DELTA_BINS
            (self.out / "visibility_stats.json").write_text(json.dumps(self.stats, indent=2))
        self.stage, self.pending = None, None

    def weights(self, points, extrinsics, valid):
        """points [S, 3] chunk-0 frame, extrinsics [N, 4, 4], valid [N, S] -> v [N, S] (float32)."""
        if self.args.vis_identity:
            return valid.float()
        self.load(points.device)
        c = self.cues
        voxel_size = self.chunk_size / self.resolution
        # The chunk-0 extrinsics include the chunk scale, so their camera frame is in camera units.
        v, delta_rel, conf = vis.visibility(points, extrinsics, c["K"], c["depth"], c["conf"], valid, voxel_size, self.p)
        v, alpha = vis.fallback(v, valid, self.p.kappa)
        s = self.stats[self.stage]
        s["pairs"] += int(valid.sum())
        s["pairs_v_below_half"] += int((valid & (v < 0.5)).sum())
        s["low_conf_pairs"] += int((valid & (conf < 0.5)).sum())
        seen = valid.any(0)
        s["voxels"] += int(seen.sum())
        s["voxels_fallback"] += int((seen & (alpha < 1)).sum())
        s["sum_v"] += float(v.sum())
        bins = torch.bucketize(delta_rel[valid], torch.tensor(DELTA_BINS, device=v.device))
        for b, n in zip(*torch.unique(bins, return_counts=True)):
            s["delta_over_tau_hist"][int(b)] += int(n)
        return v

    def record(self, feats, mask, v):
        """The per-cell values of --vis_dump for one batch of the dense aggregator: v [B, N, V]."""
        with torch.no_grad():
            self.dump["sum_valid"].append(mask.sum(1)[0].cpu())
            self.dump["sum_v"].append(v.sum(1)[0].float().cpu())
            for mode in vis.VAR_MODES:
                var = vis.moments(feats, mask, v, mode, one_pass=True)[1]  # [B, 1, V, D]
                self.dump[mode].append(var[0, 0].float().mean(-1).cpu())

    def write_dump(self, grid, R: int):
        arrays = {k: torch.cat(x).numpy() for k, x in self.dump.items()}
        if len(arrays["sum_v"]) != int(np.prod(grid)):
            raise RuntimeError(f"dumped {len(arrays['sum_v'])} cells, the grid {grid} has {int(np.prod(grid))}")
        np.savez_compressed(self.args.vis_dump, grid=np.array(grid), R=R, **arrays)
        print(f"[visibility] wrote {self.args.vis_dump}, stopping before sampling", flush=True)

    def take(self, shape):
        v, self.pending = self.pending, None
        if v is not None and tuple(v.shape) != tuple(shape):
            raise RuntimeError(f"visibility {tuple(v.shape)} does not match the aggregator input {tuple(shape)}")
        return v


def stage_of(pipeline, flow_model) -> str:
    key = next(k for k, m in pipeline.models.items() if m is flow_model)
    return "ss" if key.startswith("sparse_structure") else key.split("_")[0]  # shape / tex


def install(state: Visibility):
    def wrap_build(build):
        def wrapped(self, flow_model, *args, **kwargs):
            stage = stage_of(self, flow_model)
            state.begin(stage, flow_model.resolution)
            try:
                out = build(self, flow_model, *args, **kwargs)
            finally:
                state.end()
            if state.args.vis_dump and stage == "ss":
                rel_t, R = args[4], args[5]  # as in _build_global_dense_cond's signature
                state.write_dump(self._global_grid_layout(rel_t, R)[1], R)
                raise SystemExit(0)
            return out
        return wrapped

    P = fs.FullSceneImagesTo3DPipeline
    P._build_global_dense_cond = wrap_build(P._build_global_dense_cond)
    P._build_global_sparse_cond = wrap_build(P._build_global_sparse_cond)

    project = fs.project_features_on_points

    def project_with_visibility(proj, points, cond, extrinsics, intrinsics, eps=1e-6):
        out = project(proj, points, cond, extrinsics, intrinsics, eps=eps)
        if state.stage is not None:
            state.pending = state.weights(points, extrinsics[0], out[1][0])  # [N, S]
        return out

    fs.project_features_on_points = project_with_visibility

    dense, sparse = aggregation_net.MultiViewFeatAggregator, sparse_aggregation_net.MultiViewFeatAggregator
    dense_orig, sparse_orig = dense.forward, sparse.forward

    def dense_forward(self, feats, mask, camera_emb=None):
        v = state.take(mask.shape[1:])  # [N, V]
        if v is None:
            return dense_orig(self, feats, mask, camera_emb)
        if state.args.vis_dump:
            state.record(feats, mask, v[None])
            return dense_orig(self, feats, mask, camera_emb)
        return vis.dense_forward(self, feats, mask, v[None], state.p, camera_emb)

    def sparse_forward(self, x, mask):
        v = state.take(mask.shape[::-1])  # [N, S]
        if v is None:
            return sparse_orig(self, x, mask)
        return vis.sparse_forward(self, x, mask, v.T.contiguous(), state.p)

    dense.forward, sparse.forward = dense_forward, sparse_forward


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)  # the rest goes to reconstruct_scene.py
    d = vis.Params()
    p.add_argument("--vis_cues", type=Path, help="cue file from cues.py (not needed with --vis_identity)")
    p.add_argument("--vis_identity", action="store_true", help="v = valid: must reproduce plain GenRecon")
    p.add_argument("--vis_stages", default=",".join(STAGES), help="comma-separated subset of ss,shape,tex")
    p.add_argument("--vis_tau_vox", type=float, default=d.tau_vox)
    p.add_argument("--vis_rho", type=float, default=d.rho)
    p.add_argument("--vis_softness", type=float, default=d.softness)
    p.add_argument("--vis_lam", type=float, default=d.lam)
    p.add_argument("--vis_eps", type=float, default=d.eps)
    p.add_argument("--vis_kappa", type=float, default=d.kappa)
    p.add_argument("--vis_var_mode", choices=vis.VAR_MODES, default=d.var_mode, help="variance the aggregator gets")
    p.add_argument("--vis_dump", type=Path, help="write stage 1's per-cell variances here and stop (variance_check.py)")
    return p


def main():
    args, rest = build_parser().parse_known_args(sys.argv[1:])
    if not args.vis_identity and args.vis_cues is None:
        raise SystemExit("--vis_cues is required unless --vis_identity")
    if not set(args.vis_stages.split(",")) <= set(STAGES):
        raise SystemExit(f"--vis_stages must be a subset of {STAGES}")
    if args.vis_dump and (args.vis_identity or "ss" not in args.vis_stages.split(",")):
        raise SystemExit("--vis_dump needs cues and the ss stage")
    sys.argv = [sys.argv[0], *rest]  # the reconstruct_scene.py arguments
    out = Path(reconstruct_scene.build_parser().parse_args(rest).output_path)
    out.mkdir(parents=True, exist_ok=True)
    (out / "visibility_args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    install(Visibility(args, out))
    reconstruct_scene.main()


if __name__ == "__main__":
    main()
