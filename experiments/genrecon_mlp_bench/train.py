"""Finetunes one aggregator variant (variants.py) of GenRecon's sparse-structure stage, with depth cues.

GenRecon is not modified. This file
  - registers DepthRoomSparseStructureLatent and MlpBenchTrainer with GenRecon's registries,
  - wraps build_single_model to load the released GenRecon checkpoint, put the variant in place of the aggregator
    and freeze everything else,
and then runs papers/GenRecon/train.py unchanged, in this module's namespace (so its mp.spawn finds `main`).
The dataset, the cue cameras and the way the cues reach the model are those of
experiments/genrecon_depth_cue_finetuning (feat/depth-cue-finetuning); bench.py adds the repeatable validation
loss, Weights & Biases and the comparison with the baseline.

From papers/GenRecon, in GenRecon's env (run.sbatch does this):
    python ../../experiments/genrecon_mlp_bench/train.py --variant NAME [--seed N] --config OUT/config.json \
        --output_dir OUT/NAME --data_dir ... --val_data_dir ...

--variant and --seed are this file's; the rest are the arguments of GenRecon's train.py. The baseline is the
variant `genrecon`: GenRecon's aggregator, which ignores the cues, finetuned on the same data.

Opt-in, in the config: dataset.args.cue_noise = {"view_scale": s, "field": s, "grid": [gy, gx]} perturbs the cue
depth (not the latents or the cameras) by a per-view scale and a smooth multiplicative field, to mimic the depth
errors of real images.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
GENRECON = HERE.parents[1] / "papers/GenRecon"
sys.path[:0] = [str(GENRECON), str(HERE)]

for flag, env in (("--variant", "MLP_BENCH_VARIANT"), ("--seed", "MLP_BENCH_SEED")):
    if flag in sys.argv:  # an env var, so the processes mp.spawn starts see it too
        i = sys.argv.index(flag)
        os.environ[env] = sys.argv[i + 1]
        del sys.argv[i : i + 2]
VARIANT = os.environ.get("MLP_BENCH_VARIANT", "genrecon")
SEED = int(os.environ.get("MLP_BENCH_SEED", "0"))

import aggregator as agg_lib  # noqa: E402
import variants  # noqa: E402, F401  (registers the variants)
from bench import BenchTrainerMixin, PairedItemsMixin, claim_run  # noqa: E402
from genrecon import datasets, trainers  # noqa: E402
from genrecon.datasets.sparse_structure_latent import RoomCameraConditionedSparseStructureLatent  # noqa: E402
from genrecon.pipelines.setup_utils import load_state_dict_flexible  # noqa: E402
from genrecon.trainers.flow_matching.flow_matching import CameraConditionedFlowMatchingCFGTrainer  # noqa: E402
from genrecon.utils import model_wrapper_utils  # noqa: E402

if VARIANT not in agg_lib.VARIANTS:
    raise SystemExit(f"unknown variant {VARIANT!r}; variants.py has: {', '.join(agg_lib.VARIANTS)}")

FLIP = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))  # Blender -> OpenCV camera axes


def chunk_w2c(M: torch.Tensor, c2w_room: torch.Tensor) -> torch.Tensor:
    """Blender camera-to-room [..., 4, 4] -> OpenCV chunk-to-camera, as RoomCameraConditionedMixin."""
    return FLIP @ torch.inverse(M @ c2w_room)


class DepthRoomSparseStructureLatent(PairedItemsMixin, RoomCameraConditionedSparseStructureLatent):
    """GenRecon's dataset, plus a depth map for each sampled view from root["depth_cues"]/<scene>/:

        cameras.npz   K [F, 3, 3] normalised to the full render, c2w [F, 4, 4] camera-to-room (Blender
                      convention, as transforms.json), F = the frames of transforms.json in its order
        <i:04d>.npz   depth [H, W] z-depth in room units, conf [H, W], for frame i; covering the full render

    The cue maps are padded to max_num_cond_views so they collate batch-first. With cue_cameras, GenRecon's
    true cameras are replaced by the cue's (crop applied), so training sees the cameras of the test-time setting.
    As the validation set its items repeat (PairedItemsMixin).
    """

    def __init__(self, roots, *, cue_cameras: bool = True, cue_noise: dict | None = None, **kwargs):
        self.cue_cameras = cue_cameras
        self.cue_noise = cue_noise
        super().__init__(roots, **kwargs)

    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        ids = metadata["sha256"] if "sha256" in metadata.columns else metadata.index.to_series()
        scenes = ids.astype(str).str.extract(r"^(.*)_\d+$")[0]
        cue_root = self._current_root["depth_cues"]
        has = scenes.map(lambda s: os.path.isfile(os.path.join(cue_root, s, "cameras.npz"))).fillna(False)
        metadata = metadata[has.to_numpy()]
        stats["With depth cues"] = len(metadata)
        return metadata, stats

    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)
        scene, chunk = instance.rsplit("_", 1)
        with open(os.path.join(root["base"], "renders_room", scene, "transforms.json")) as f:
            frames = json.load(f)["frames"]
        with open(os.path.join(root["base"], "crops", f"{scene}.json")) as f:
            M = next(c for c in json.load(f)["chunks"] if int(c["index"]) == int(chunk))["M_original_to_chunk"]
        M = torch.tensor(M, dtype=torch.float32)

        # The parent picks the views at random and returns only their cameras: find the frames from those.
        true_w2c = chunk_w2c(M, torch.tensor([fr["transform_matrix"] for fr in frames], dtype=torch.float32))
        err = (pack["extrinsics"][:, None] - true_w2c[None]).flatten(2).abs().amax(-1)  # [n, F]
        err, idx = err.min(1)
        assert float(err.max()) < 1e-4, f"{instance}: sampled views not found in transforms.json"
        idx = idx.numpy()

        cue_dir = os.path.join(root["depth_cues"], scene)
        cams = np.load(os.path.join(cue_dir, "cameras.npz"))
        K = torch.from_numpy(cams["K"][idx]).float()
        w2c = chunk_w2c(M, torch.from_numpy(cams["c2w"][idx]).float())
        maps = [np.load(os.path.join(cue_dir, f"{i:04d}.npz")) for i in idx]
        scale = float(torch.linalg.det(M[:3, :3]).abs() ** (1 / 3))  # room units -> chunk units
        depth = torch.stack([torch.from_numpy(m["depth"].astype(np.float32)) for m in maps]) * scale
        conf = torch.stack([torch.from_numpy(m["conf"].astype(np.float32)) for m in maps])
        if self.cue_noise:
            depth = perturb_depth(depth, **self.cue_noise)

        if self.cue_cameras:
            # The parent's crop maps the full-image K to the crop's K per axis as f -> s f, c -> s c + o.
            fov = torch.tensor([frames[i]["camera_angle_x"] for i in idx], dtype=torch.float32)
            f_true = 0.5 / torch.tan(fov / 2)
            Kc = pack["intrinsics"]
            s = Kc[:, [0, 1], [0, 1]] / f_true[:, None]  # [n, 2]
            o = Kc[:, [0, 1], [2, 2]] - s * 0.5
            K_crop = Kc.clone()
            K_crop[:, [0, 1], [0, 1]] = s * K[:, [0, 1], [0, 1]]
            K_crop[:, [0, 1], [2, 2]] = s * K[:, [0, 1], [2, 2]] + o
            pack["extrinsics"], pack["intrinsics"] = w2c, K_crop

        n, P = len(idx), self.max_num_cond_views
        pad = lambda x: torch.cat([x, x.new_zeros(P - n, *x.shape[1:])])  # noqa: E731
        pack["cue_depth"] = pad(depth.half())  # padded views: depth 0 = unknown
        pack["cue_conf"] = pad(conf.half())
        pack["cue_K"] = pad(K)
        pack["cue_extrinsics"] = pad(w2c)
        return pack


def perturb_depth(depth: torch.Tensor, view_scale: float = 0.0, field: float = 0.0, grid=(4, 6)) -> torch.Tensor:
    """depth [n, H, W] * exp(per-view N(0, view_scale) + a bilinear field from a grid of N(0, field)).
    torch's RNG, which the DataLoader seeds differently in every worker."""
    n, H, W = depth.shape
    log = view_scale * torch.randn(n, 1, 1)
    if field > 0:
        g = field * torch.randn(n, 1, *grid)
        log = log + F.interpolate(g, size=(H, W), mode="bilinear", align_corners=True)[:, 0]
    return depth * torch.exp(log)  # unknown depth (<= 0, non-finite) stays unknown


class MlpBenchTrainer(BenchTrainerMixin, CameraConditionedFlowMatchingCFGTrainer):
    """GenRecon's trainer with bench.py's additions. Gives the aggregator its depth cues from the batch before
    GenRecon's own conditioning runs."""

    def __init__(self, models, *args, conf_mean: float = 0.0, conf_std: float = 1.0, **kwargs):
        self.conf_mean, self.conf_std = conf_mean, conf_std
        model = models["denoiser"]
        info = {"variant": VARIANT, "seed": SEED, "code_hash": agg_lib.source_hash(VARIANT),
                "start_difference": model.aggregator.start_difference, "trained_weights": model.trained_weights,
                "conf_mean": conf_mean, "conf_std": conf_std, "code": agg_lib.source(VARIANT)}
        claim_run(kwargs["output_dir"], info)
        shown = ("max_steps", "batch_size_per_gpu", "optimizer", "lr_scheduler", "ema_rate", "i_val")
        config = {**info, **{k: kwargs[k] for k in shown if k in kwargs},
                  "train_chunks": len(args[0]) if args else None, "val_seed": kwargs.get("val_seed", 0),
                  "val_repeats": kwargs.get("val_repeats", 4)}
        config = json.loads(json.dumps(config))  # plain dicts and lists for W&B
        super().__init__(models, *args, seed=SEED, run_config=config, code_dir=HERE, **kwargs)

    def _set_cues(self, x_0, cue_depth, cue_conf, cue_K, cue_extrinsics):
        model = self.models["denoiser"]
        # GenRecon pads the views of each sample to the batch's largest view count, real views first
        # (_build_padded_feats), or stacks them when all samples have the same count. The cue maps are padded
        # to max_num_cond_views in the same order, with K = 0 for padding: cut them to the batch's view count.
        n = int((cue_K[:, :, 2, 2] != 0).sum(1).max())
        sl = lambda x: x[:, :n].to(x_0.device)  # noqa: E731
        points = model.projection.coords_homg[:, :3].to(x_0.device)
        voxel = 1.0 / round(points.shape[0] ** (1 / 3))  # the grid spans the unit chunk cube
        model.aggregator.cues = agg_lib.compute_cues(points, sl(cue_extrinsics), sl(cue_K), sl(cue_depth),
                                                     sl(cue_conf), voxel, self.conf_mean, self.conf_std)

    def training_losses(self, x_0, cond_batch_idx=None, cue_depth=None, cue_conf=None, cue_K=None,
                        cue_extrinsics=None, **kwargs):
        self._set_cues(x_0, cue_depth, cue_conf, cue_K, cue_extrinsics)
        return super().training_losses(x_0, cond_batch_idx=cond_batch_idx, **kwargs)

    def validation_losses(self, x_0, cond_batch_idx=None, cue_depth=None, cue_conf=None, cue_K=None,
                          cue_extrinsics=None, **kwargs):
        self._set_cues(x_0, cue_depth, cue_conf, cue_K, cue_extrinsics)
        return super().validation_losses(x_0, cond_batch_idx=cond_batch_idx, **kwargs)

    def get_inference_cond(self, denoiser, cond, extrinsics, intrinsics, cond_batch_idx=None, x_0=None,
                           cue_depth=None, cue_conf=None, cue_K=None, cue_extrinsics=None, **kwargs):
        self._set_cues(x_0, cue_depth, cue_conf, cue_K, cue_extrinsics)
        return super().get_inference_cond(denoiser, cond, extrinsics, intrinsics, cond_batch_idx=cond_batch_idx,
                                          x_0=x_0, **kwargs)


_build_single_model = model_wrapper_utils.build_single_model


def build_single_model(model_cfg):
    """GenRecon's builder, then: load the released checkpoint, put the variant in place of the aggregator and
    freeze everything outside it."""
    model = _build_single_model(model_cfg)
    cfg = model_cfg.get("bench")
    if cfg is None:
        return model
    report = load_state_dict_flexible(model, cfg["init_ckpt"])
    missing = report["missing_key_count"]
    assert missing == 0, f"{missing} parameters not in {cfg['init_ckpt']}: {report['missing_key_head']}"
    model.aggregator = agg_lib.build(VARIANT, model.aggregator)
    model.trained_weights = agg_lib.set_trainable(model)
    print(f"Loaded {cfg['init_ckpt']}; variant {VARIANT} trains {model.trained_weights:,} weights"
          + (f" (only {model.aggregator.train_only})" if model.aggregator.train_only else ""))
    return model


model_wrapper_utils.build_single_model = build_single_model
datasets.DepthRoomSparseStructureLatent = DepthRoomSparseStructureLatent
trainers.MlpBenchTrainer = MlpBenchTrainer

# GenRecon's train.py, unchanged. Its `if __name__ == "__main__"` block runs only in the launching process.
exec(compile((GENRECON / "train.py").read_text(), str(GENRECON / "train.py"), "exec"))
