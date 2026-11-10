"""Finetunes GenRecon's sparse-structure aggregator with per-view depth cues (variant A).

GenRecon is not modified. This file
  - registers CueRoomSparseStructureLatent and CueFlowMatchingCFGTrainer with GenRecon's registries,
  - wraps build_single_model to load the released GenRecon checkpoint, add the cue layers (cues.py) and
    freeze everything outside cues.train (default: the aggregator),
and then runs papers/GenRecon/train.py unchanged, in this module's namespace (so its mp.spawn finds `main`).

From papers/GenRecon, in GenRecon's env:
    python ../../experiments/genrecon_depth_cue_finetuning/train.py \
        --config ../../experiments/genrecon_depth_cue_finetuning/config_ss.json --output_dir OUT \
        --data_dir '{"sage": {"base": B, "ss_latent": L, "depth_cues": C}}' [--val_data_dir ...] [--control]

--control trains the same model with all cues set to 0: the cue layers then get no gradient, so this is the
"finetuned without cues" control. Everything else is train.py's arguments.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
GENRECON = HERE.parents[1] / "papers/GenRecon"
sys.path[:0] = [str(GENRECON), str(HERE)]

if "--control" in sys.argv:  # an env var, so the processes mp.spawn starts see it too
    sys.argv.remove("--control")
    os.environ["CUE_CONTROL"] = "1"

import cues as cue_lib  # noqa: E402
from genrecon import datasets, trainers  # noqa: E402
from genrecon.datasets.sparse_structure_latent import RoomCameraConditionedSparseStructureLatent  # noqa: E402
from genrecon.pipelines.setup_utils import load_state_dict_flexible  # noqa: E402
from genrecon.trainers.flow_matching.flow_matching import CameraConditionedFlowMatchingCFGTrainer  # noqa: E402
from genrecon.utils import model_wrapper_utils  # noqa: E402

FLIP = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))  # Blender -> OpenCV camera axes


def chunk_w2c(M: torch.Tensor, c2w_room: torch.Tensor) -> torch.Tensor:
    """Blender camera-to-room [..., 4, 4] -> OpenCV chunk-to-camera, as RoomCameraConditionedMixin."""
    return FLIP @ torch.inverse(M @ c2w_room)


class CueRoomSparseStructureLatent(RoomCameraConditionedSparseStructureLatent):
    """GenRecon's dataset, plus a depth map for each sampled view from root["depth_cues"]/<scene>/:

        cameras.npz   K [F, 3, 3] normalised to the full render, c2w [F, 4, 4] camera-to-room (Blender
                      convention, as transforms.json), F = the frames of transforms.json in its order
        <i:04d>.npz   depth [H, W] z-depth in room units, conf [H, W], for frame i; covering the full render

    The cue maps are padded to max_num_cond_views so they collate batch-first. With cue_cameras, GenRecon's
    true cameras are replaced by the cue's (crop applied), so training sees the cameras of the test-time setting.
    """

    def __init__(self, roots, *, cue_cameras: bool = True, **kwargs):
        self.cue_cameras = cue_cameras
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


class CueFlowMatchingCFGTrainer(CameraConditionedFlowMatchingCFGTrainer):
    """Sets the aggregator's cues from the batch before GenRecon's own conditioning runs."""

    def __init__(self, *args, conf_mean: float = 0.0, conf_std: float = 1.0, **kwargs):
        self.conf_mean, self.conf_std = conf_mean, conf_std
        self.use_cues = os.environ.get("CUE_CONTROL") != "1"
        super().__init__(*args, **kwargs)
        if self.is_master:
            with open(os.path.join(self.output_dir, "cues.json"), "w") as f:
                json.dump({"use_cues": self.use_cues, "conf_mean": conf_mean, "conf_std": conf_std}, f, indent=4)
            print(f"Depth cues: {'on' if self.use_cues else 'off (control)'}")

    def _set_cues(self, x_0, cond_batch_idx, cue_depth, cue_conf, cue_K, cue_extrinsics):
        model = self.models["denoiser"]
        # GenRecon pads the views of each sample to the batch's largest view count, real views first
        # (_build_padded_feats), or stacks them when all samples have the same count. The cue maps are padded
        # to max_num_cond_views in the same order, with K = 0 for padding: cut them to the batch's view count.
        n = int((cue_K[:, :, 2, 2] != 0).sum(1).max())
        sl = lambda x: x[:, :n].to(x_0.device)  # noqa: E731
        points = model.projection.coords_homg[:, :3].to(x_0.device)
        voxel = 1.0 / round(points.shape[0] ** (1 / 3))  # the grid spans the unit chunk cube
        cues = cue_lib.compute_cues(points, sl(cue_extrinsics), sl(cue_K), sl(cue_depth), sl(cue_conf), voxel,
                                    self.conf_mean, self.conf_std)
        model.aggregator.cues = cues if self.use_cues else torch.zeros_like(cues)

    def training_losses(self, x_0, cond_batch_idx=None, cue_depth=None, cue_conf=None, cue_K=None,
                        cue_extrinsics=None, **kwargs):
        self._set_cues(x_0, cond_batch_idx, cue_depth, cue_conf, cue_K, cue_extrinsics)
        return super().training_losses(x_0, cond_batch_idx=cond_batch_idx, **kwargs)

    def validation_losses(self, x_0, cond_batch_idx=None, cue_depth=None, cue_conf=None, cue_K=None,
                          cue_extrinsics=None, **kwargs):
        self._set_cues(x_0, cond_batch_idx, cue_depth, cue_conf, cue_K, cue_extrinsics)
        return super().validation_losses(x_0, cond_batch_idx=cond_batch_idx, **kwargs)

    def get_inference_cond(self, denoiser, cond, extrinsics, intrinsics, cond_batch_idx=None, x_0=None,
                           cue_depth=None, cue_conf=None, cue_K=None, cue_extrinsics=None, **kwargs):
        self._set_cues(x_0, cond_batch_idx, cue_depth, cue_conf, cue_K, cue_extrinsics)
        return super().get_inference_cond(denoiser, cond, extrinsics, intrinsics, cond_batch_idx=cond_batch_idx,
                                          x_0=x_0, **kwargs)


_build_single_model = model_wrapper_utils.build_single_model


def build_single_model(model_cfg):
    """GenRecon's builder, then: load the released checkpoint, add the cue layers, freeze all but cues.train."""
    model = _build_single_model(model_cfg)
    cfg = model_cfg.get("cues")
    if cfg is None:
        return model
    report = load_state_dict_flexible(model, cfg["init_ckpt"])
    cue_lib.add_cue_layers(model.aggregator)
    missing = report["missing_key_count"]
    assert missing == 0, f"{missing} parameters not in {cfg['init_ckpt']}: {report['missing_key_head']}"
    train = tuple(cfg.get("train", ["aggregator."]))
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith(train))
    print(f"Loaded {cfg['init_ckpt']}; training {sum(p.numel() for p in model.parameters() if p.requires_grad):,} "
          f"parameters under {train}")
    return model


model_wrapper_utils.build_single_model = build_single_model
datasets.CueRoomSparseStructureLatent = CueRoomSparseStructureLatent
trainers.CueFlowMatchingCFGTrainer = CueFlowMatchingCFGTrainer

# GenRecon's train.py, unchanged. Its `if __name__ == "__main__"` block runs only in the launching process.
exec(compile((GENRECON / "train.py").read_text(), str(GENRECON / "train.py"), "exec"))
