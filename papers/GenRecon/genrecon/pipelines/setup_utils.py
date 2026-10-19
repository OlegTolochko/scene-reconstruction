from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from easydict import EasyDict as edict

from .. import models as trellis_models
from ..modules import image_feature_extractor
from ..modules.utils import convert_module_to
from ..utils.model_wrapper_utils import build_single_model
from . import samplers

SUPPORTED_STAGE_MODEL_KEYS = {
    "sparse_structure_flow_model",
    "shape_slat_flow_model_512",
    "shape_slat_flow_model_1024",
    "tex_slat_flow_model_512",
    "tex_slat_flow_model_1024",
}

STAGE_FAMILY_BY_MODEL_KEY = {
    "sparse_structure_flow_model": "sparse_structure_flow_model",
    "shape_slat_flow_model_512": "shape_slat_flow_model",
    "shape_slat_flow_model_1024": "shape_slat_flow_model",
    "tex_slat_flow_model_512": "tex_slat_flow_model",
    "tex_slat_flow_model_1024": "tex_slat_flow_model",
}

STAGE_RESOLUTION_BY_MODEL_KEY = {
    "shape_slat_flow_model_512": 512,
    "shape_slat_flow_model_1024": 1024,
    "tex_slat_flow_model_512": 512,
    "tex_slat_flow_model_1024": 1024,
}


@dataclass
class LoadedStage:
    model: nn.Module
    load_report: dict
    output_resolution: int | None = None
    num_cond_views: int = 1
    image_size: int | None = None


PBR_ATTR_LAYOUT = {
    "base_color": slice(0, 3),
    "metallic": slice(3, 4),
    "roughness": slice(4, 5),
    "alpha": slice(5, 6),
}


def load_pipeline_args(config_file: str) -> dict:
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Pipeline config not found: {config_file}")

    with open(config_file, "r") as f:
        config = json.load(f)

    if "args" not in config:
        raise ValueError(f"Pipeline config '{config_file}' does not contain an 'args' section.")
    return config["args"]


def resolve_model_ref(model_ref: str, pretrained_source: str) -> str:
    if model_ref.startswith("ckpts/"):
        return f"{pretrained_source}/{model_ref}"
    return model_ref


def load_pretrained_model_from_args(args: dict, model_key: str, pretrained_source: str) -> nn.Module:
    model_ref = args.get("models", {}).get(model_key)
    if model_ref is None:
        raise ValueError(f"Pipeline config does not define model '{model_key}'.")
    return trellis_models.from_pretrained(resolve_model_ref(model_ref, pretrained_source)).eval()


def validate_requested_stage_models(stage_models: dict[str, str | None]) -> None:
    if not stage_models:
        raise ValueError(
            "stage_models must contain at least one supported key: " f"{sorted(SUPPORTED_STAGE_MODEL_KEYS)}"
        )

    unknown_keys = set(stage_models) - SUPPORTED_STAGE_MODEL_KEYS
    if unknown_keys:
        raise ValueError(
            f"Unknown stage model keys: {sorted(unknown_keys)}. "
            f"Supported keys: {sorted(SUPPORTED_STAGE_MODEL_KEYS)}"
        )

    invalid_value_keys = [
        key for key, value in stage_models.items() if value is not None and not isinstance(value, str)
    ]
    if invalid_value_keys:
        raise TypeError(
            "Each stage model value must be either a checkpoint path string or None. "
            f"Invalid keys: {sorted(invalid_value_keys)}"
        )


def get_requested_stage_families(requested_model_keys: set[str]) -> set[str]:
    return {STAGE_FAMILY_BY_MODEL_KEY[key] for key in requested_model_keys}


def init_runtime_assets(
    pipeline,
    args: dict,
    requested_model_keys: set[str],
    pretrained_source: str,
) -> None:
    pipeline.models = getattr(pipeline, "models", {})
    requested_stage_families = get_requested_stage_families(requested_model_keys)

    pipeline.image_cond_model = getattr(image_feature_extractor, args["image_cond_model"]["name"])(
        **args["image_cond_model"]["args"]
    )
    pipeline.rembg_model = None
    pipeline.low_vram = args.get("low_vram", True)
    pipeline.default_pipeline_type = None
    pipeline.pbr_attr_layout = PBR_ATTR_LAYOUT.copy()
    pipeline._device = "cpu"
    pipeline._num_cond_views_by_model_key = {}

    pipeline.sparse_structure_sampler = None
    pipeline.shape_slat_sampler = None
    pipeline.tex_slat_sampler = None
    pipeline.sparse_structure_sampler_params = None
    pipeline.shape_slat_sampler_params = None
    pipeline.tex_slat_sampler_params = None
    pipeline.shape_slat_normalization = None
    pipeline.tex_slat_normalization = None

    if "sparse_structure_flow_model" in requested_stage_families:
        pipeline.models["sparse_structure_decoder"] = load_pretrained_model_from_args(
            args,
            "sparse_structure_decoder",
            pretrained_source,
        )
        pipeline.sparse_structure_sampler = getattr(samplers, args["sparse_structure_sampler"]["name"])(
            **args["sparse_structure_sampler"]["args"]
        )
        pipeline.sparse_structure_sampler_params = args["sparse_structure_sampler"]["params"]

    if "shape_slat_flow_model" in requested_stage_families:
        pipeline.models["shape_slat_decoder"] = load_pretrained_model_from_args(
            args,
            "shape_slat_decoder",
            pretrained_source,
        )
        pipeline.shape_slat_sampler = getattr(samplers, args["shape_slat_sampler"]["name"])(
            **args["shape_slat_sampler"]["args"]
        )
        pipeline.shape_slat_sampler_params = args["shape_slat_sampler"]["params"]
        pipeline.shape_slat_normalization = args["shape_slat_normalization"]

    if "tex_slat_flow_model" in requested_stage_families:
        pipeline.models["tex_slat_decoder"] = load_pretrained_model_from_args(
            args,
            "tex_slat_decoder",
            pretrained_source,
        )
        pipeline.tex_slat_sampler = getattr(samplers, args["tex_slat_sampler"]["name"])(
            **args["tex_slat_sampler"]["args"]
        )
        pipeline.tex_slat_sampler_params = args["tex_slat_sampler"]["params"]
        pipeline.tex_slat_normalization = args["tex_slat_normalization"]


def validate_loaded_stage_resolution(model_key: str, output_resolution: int | None) -> None:
    expected_resolution = STAGE_RESOLUTION_BY_MODEL_KEY.get(model_key)
    if expected_resolution is None:
        return
    if output_resolution != expected_resolution:
        raise ValueError(
            f"Checkpoint for '{model_key}' resolved to output resolution {output_resolution}, "
            f"expected {expected_resolution}."
        )


def infer_default_pipeline_type(loaded_model_keys: set[str]) -> str | None:
    if "sparse_structure_flow_model" not in loaded_model_keys:
        return None
    if {
        "shape_slat_flow_model_512",
        "shape_slat_flow_model_1024",
        "tex_slat_flow_model_1024",
    }.issubset(loaded_model_keys):
        return "1024_cascade"
    if {"shape_slat_flow_model_1024", "tex_slat_flow_model_1024"}.issubset(loaded_model_keys):
        return "1024"
    if {"shape_slat_flow_model_512", "tex_slat_flow_model_512"}.issubset(loaded_model_keys):
        return "512"
    return None


def load_sparse_flow_stage(ckpt_path: str, train_config_path: str | None = None) -> LoadedStage:
    train_cfg = load_train_config(ckpt_path, train_config_path)
    flow_cfg = find_model_cfg(
        train_cfg,
        ckpt_path,
        allowed_model_names={"SparseStructureFlowModel"},
    )
    model, load_report = build_and_load_model(flow_cfg, ckpt_path)
    num_cond_views = extract_num_cond_views(train_cfg)
    image_size = extract_image_size(train_cfg)
    return LoadedStage(
        model=model,
        load_report=load_report,
        output_resolution=None,
        num_cond_views=num_cond_views,
        image_size=image_size,
    )


def load_slat_stage(
    ckpt_path: str,
    allowed_model_names: set[str],
    train_config_path: str | None = None,
) -> LoadedStage:
    train_cfg = load_train_config(ckpt_path, train_config_path)
    flow_cfg = find_model_cfg(
        train_cfg,
        ckpt_path,
        allowed_model_names=allowed_model_names,
    )
    model, load_report = build_and_load_model(flow_cfg, ckpt_path)
    output_resolution = extract_output_resolution(train_cfg, flow_cfg)
    num_cond_views = extract_num_cond_views(train_cfg)
    image_size = extract_image_size(train_cfg)
    return LoadedStage(
        model=model,
        load_report=load_report,
        output_resolution=output_resolution,
        num_cond_views=num_cond_views,
        image_size=image_size,
    )


def build_and_load_model(flow_cfg: dict, ckpt_path: str) -> tuple[nn.Module, dict]:
    model = build_single_model(edict(flow_cfg))
    load_report = load_state_dict_flexible(model, ckpt_path)
    coerce_model_to_bfloat16(model)
    align_model_dtype_modules(model)
    return model.eval(), load_report


def extract_num_cond_views(train_cfg: dict) -> int:
    dataset_args = train_cfg.get("dataset", {}).get("args", {})
    # max_num_cond_views supersedes the old num_cond_views key
    num_cond_views = dataset_args.get("max_num_cond_views", dataset_args.get("num_cond_views", 1))
    try:
        num_cond_views = int(num_cond_views)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid num_cond_views: {num_cond_views}") from e
    if num_cond_views <= 0:
        raise ValueError(f"num_cond_views must be > 0, got {num_cond_views}")
    return num_cond_views


def extract_image_size(train_cfg: dict) -> int | None:
    """Return the conditioning image resolution from the training config, or None if absent."""
    raw = train_cfg.get("dataset", {}).get("args", {}).get("image_size", None)
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid image_size in training config: {raw}") from e
    if value <= 0:
        raise ValueError(f"image_size must be > 0, got {value}")
    return value


def find_model_cfg(train_cfg: dict, ckpt_path: str, allowed_model_names: set[str]) -> dict:
    ckpt_name = os.path.basename(ckpt_path)
    ckpt_prefix = ckpt_name.split("_step")[0]

    if "models" in train_cfg and ckpt_prefix in train_cfg["models"]:
        cfg = train_cfg["models"][ckpt_prefix]
        if cfg.get("name") in allowed_model_names:
            return cfg

    for cfg in train_cfg.get("models", {}).values():
        if cfg.get("name") in allowed_model_names:
            return cfg

    raise ValueError(
        f"Could not find model config for checkpoint '{ckpt_path}'. Expected one of: {sorted(allowed_model_names)}"
    )


def extract_output_resolution(train_cfg: dict, flow_cfg: dict) -> int:
    dataset_resolution = train_cfg.get("dataset", {}).get("args", {}).get("resolution")
    if dataset_resolution is not None:
        try:
            dataset_resolution = int(dataset_resolution)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid dataset resolution: {dataset_resolution}") from e
        if dataset_resolution in {512, 1024}:
            return dataset_resolution

    latent_resolution = flow_cfg.get("args", {}).get("resolution")
    if latent_resolution is None:
        raise ValueError("Could not infer output resolution from train config or flow config.")
    try:
        latent_resolution = int(latent_resolution)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid latent resolution in flow config: {latent_resolution}") from e

    output_resolution = latent_resolution * 16
    if output_resolution not in {512, 1024}:
        raise ValueError(f"Unsupported inferred output resolution: {output_resolution}")
    return output_resolution


def load_train_config(ckpt_path: str, train_config_path: str | None = None) -> dict:
    if train_config_path is not None:
        config_path = train_config_path
    else:
        ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
        run_dir = os.path.dirname(ckpt_dir)
        config_path = os.path.join(run_dir, "config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Training config not found: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)


def load_state_dict_flexible(model: nn.Module, ckpt_path: str) -> dict:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    raw_ckpt = load_checkpoint(ckpt_path)
    state, report = extract_best_state_dict(raw_ckpt, model)
    model_state = model.state_dict()
    load_result = model.load_state_dict(state, strict=False)

    model_keys = list(model_state.keys())
    loaded_keys = list(state.keys())
    report.update(
        {
            "checkpoint_path": ckpt_path,
            "missing_key_count": len(load_result.missing_keys),
            "missing_key_head": list(load_result.missing_keys[:20]),
            "unexpected_key_count": len(load_result.unexpected_keys),
            "unexpected_key_head": list(load_result.unexpected_keys[:20]),
            "model_key_prefix_counts": count_prefixed_keys(
                model_keys,
                ("projection.", "aggregator.", "proj_linears.", "blocks.", "input_layer.", "out_layer."),
            ),
            "loaded_key_prefix_counts": count_prefixed_keys(
                loaded_keys,
                ("projection.", "aggregator.", "proj_linears.", "blocks.", "input_layer.", "out_layer."),
            ),
        }
    )
    return report


def coerce_model_to_bfloat16(model: nn.Module) -> None:
    convert_to = getattr(model, "convert_to", None)
    if callable(convert_to):
        convert_to(torch.bfloat16)
    elif hasattr(model, "dtype"):
        model.dtype = torch.bfloat16
    else:
        raise ValueError(f"Model {type(model).__name__} does not expose dtype/convert_to for bf16 coercion.")


def align_model_dtype_modules(model: nn.Module) -> None:
    model.blocks.apply(lambda m: convert_module_to(m, dtype=model.dtype))
    for attr in ("projection", "aggregator", "proj_linears", "plucker_modulations", "linear_plucker"):
        module = getattr(model, attr, None)
        if module is not None:
            module.apply(lambda m: convert_module_to(m, dtype=model.dtype))


def load_checkpoint(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def tensor_state_dict(obj: Any) -> dict[str, torch.Tensor] | None:
    if not isinstance(obj, dict):
        return None
    state = {k: v for k, v in obj.items() if isinstance(k, str) and isinstance(v, torch.Tensor)}
    return state if state else None


def collect_state_dict_candidates(raw_ckpt: Any) -> list[tuple[str, dict[str, torch.Tensor]]]:
    candidates: list[tuple[str, dict[str, torch.Tensor]]] = []
    seen: set[int] = set()

    def add(label: str, obj: Any) -> None:
        if not isinstance(obj, dict):
            return
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)
        tensor_dict = tensor_state_dict(obj)
        if tensor_dict is not None:
            candidates.append((label, tensor_dict))

    add("checkpoint", raw_ckpt)
    if isinstance(raw_ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model", "module", "denoiser", "ema_state_dict", "weights"):
            if key in raw_ckpt:
                add(key, raw_ckpt[key])
        for key, value in raw_ckpt.items():
            if isinstance(value, dict):
                add(key, value)

    return candidates


def strip_prefix(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in state.items()}


def state_match_score(state: dict[str, torch.Tensor], model_state: dict[str, torch.Tensor]) -> tuple[int, int]:
    overlap = 0
    shape_match = 0
    for key, value in state.items():
        if key in model_state:
            overlap += 1
            if tuple(value.shape) == tuple(model_state[key].shape):
                shape_match += 1
    return shape_match, overlap


def extract_best_state_dict(raw_ckpt: Any, model: nn.Module) -> tuple[dict[str, torch.Tensor], dict]:
    model_state = model.state_dict()
    candidates = collect_state_dict_candidates(raw_ckpt)
    if not candidates:
        raise ValueError("No tensor state_dict-like mapping found in checkpoint.")

    prefixes = ("module.", "model.", "denoiser.", "models.denoiser.", "backbone.", "net.")
    best_state = None
    best_label = ""
    best_score = (-1, -1)

    for label, state in candidates:
        transformed = [(label, state)]
        for prefix in prefixes:
            if any(key.startswith(prefix) for key in state):
                transformed.append((f"{label} (strip:{prefix})", strip_prefix(state, prefix)))

        for candidate_label, candidate_state in transformed:
            score = state_match_score(candidate_state, model_state)
            if score > best_score:
                best_score = score
                best_state = candidate_state
                best_label = candidate_label

    if best_state is None or best_score[1] == 0:
        model_keys = list(model_state.keys())[:8]
        raise ValueError(f"Could not match checkpoint keys to model parameters. Example model keys: {model_keys}")

    filtered = {
        key: value
        for key, value in best_state.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    if not filtered:
        raise ValueError(f"Matched checkpoint candidate '{best_label}' but found no shape-compatible parameters.")

    shape_mismatch = sorted(
        key
        for key, value in best_state.items()
        if key in model_state and tuple(value.shape) != tuple(model_state[key].shape)
    )
    report = {
        "matched_candidate": best_label,
        "candidate_shape_match_count": best_score[0],
        "candidate_overlap_count": best_score[1],
        "candidate_key_count": len(best_state),
        "loaded_key_count": len(filtered),
        "model_key_count": len(model_state),
        "shape_mismatch_count": len(shape_mismatch),
        "shape_mismatch_head": shape_mismatch[:20],
    }
    return filtered, report


def count_prefixed_keys(keys: list[str], prefixes: tuple[str, ...]) -> dict[str, int]:
    return {prefix.rstrip("."): sum(1 for key in keys if key.startswith(prefix)) for prefix in prefixes}
