from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict
from PIL import Image

from ..modules.sparse import SparseTensor
from ..representations import Mesh, MeshWithVoxel
from .base import Pipeline
from .setup_utils import (
    SUPPORTED_STAGE_MODEL_KEYS,
    infer_default_pipeline_type,
    init_runtime_assets,
    load_pipeline_args,
    load_pretrained_model_from_args,
    load_slat_stage,
    load_sparse_flow_stage,
    validate_loaded_stage_resolution,
    validate_requested_stage_models,
)


class ImagesTo3DPipeline(Pipeline):
    """Pipeline for TRELLIS2 image-to-3D inference with partial stage loading."""

    DEFAULT_PRETRAINED_SOURCE = "microsoft/TRELLIS.2-4B"
    DEFAULT_PIPELINE_CONFIG_FILE = "configs/pipelines/original.json"
    SUPPORTED_STAGE_MODEL_KEYS = SUPPORTED_STAGE_MODEL_KEYS
    SUPPORTED_OCCUPANCY_RESOLUTIONS = frozenset({32, 64})

    def __init__(
        self,
        models: Optional[dict[str, nn.Module]] = None,
        sparse_structure_sampler: Any = None,
        shape_slat_sampler: Any = None,
        tex_slat_sampler: Any = None,
        sparse_structure_sampler_params: Optional[dict] = None,
        shape_slat_sampler_params: Optional[dict] = None,
        tex_slat_sampler_params: Optional[dict] = None,
        shape_slat_normalization: Optional[dict] = None,
        tex_slat_normalization: Optional[dict] = None,
        image_cond_model: Optional[Callable] = None,
        low_vram: bool = True,
        default_pipeline_type: Optional[str] = None,
    ):
        super().__init__(models or {})
        self.sparse_structure_sampler = sparse_structure_sampler
        self.shape_slat_sampler = shape_slat_sampler
        self.tex_slat_sampler = tex_slat_sampler
        self.sparse_structure_sampler_params = sparse_structure_sampler_params
        self.shape_slat_sampler_params = shape_slat_sampler_params
        self.tex_slat_sampler_params = tex_slat_sampler_params
        self.shape_slat_normalization = shape_slat_normalization
        self.tex_slat_normalization = tex_slat_normalization
        self.image_cond_model = image_cond_model
        self.low_vram = low_vram
        self.default_pipeline_type = default_pipeline_type
        self.pbr_attr_layout = {
            "base_color": slice(0, 3),
            "metallic": slice(3, 4),
            "roughness": slice(4, 5),
            "alpha": slice(5, 6),
        }
        self._device = "cpu"
        self._num_cond_views_by_model_key: dict[str, int] = {}
        self._image_size_by_model_key: dict[str, int] = {}
        self._checkpoint_load_report: Optional[dict[str, dict]] = None
        self._pretrained_args: Optional[dict] = None

    @classmethod
    def from_finetuned(
        cls,
        stage_models: dict[str, str | None],
        pipeline_config_file: str | None = None,
    ) -> "ImagesTo3DPipeline":
        """Build a partially loaded pipeline from explicit pretrained or finetuned stage selections."""
        validate_requested_stage_models(stage_models)

        args = load_pipeline_args(pipeline_config_file or cls.DEFAULT_PIPELINE_CONFIG_FILE)
        pipeline = cls(models={})
        pipeline._pretrained_args = args
        init_runtime_assets(
            pipeline,
            args,
            requested_model_keys=set(stage_models.keys()),
            pretrained_source=cls.DEFAULT_PRETRAINED_SOURCE,
        )

        load_report: dict[str, dict] = {}

        for model_key, ckpt_path in stage_models.items():
            if ckpt_path is None:
                pipeline.models[model_key] = load_pretrained_model_from_args(
                    args,
                    model_key,
                    cls.DEFAULT_PRETRAINED_SOURCE,
                )
                pipeline._num_cond_views_by_model_key[model_key] = 1
                continue

            if model_key == "sparse_structure_flow_model":
                loaded = load_sparse_flow_stage(ckpt_path=ckpt_path)
            elif model_key.startswith("shape_slat_flow_model_"):
                loaded = load_slat_stage(
                    ckpt_path=ckpt_path,
                    allowed_model_names={"SLatFlowModel", "ElasticSLatFlowModel"},
                )
                validate_loaded_stage_resolution(model_key, loaded.output_resolution)
            elif model_key.startswith("tex_slat_flow_model_"):
                loaded = load_slat_stage(
                    ckpt_path=ckpt_path,
                    allowed_model_names={"SLatFlowModel", "ElasticSLatFlowModel"},
                )
                validate_loaded_stage_resolution(model_key, loaded.output_resolution)
            else:
                raise ValueError(f"Unsupported stage model key: {model_key}")

            pipeline.models[model_key] = loaded.model
            pipeline._num_cond_views_by_model_key[model_key] = loaded.num_cond_views
            if loaded.image_size is not None:
                pipeline._image_size_by_model_key[model_key] = loaded.image_size
            load_report[model_key] = loaded.load_report

        pipeline.default_pipeline_type = infer_default_pipeline_type(set(stage_models.keys()))
        pipeline._checkpoint_load_report = load_report or None
        return pipeline

    def _require_components(
        self,
        required_model_keys: tuple[str, ...] = (),
        required_attrs: tuple[str, ...] = (),
    ) -> None:
        """Raise if the pipeline is missing any models or runtime attributes required by a later step."""
        missing_model_keys = [key for key in required_model_keys if key not in self.models]
        missing_attrs = [name for name in required_attrs if getattr(self, name, None) is None]
        if not missing_model_keys and not missing_attrs:
            return

        parts = []
        if missing_model_keys:
            parts.append(f"missing models: {', '.join(missing_model_keys)}")
        if missing_attrs:
            parts.append(f"missing attributes: {', '.join(missing_attrs)}")
        raise RuntimeError(
            "Pipeline is missing required components ("
            + "; ".join(parts)
            + "). Load the needed stages with ImagesTo3DPipeline.from_finetuned(...)."
        )

    def _require_run_components(self, pipeline_type: str) -> None:
        """Validate that all models and runtime assets needed for the requested pipeline path are loaded."""
        required_model_keys = [
            "sparse_structure_flow_model",
            "sparse_structure_decoder",
            "shape_slat_decoder",
            "tex_slat_decoder",
        ]
        required_attrs = [
            "image_cond_model",
            "sparse_structure_sampler",
            "sparse_structure_sampler_params",
            "shape_slat_sampler",
            "shape_slat_sampler_params",
            "shape_slat_normalization",
            "tex_slat_sampler",
            "tex_slat_sampler_params",
            "tex_slat_normalization",
        ]

        if pipeline_type == "512":
            required_model_keys.extend(["shape_slat_flow_model_512", "tex_slat_flow_model_512"])
        elif pipeline_type == "1024":
            required_model_keys.extend(["shape_slat_flow_model_1024", "tex_slat_flow_model_1024"])
        elif pipeline_type in {"1024_cascade", "1536_cascade"}:
            required_model_keys.extend(
                ["shape_slat_flow_model_512", "shape_slat_flow_model_1024", "tex_slat_flow_model_1024"]
            )
        else:
            raise ValueError(f"Invalid pipeline type: {pipeline_type}")

        self._require_components(tuple(required_model_keys), tuple(required_attrs))

    def to(self, device: torch.device) -> None:
        self._device = device
        if not self.low_vram:
            super().to(device)
            if self.image_cond_model is not None:
                self.image_cond_model.to(device)

    @staticmethod
    def _pil_to_tensor(image: Image.Image, resolution: int) -> torch.Tensor:
        if image.size != (resolution, resolution):
            image = image.resize((resolution, resolution), Image.Resampling.LANCZOS)

        if image.mode == "RGBA":
            alpha = image.getchannel(3)
        else:
            alpha = Image.new("L", image.size, 255)
            image = image.convert("RGB")

        image_rgb = image.convert("RGB")
        image_t = torch.tensor(np.array(image_rgb)).permute(2, 0, 1).float() / 255.0
        alpha_t = torch.tensor(np.array(alpha)).float() / 255.0
        return image_t * alpha_t.unsqueeze(0)

    def _build_conditioning_tensor(self, images: list[Image.Image], resolution: int) -> torch.Tensor:
        view_tensors = [self._pil_to_tensor(img, resolution=resolution) for img in images]
        hw = {tuple(v.shape[-2:]) for v in view_tensors}
        if len(hw) != 1:
            raise ValueError(f"All conditioning views must share the same size. Found: {sorted(hw)}")
        return torch.stack(view_tensors, dim=0).unsqueeze(0)

    def _encode_conditioning_tokens(self, cond_input: torch.Tensor, resolution: int) -> torch.Tensor:
        # cond_input: [B, N, C, H, W] → returns [B, N, T, D]
        self.image_cond_model.image_size = resolution
        if self.low_vram:
            self.image_cond_model.to(self.device)

        cond = self.image_cond_model(cond_input.to(self.device))

        if self.low_vram:
            self.image_cond_model.cpu()

        return cond

    def get_cond(
        self,
        images: Union[torch.Tensor, list[Image.Image]],
        resolution: int,
        model_key: str,
        include_neg_cond: bool = True,
        intrinsics: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None,
        num_samples: int = 1,
        coords: Optional[torch.Tensor] = None,
        num_cond_views: Optional[int] = None,
    ) -> dict:
        self._require_components(required_attrs=("image_cond_model",))
        flow_model = self.models.get(model_key)

        num_views = (
            num_cond_views if num_cond_views is not None else self._num_cond_views_by_model_key.get(model_key, 1)
        )
        if len(images) < num_views:
            raise ValueError(f"Model '{model_key}' expects at least {num_views} input images, got {len(images)}.")

        if (intrinsics is None) != (extrinsics is None):
            raise ValueError("Both intrinsics and extrinsics must be provided together.")
        if intrinsics is not None and extrinsics is not None:
            if intrinsics.ndim != 3 or tuple(intrinsics.shape[1:]) != (3, 3):
                raise ValueError(f"Expected intrinsics to have shape [N, 3, 3], got {tuple(intrinsics.shape)}")
            if extrinsics.ndim != 3 or tuple(extrinsics.shape[1:]) != (4, 4):
                raise ValueError(f"Expected extrinsics to have shape [N, 4, 4], got {tuple(extrinsics.shape)}")
            if intrinsics.shape[0] < num_views or extrinsics.shape[0] < num_views:
                raise ValueError(
                    f"Model '{model_key}' expects at least {num_views} camera views, got "
                    f"{intrinsics.shape[0]} intrinsics and {extrinsics.shape[0]} extrinsics."
                )
            stage_intrinsics = intrinsics[:num_views].unsqueeze(0).to(self.device)
            stage_extrinsics = extrinsics[:num_views].unsqueeze(0).to(self.device)
            if num_samples > 1:
                stage_intrinsics = stage_intrinsics.repeat(num_samples, 1, 1, 1)
                stage_extrinsics = stage_extrinsics.repeat(num_samples, 1, 1, 1)
        else:
            stage_intrinsics = None
            stage_extrinsics = None

        # Encode images → img_feats_all [B, N, T, D]
        stage_images = images[:num_views]
        if isinstance(stage_images, torch.Tensor):
            img_feats_all = stage_images.to(self.device)
            if img_feats_all.ndim == 4:  # [N, T, D] → [1, N, T, D]
                img_feats_all = img_feats_all.unsqueeze(0)
        else:
            cond_input = self._build_conditioning_tensor(stage_images, resolution=resolution)
            img_feats_all = self._encode_conditioning_tokens(cond_input, resolution=resolution)

        cond_2D = img_feats_all[:, 0]  # [B, T, D] — first-view tokens for cross-attention

        if flow_model is not None and hasattr(flow_model, "dtype"):
            img_feats_all = img_feats_all.to(dtype=flow_model.dtype)
            cond_2D = cond_2D.to(dtype=flow_model.dtype)
            if stage_intrinsics is not None:
                stage_intrinsics = stage_intrinsics.to(dtype=flow_model.dtype)
            if stage_extrinsics is not None:
                stage_extrinsics = stage_extrinsics.to(dtype=flow_model.dtype)

        if num_samples > 1:
            cond_2D = cond_2D.repeat(num_samples, *([1] * (cond_2D.ndim - 1)))

        # 3D conditioning: project image features onto sparse voxel locations
        cond_3D = None
        if flow_model is not None and stage_extrinsics is not None:
            x_0 = None
            if coords is not None:
                x_0 = SparseTensor(
                    feats=torch.zeros(coords.shape[0], 1, device=self.device),
                    coords=coords,
                )
            flow_model_on_device = False
            if self.low_vram:
                flow_model.to(self.device)
                flow_model_on_device = True
            cond_3D = flow_model.get_3D_cond(img_feats_all, stage_extrinsics, stage_intrinsics, x_0)
            if flow_model_on_device:
                flow_model.cpu()

        if cond_3D is None:
            neg_cond_3D = None
        elif isinstance(cond_3D, torch.Tensor):
            neg_cond_3D = torch.zeros_like(cond_3D)
        else:  # SparseTensor
            neg_cond_3D = cond_3D.replace(torch.zeros_like(cond_3D.feats))

        cond_dict = {"cond_2D": cond_2D, "cond_3D": cond_3D}
        neg_cond_dict = {"cond_2D": torch.zeros_like(cond_2D), "cond_3D": neg_cond_3D}

        ret = {"cond": cond_dict}
        if include_neg_cond:
            ret["neg_cond"] = neg_cond_dict
        return ret

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
        inpaint_constr: Optional[EasyDict] = None,
    ) -> torch.Tensor:
        self._require_components(
            required_model_keys=("sparse_structure_flow_model",),
            required_attrs=("sparse_structure_sampler", "sparse_structure_sampler_params"),
        )

        flow_model = self.models["sparse_structure_flow_model"]
        reso = flow_model.resolution
        in_channels = flow_model.in_channels
        noise_dtype = flow_model.dtype if hasattr(flow_model, "dtype") else torch.float32
        noise = torch.randn(num_samples, in_channels, reso, reso, reso, dtype=noise_dtype, device=self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        sample_kwargs = {
            **cond,
            **sampler_params,
            "verbose": True,
            "tqdm_desc": "Sampling sparse structure",
        }
        if inpaint_constr is not None:
            sample_kwargs["inpaint_constr"] = inpaint_constr
        if self.low_vram:
            flow_model.to(self.device)
        amp_context = (
            torch.autocast(device_type="cuda", dtype=flow_model.dtype)
            if self.device.type == "cuda" and getattr(flow_model, "dtype", None) in {torch.float16, torch.bfloat16}
            else nullcontext()
        )
        with amp_context:
            z_s = self.sparse_structure_sampler.sample(flow_model, noise, **sample_kwargs).samples
        if self.low_vram:
            flow_model.cpu()
        return z_s

    def decode_sparse_structure_latent_to_logit(self, resolution: int, z_s: torch.Tensor) -> torch.Tensor:
        self._require_components(required_model_keys=("sparse_structure_decoder",))

        decoder = self.models["sparse_structure_decoder"]
        if self.low_vram:
            decoder.to(self.device)
        decoder_dtype = next(decoder.parameters()).dtype
        decoded = decoder(z_s.to(dtype=decoder_dtype))
        if self.low_vram:
            decoder.cpu()

        if resolution != decoded.shape[2]:
            ratio = decoded.shape[2] // resolution
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0)

        return decoded

    def decode_sparse_structure_latent(self, resolution: int, z_s: torch.Tensor) -> torch.Tensor:
        self._require_components(required_model_keys=("sparse_structure_decoder",))

        decoder = self.models["sparse_structure_decoder"]
        if self.low_vram:
            decoder.to(self.device)
        decoder_dtype = next(decoder.parameters()).dtype
        decoded = decoder(z_s.to(dtype=decoder_dtype)) > 0
        if self.low_vram:
            decoder.cpu()

        if resolution != decoded.shape[2]:
            ratio = decoded.shape[2] // resolution
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5

        coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()
        return coords

    def sample_shape_slat(
        self,
        cond: dict,
        flow_model: nn.Module,
        coords: torch.Tensor,
        sampler_params: dict = {},
        inpaint_constr: Optional[EasyDict] = None,
    ) -> SparseTensor:
        self._require_components(
            required_attrs=("shape_slat_sampler", "shape_slat_sampler_params", "shape_slat_normalization"),
        )

        std = torch.tensor(self.shape_slat_normalization["std"])[None].to(self.device)
        mean = torch.tensor(self.shape_slat_normalization["mean"])[None].to(self.device)

        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        sample_kwargs = {
            **cond,
            **sampler_params,
            "verbose": True,
            "tqdm_desc": "Sampling shape SLat",
        }
        if inpaint_constr is not None:
            normalized_inpaint_constr = EasyDict(
                mask=inpaint_constr.mask,
                x0=(inpaint_constr.x0 - mean) / std,
            )
            sample_kwargs["inpaint_constr"] = normalized_inpaint_constr
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.shape_slat_sampler.sample(flow_model, noise, **sample_kwargs).samples
        if self.low_vram:
            flow_model.cpu()

        slat = slat * std + mean
        return slat

    def sample_shape_slat_cascade(
        self,
        lr_cond: dict,
        cond: dict,
        flow_model_lr: nn.Module,
        flow_model: nn.Module,
        lr_resolution: int,
        resolution: int,
        coords: torch.Tensor,
        sampler_params: dict = {},
        inpaint_constr_lr: Optional[EasyDict] = None,
        inpaint_constr_hr: Optional[EasyDict] = None,
        max_num_tokens: int = 75000,
    ) -> tuple[SparseTensor, int]:
        self._require_components(
            required_model_keys=("shape_slat_decoder",),
            required_attrs=("shape_slat_sampler", "shape_slat_sampler_params", "shape_slat_normalization"),
        )

        std = torch.tensor(self.shape_slat_normalization["std"])[None].to(self.device)
        mean = torch.tensor(self.shape_slat_normalization["mean"])[None].to(self.device)

        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model_lr.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        sample_kwargs = {
            **lr_cond,
            **sampler_params,
            "verbose": True,
            "tqdm_desc": "Sampling shape SLat low-res",
        }
        if inpaint_constr_lr is not None:
            normalized_inpaint_constr_lr = EasyDict(
                mask=inpaint_constr_lr.mask,
                x0=(inpaint_constr_lr.x0 - mean) / std,
            )
            sample_kwargs["inpaint_constr"] = normalized_inpaint_constr_lr
        if self.low_vram:
            flow_model_lr.to(self.device)
        slat = self.shape_slat_sampler.sample(flow_model_lr, noise, **sample_kwargs).samples
        if self.low_vram:
            flow_model_lr.cpu()

        slat = slat * std + mean

        decoder = self.models["shape_slat_decoder"]
        if self.low_vram:
            decoder.to(self.device)
            decoder.low_vram = True
        hr_coords = decoder.upsample(slat, upsample_times=4)
        if self.low_vram:
            decoder.cpu()
            decoder.low_vram = False

        hr_resolution = resolution

        while True:
            quant_coords = torch.cat(
                [
                    hr_coords[:, :1],
                    ((hr_coords[:, 1:] + 0.5) / lr_resolution * (hr_resolution // 16)).int(),
                ],
                dim=1,
            )
            coords = quant_coords.unique(dim=0)
            num_tokens = coords.shape[0]
            if num_tokens < max_num_tokens or hr_resolution == 1024:
                if hr_resolution != resolution:
                    print(f"Due to the limited number of tokens, the resolution is reduced to {hr_resolution}.")
                break
            hr_resolution -= 128

        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sample_kwargs = {
            **cond,
            **sampler_params,
            "verbose": True,
            "tqdm_desc": "Sampling shape SLat high-res",
        }
        if inpaint_constr_hr is not None:
            normalized_inpaint_constr_hr = EasyDict(
                mask=inpaint_constr_hr.mask,
                x0=(inpaint_constr_hr.x0 - mean) / std,
            )
            sample_kwargs["inpaint_constr"] = normalized_inpaint_constr_hr
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.shape_slat_sampler.sample(flow_model, noise, **sample_kwargs).samples
        if self.low_vram:
            flow_model.cpu()

        slat = slat * std + mean
        return slat, hr_resolution

    def decode_shape_slat(self, slat: SparseTensor, resolution: int) -> tuple[list[Mesh], list[SparseTensor]]:
        self._require_components(required_model_keys=("shape_slat_decoder",))

        decoder = self.models["shape_slat_decoder"]
        decoder.set_resolution(resolution)
        if self.low_vram:
            decoder.to(self.device)
            decoder.low_vram = True
        ret = decoder(slat, return_subs=True)
        if self.low_vram:
            decoder.cpu()
            decoder.low_vram = False
        return ret

    def sample_tex_slat(
        self,
        cond: dict,
        flow_model: nn.Module,
        shape_slat: SparseTensor,
        sampler_params: dict = {},
        inpaint_constr: Optional[EasyDict] = None,
    ) -> SparseTensor:
        self._require_components(
            required_attrs=(
                "tex_slat_sampler",
                "tex_slat_sampler_params",
                "shape_slat_normalization",
                "tex_slat_normalization",
            ),
        )
        std = torch.tensor(self.shape_slat_normalization["std"])[None].to(self.device)
        mean = torch.tensor(self.shape_slat_normalization["mean"])[None].to(self.device)
        shape_slat = (shape_slat - mean) / std

        tex_std = torch.tensor(self.tex_slat_normalization["std"])[None].to(self.device)
        tex_mean = torch.tensor(self.tex_slat_normalization["mean"])[None].to(self.device)
        in_channels = flow_model.in_channels if isinstance(flow_model, nn.Module) else flow_model[0].in_channels
        noise = shape_slat.replace(
            feats=torch.randn(shape_slat.coords.shape[0], in_channels - shape_slat.feats.shape[1]).to(self.device)
        )
        sampler_params = {**self.tex_slat_sampler_params, **sampler_params}
        sample_kwargs = {
            **cond,
            **sampler_params,
            "verbose": True,
            "tqdm_desc": "Sampling texture SLat",
        }
        if inpaint_constr is not None:
            normalized_inpaint_constr = EasyDict(
                mask=inpaint_constr.mask,
                x0=(inpaint_constr.x0 - tex_mean) / tex_std,
            )
            sample_kwargs["inpaint_constr"] = normalized_inpaint_constr
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.tex_slat_sampler.sample(flow_model, noise, concat_cond=shape_slat, **sample_kwargs).samples
        if self.low_vram:
            flow_model.cpu()

        slat = slat * tex_std + tex_mean
        return slat

    def decode_tex_slat(self, slat: SparseTensor, subs: list[SparseTensor]) -> SparseTensor:
        self._require_components(required_model_keys=("tex_slat_decoder",))

        decoder = self.models["tex_slat_decoder"]
        if self.low_vram:
            decoder.to(self.device)
        ret = decoder(slat, guide_subs=subs) * 0.5 + 0.5
        if self.low_vram:
            decoder.cpu()
        return ret

    @torch.no_grad()
    def decode_latent(
        self,
        shape_slat: SparseTensor,
        tex_slat: SparseTensor,
        resolution: int,
    ) -> list[MeshWithVoxel]:
        meshes, subs = self.decode_shape_slat(shape_slat, resolution)
        tex_voxels = self.decode_tex_slat(tex_slat, subs)
        out_mesh = []
        for mesh, voxels in zip(meshes, tex_voxels):
            mesh.fill_holes()
            out_mesh.append(
                MeshWithVoxel(
                    mesh.vertices,
                    mesh.faces,
                    origin=[-0.5, -0.5, -0.5],
                    voxel_size=1 / resolution,
                    coords=voxels.coords[:, 1:],
                    attrs=voxels.feats,
                    voxel_shape=torch.Size([*voxels.shape, *voxels.spatial_shape]),
                    layout=self.pbr_attr_layout,
                )
            )
        return out_mesh

    @torch.no_grad()
    def run_occupancy(
        self,
        images: list[Image.Image],
        intrinsics: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: Optional[dict] = None,
        resolutions: Union[int, Sequence[int]] = 64,
        num_cond_views: Optional[int] = None,
    ) -> dict[int, torch.Tensor]:
        """Run sparse-structure-only inference and return occupancy coordinates."""
        if isinstance(resolutions, int):
            resolutions = [resolutions]

        self._require_components(
            required_model_keys=("sparse_structure_flow_model", "sparse_structure_decoder"),
            required_attrs=("image_cond_model", "sparse_structure_sampler", "sparse_structure_sampler_params"),
        )

        torch.manual_seed(seed)

        cond_sparse = self.get_cond(
            images,
            512,
            model_key="sparse_structure_flow_model",
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            num_samples=num_samples,
            num_cond_views=num_cond_views,
        )
        ss_latent = self.sample_sparse_structure(
            cond_sparse,
            num_samples,
            sparse_structure_sampler_params or {},
        )

        coords_by_resolution: dict[int, torch.Tensor] = {}
        for target_resolution in resolutions:
            coords_by_resolution[target_resolution] = self.decode_sparse_structure_latent(target_resolution, ss_latent)

        return coords_by_resolution

    @torch.no_grad()
    def run_shape(
        self,
        coords: torch.Tensor,
        images: list[Image.Image],
        intrinsics: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None,
        shape_slat_sampler_params: dict = {},
        pipeline_type: Optional[str] = None,
        seed: int = 42,
        max_num_tokens: int = 75000,
    ) -> list[Mesh]:
        """Run the shape stage from precomputed sparse coords and return decoded meshes."""
        pipeline_type = pipeline_type or self.default_pipeline_type
        if pipeline_type is None:
            raise RuntimeError(
                "No pipeline_type was provided and this partially loaded pipeline has no default pipeline type."
            )
        if not images:
            raise ValueError("At least one input image is required.")

        if coords.ndim != 2 or coords.shape[1] not in {3, 4}:
            raise ValueError(f"Expected coords to have shape [K, 3] or [K, 4], got {tuple(coords.shape)}.")
        if coords.shape[1] == 3:
            batch_column = torch.zeros((coords.shape[0], 1), dtype=torch.int32, device=coords.device)
            coords = torch.cat([batch_column, coords.int()], dim=1)
        else:
            coords = coords.int()

        coords = coords.to(self.device)
        num_samples = int(coords[:, 0].max().item()) + 1
        torch.manual_seed(seed)

        if pipeline_type == "512":
            img_size_512 = self._image_size_by_model_key.get("shape_slat_flow_model_512", 512)
            cond_shape = self.get_cond(
                images,
                img_size_512,
                model_key="shape_slat_flow_model_512",
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                num_samples=num_samples,
                coords=coords,
            )
            shape_slat = self.sample_shape_slat(
                cond_shape,
                self.models["shape_slat_flow_model_512"],
                coords,
                shape_slat_sampler_params,
            )
            res = 512
        elif pipeline_type == "1024":
            img_size_1024 = self._image_size_by_model_key.get("shape_slat_flow_model_1024", 1024)
            cond_shape = self.get_cond(
                images,
                img_size_1024,
                model_key="shape_slat_flow_model_1024",
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                num_samples=num_samples,
                coords=coords,
            )
            shape_slat = self.sample_shape_slat(
                cond_shape,
                self.models["shape_slat_flow_model_1024"],
                coords,
                shape_slat_sampler_params,
            )
            res = 1024
        elif pipeline_type == "1024_cascade":
            img_size_512 = self._image_size_by_model_key.get("shape_slat_flow_model_512", 512)
            img_size_1024 = self._image_size_by_model_key.get("shape_slat_flow_model_1024", 1024)
            cond_shape_lr = self.get_cond(
                images,
                img_size_512,
                model_key="shape_slat_flow_model_512",
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                num_samples=num_samples,
                coords=coords,
            )
            cond_shape_hr = self.get_cond(
                images,
                img_size_1024,
                model_key="shape_slat_flow_model_1024",
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                num_samples=num_samples,
                coords=coords,
            )
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_shape_lr,
                cond_shape_hr,
                self.models["shape_slat_flow_model_512"],
                self.models["shape_slat_flow_model_1024"],
                512,
                1024,
                coords,
                shape_slat_sampler_params,
                max_num_tokens=max_num_tokens,
            )
        else:
            raise ValueError(
                f"Invalid pipeline type: {pipeline_type}. Supported values are '512', '1024', and '1024_cascade'."
            )

        torch.cuda.empty_cache()
        meshes, _ = self.decode_shape_slat(shape_slat, res)
        return meshes

    @torch.no_grad()
    def run_occ_and_shape(
        self,
        images: list[Image.Image],
        intrinsics: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: Optional[dict] = None,
        shape_slat_sampler_params: dict = {},
        pipeline_type: Optional[str] = None,
        max_num_tokens: int = 75000,
    ) -> list[Mesh]:
        """Run sparse occupancy generation followed by shape decoding and return meshes."""
        pipeline_type = pipeline_type or self.default_pipeline_type

        try:
            sparse_resolution = {"512": 32, "1024": 64, "1024_cascade": 32}[pipeline_type]
        except KeyError as exc:
            raise ValueError(
                f"Invalid pipeline type: {pipeline_type}. Supported values are '512', '1024', and '1024_cascade'."
            ) from exc

        coords_by_resolution = self.run_occupancy(
            images=images,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            num_samples=num_samples,
            seed=seed,
            sparse_structure_sampler_params=sparse_structure_sampler_params,
            resolutions=sparse_resolution,
        )
        coords = coords_by_resolution[sparse_resolution]

        return self.run_shape(
            coords=coords,
            images=images,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            shape_slat_sampler_params=shape_slat_sampler_params,
            pipeline_type=pipeline_type,
            seed=seed,
            max_num_tokens=max_num_tokens,
        )

    @torch.no_grad()
    def run(
        self,
        images: list[Image.Image],
        intrinsics: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        return_latent: bool = False,
        pipeline_type: Optional[str] = None,
        max_num_tokens: int = 75000,
    ) -> Union[list[MeshWithVoxel], tuple[list[MeshWithVoxel], tuple[SparseTensor, SparseTensor, int]]]:
        pipeline_type = pipeline_type or self.default_pipeline_type
        if pipeline_type is None:
            raise RuntimeError(
                "No pipeline_type was provided and this partially loaded pipeline has no default pipeline type."
            )
        if not images:
            raise ValueError("At least one input image is required.")

        self._require_run_components(pipeline_type)
        torch.manual_seed(seed)

        cond_sparse = self.get_cond(
            images,
            512,
            model_key="sparse_structure_flow_model",
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            num_samples=num_samples,
        )
        ss_res = {"512": 32, "1024": 64, "1024_cascade": 32, "1536_cascade": 32}[pipeline_type]
        ss_latent = self.sample_sparse_structure(cond_sparse, num_samples, sparse_structure_sampler_params)
        coords = self.decode_sparse_structure_latent(ss_res, ss_latent)

        if pipeline_type == "512":
            cond_shape = self.get_cond(
                images,
                512,
                model_key="shape_slat_flow_model_512",
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                num_samples=num_samples,
                coords=coords,
            )
            shape_slat = self.sample_shape_slat(
                cond_shape,
                self.models["shape_slat_flow_model_512"],
                coords,
                shape_slat_sampler_params,
            )
            cond_tex = self.get_cond(
                images,
                512,
                model_key="tex_slat_flow_model_512",
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                num_samples=num_samples,
                coords=shape_slat.coords,
            )
            tex_slat = self.sample_tex_slat(
                cond_tex, self.models["tex_slat_flow_model_512"], shape_slat, tex_slat_sampler_params
            )
            res = 512
        elif pipeline_type == "1024":
            cond_shape = self.get_cond(
                images,
                1024,
                model_key="shape_slat_flow_model_1024",
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                num_samples=num_samples,
                coords=coords,
            )
            shape_slat = self.sample_shape_slat(
                cond_shape,
                self.models["shape_slat_flow_model_1024"],
                coords,
                shape_slat_sampler_params,
            )
            cond_tex = self.get_cond(
                images,
                1024,
                model_key="tex_slat_flow_model_1024",
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                num_samples=num_samples,
                coords=shape_slat.coords,
            )
            tex_slat = self.sample_tex_slat(
                cond_tex, self.models["tex_slat_flow_model_1024"], shape_slat, tex_slat_sampler_params
            )
            res = 1024
        elif pipeline_type == "1024_cascade":
            cond_shape_lr = self.get_cond(
                images,
                512,
                model_key="shape_slat_flow_model_512",
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                num_samples=num_samples,
                coords=coords,
            )
            cond_shape_hr = self.get_cond(
                images,
                1024,
                model_key="shape_slat_flow_model_1024",
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                num_samples=num_samples,
                coords=coords,
            )
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_shape_lr,
                cond_shape_hr,
                self.models["shape_slat_flow_model_512"],
                self.models["shape_slat_flow_model_1024"],
                512,
                1024,
                coords,
                shape_slat_sampler_params,
                max_num_tokens=max_num_tokens,
            )
            cond_tex = self.get_cond(
                images,
                1024,
                model_key="tex_slat_flow_model_1024",
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                num_samples=num_samples,
                coords=shape_slat.coords,
            )
            tex_slat = self.sample_tex_slat(
                cond_tex, self.models["tex_slat_flow_model_1024"], shape_slat, tex_slat_sampler_params
            )
        elif pipeline_type == "1536_cascade":
            cond_shape_lr = self.get_cond(
                images,
                512,
                model_key="shape_slat_flow_model_512",
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                num_samples=num_samples,
                coords=coords,
            )
            cond_shape_hr = self.get_cond(
                images,
                1024,
                model_key="shape_slat_flow_model_1024",
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                num_samples=num_samples,
                coords=coords,
            )
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_shape_lr,
                cond_shape_hr,
                self.models["shape_slat_flow_model_512"],
                self.models["shape_slat_flow_model_1024"],
                512,
                1536,
                coords,
                shape_slat_sampler_params,
                max_num_tokens=max_num_tokens,
            )
            cond_tex = self.get_cond(
                images,
                1024,
                model_key="tex_slat_flow_model_1024",
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                num_samples=num_samples,
                coords=shape_slat.coords,
            )
            tex_slat = self.sample_tex_slat(
                cond_tex, self.models["tex_slat_flow_model_1024"], shape_slat, tex_slat_sampler_params
            )
        else:
            raise ValueError(f"Invalid pipeline type: {pipeline_type}")

        torch.cuda.empty_cache()
        out_mesh = self.decode_latent(shape_slat, tex_slat, res)
        if return_latent:
            return out_mesh, (shape_slat, tex_slat, res)
        return out_mesh
