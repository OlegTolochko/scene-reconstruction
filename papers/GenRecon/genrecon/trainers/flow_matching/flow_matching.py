import functools
from typing import *

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

from ...pipelines import samplers
from ...utils.data_utils import ResumableSampler, cycle
from ...utils.general_utils import dict_reduce
from ..basic import BasicTrainer
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.image_conditioned import CameraConditionedMixin


class FlowMatchingTrainer(BasicTrainer):
    """
    Trainer for diffusion model with flow matching objective.

    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
    """

    def __init__(
        self,
        *args,
        t_schedule: dict = {
            "name": "logitNormal",
            "args": {
                "mean": 0.0,
                "std": 1.0,
            },
        },
        sigma_min: float = 1e-5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.t_schedule = t_schedule
        self.sigma_min = sigma_min

    def diffuse(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            t: The [N] tensor of diffusion steps [0-1].
            noise: If specified, use this noise instead of generating new noise.

        Returns:
            x_t, the noisy version of x_0 under timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        assert noise.shape == x_0.shape, "noise must have same shape as x_0"

        t = t.view(-1, *[1 for _ in range(len(x_0.shape) - 1)])
        x_t = (1 - t) * x_0 + (self.sigma_min + (1 - self.sigma_min) * t) * noise

        return x_t

    def reverse_diffuse(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Get original image from noisy version under timestep t.
        """
        assert noise.shape == x_t.shape, "noise must have same shape as x_t"
        t = t.view(-1, *[1 for _ in range(len(x_t.shape) - 1)])
        x_0 = (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * noise) / (1 - t)
        return x_0

    def get_v(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the velocity of the diffusion process at time t.
        """
        return (1 - self.sigma_min) * noise - x_0

    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data.
        """
        return cond

    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.
        """
        return {"cond": cond, **kwargs}

    def get_sampler(self, **kwargs) -> samplers.FlowEulerSampler:
        """
        Get the sampler for the diffusion process.
        """
        return samplers.FlowEulerSampler(self.sigma_min)

    def vis_cond(self, **kwargs):
        """
        Visualize the conditioning data.
        """
        return {}

    def sample_t(self, batch_size: int) -> torch.Tensor:
        """
        Sample timesteps.
        """
        if self.t_schedule["name"] == "uniform":
            t = torch.rand(batch_size)
        elif self.t_schedule["name"] == "logitNormal":
            mean = self.t_schedule["args"]["mean"]
            std = self.t_schedule["args"]["std"]
            t = torch.sigmoid(torch.randn(batch_size) * std + mean)
        else:
            raise ValueError(f"Unknown t_schedule: {self.t_schedule['name']}")
        return t

    def prepare_dataloader(self, **kwargs):
        """
        Prepare dataloader. Passes split_size to collate_fn so that flat-format
        view tensors (cond/extrinsics/intrinsics) are split correctly across
        gradient-accumulation micro-batches instead of naive tensor slicing.
        """
        self.data_sampler = ResumableSampler(self.dataset, shuffle=True)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=self.num_workers_per_gpu,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            timeout=120,
            collate_fn=(
                functools.partial(self.dataset.collate_fn, split_size=self.batch_split)
                if hasattr(self.dataset, "collate_fn")
                else None
            ),
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)

        if self.val_dataset is None:
            self.val_dataloader = None
        else:
            self.val_data_sampler = ResumableSampler(self.val_dataset, shuffle=False)
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size_per_gpu // self.batch_split,
                num_workers=self.num_workers_per_gpu,
                pin_memory=True,
                drop_last=False,
                persistent_workers=True,
                timeout=120,
                collate_fn=(
                    functools.partial(self.val_dataset.collate_fn) if hasattr(self.val_dataset, "collate_fn") else None
                ),
                sampler=self.val_data_sampler,
            )

    def training_losses(
        self, x_0: torch.Tensor, cond=None, extrinsics=None, intrinsics=None, cond_batch_idx=None, **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            cond: The [N x ...] tensor of additional conditions.
            kwargs: Additional arguments to pass to the backbone.

        Returns:
            a dict with the key "loss" containing a tensor of shape [N].
            may also contain other keys for different terms.
        """
        noise = torch.randn_like(x_0)
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        denoiser = self.training_models["denoiser"]
        if hasattr(denoiser, "module"):
            denoiser = denoiser.module
        cond = self.get_cond(
            denoiser,
            cond,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            cond_batch_idx=cond_batch_idx,
            x_0=x_0,
            **kwargs,
        )

        pred = self.training_models["denoiser"](x_t, t * 1000, cond, **kwargs)
        assert pred.shape == noise.shape == x_0.shape
        target = self.get_v(x_0, noise, t)
        terms = edict()
        terms["mse"] = F.mse_loss(pred, target)
        terms["loss"] = terms["mse"]

        # log loss with time bins
        mse_per_instance = np.array([F.mse_loss(pred[i], target[i]).item() for i in range(x_0.shape[0])])
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean()}

        status = {}

        return terms, status

    def validation_losses(
        self, x_0: torch.Tensor, cond=None, extrinsics=None, intrinsics=None, cond_batch_idx=None, **kwargs
    ) -> Tuple[Dict, Dict]:
        noise = torch.randn_like(x_0)
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        denoiser = self.training_models["denoiser"]
        if hasattr(denoiser, "module"):
            denoiser = denoiser.module
        cond = self.get_cond(
            denoiser,
            cond,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            cond_batch_idx=cond_batch_idx,
            x_0=x_0,
            **kwargs,
        )

        pred = self.training_models["denoiser"](x_t, t * 1000, cond, **kwargs)
        assert pred.shape == noise.shape == x_0.shape
        target = self.get_v(x_0, noise, t)
        terms = edict()
        terms["mse"] = F.mse_loss(pred, target)
        terms["loss"] = terms["mse"]

        # log loss with time bins
        mse_per_instance = np.array([F.mse_loss(pred[i], target[i]).item() for i in range(x_0.shape[0])])
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean()}

        return terms, {}

    @torch.no_grad()
    def run_snapshot(
        self,
        dataloader,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:

        # inference
        sampler = self.get_sampler()
        data_iter = iter(dataloader)
        sample_gt = []
        sample = []
        cond_vis = []
        for i in range(0, num_samples, batch_size):
            try:
                data = next(data_iter)
            except StopIteration:
                break
            batch = min(batch_size, num_samples - i, next(iter(data.values())).shape[0])
            if "cond_batch_idx" in data:
                keep = data["cond_batch_idx"] < batch
                flat_keys = {"cond", "extrinsics", "intrinsics", "cond_batch_idx"}
                data = {
                    k: (
                        (v[keep].cuda() if isinstance(v, torch.Tensor) else v[keep])
                        if k in flat_keys
                        else (v[:batch].cuda() if isinstance(v, torch.Tensor) else v[:batch])
                    )
                    for k, v in data.items()
                }
            else:
                data = {k: v[:batch].cuda() if isinstance(v, torch.Tensor) else v[:batch] for k, v in data.items()}
            x_0 = data.pop("x_0")
            noise = torch.randn_like(x_0)
            sample_gt.append(x_0)
            cond_vis.append(self.vis_cond(**data))
            args = self.get_inference_cond(self.models["denoiser"], x_0=x_0, **data)
            res = sampler.sample(
                self.models["denoiser"],
                noise=noise,
                **args,
                steps=50,
                guidance_strength=3.0,
                verbose=verbose,
            )
            sample.append(res.samples)

        sample_gt = torch.cat(sample_gt, dim=0)
        sample = torch.cat(sample, dim=0)
        sample_dict = {
            "sample_gt": {"value": sample_gt, "type": "sample"},
            "sample": {"value": sample, "type": "sample"},
        }
        sample_dict.update(
            dict_reduce(
                cond_vis,
                None,
                {
                    "value": lambda x: torch.cat(x, dim=0),
                    "type": lambda x: x[0],
                },
            )
        )

        return sample_dict


class FlowMatchingCFGTrainer(ClassifierFreeGuidanceMixin, FlowMatchingTrainer):
    """
    Trainer for diffusion model with flow matching objective and classifier-free guidance.

    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
    """

    pass


class CameraConditionedFlowMatchingCFGTrainer(CameraConditionedMixin, FlowMatchingCFGTrainer):
    pass
