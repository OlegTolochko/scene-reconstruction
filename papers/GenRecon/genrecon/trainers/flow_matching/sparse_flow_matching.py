import functools
from typing import *

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

from ...modules import sparse as sp
from ...utils.data_utils import BalancedResumableSampler, cycle, recursive_to_device
from ...utils.general_utils import dict_reduce
from .flow_matching import FlowMatchingTrainer
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.image_conditioned import CameraConditionedMixin


class SparseFlowMatchingTrainer(FlowMatchingTrainer):
    """
    Trainer for sparse diffusion model with flow matching objective.

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

    def prepare_dataloader(self, **kwargs):
        """
        Prepare dataloader.
        """
        self.data_sampler = BalancedResumableSampler(
            self.dataset,
            shuffle=True,
            batch_size=self.batch_size_per_gpu,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=self.num_workers_per_gpu,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            timeout=120,
            collate_fn=functools.partial(self.dataset.collate_fn, split_size=self.batch_split),
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)

        if self.val_dataset is None:
            self.val_dataloader = None
        else:
            self.val_data_sampler = BalancedResumableSampler(
                self.val_dataset,
                shuffle=False,
                batch_size=self.batch_size_per_gpu // self.batch_split,
            )
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size_per_gpu // self.batch_split,
                num_workers=self.num_workers_per_gpu,
                pin_memory=True,
                drop_last=False,
                persistent_workers=True,
                timeout=120,
                collate_fn=functools.partial(self.val_dataset.collate_fn),
                sampler=self.val_data_sampler,
            )

    def training_losses(
        self, x_0: sp.SparseTensor, cond=None, extrinsics=None, intrinsics=None, cond_batch_idx=None, **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.

        Args:
            x_0: The [N x ... x C] sparse tensor of the inputs.
            cond: The [N x ...] tensor of additional conditions.
            kwargs: Additional arguments to pass to the backbone.

        Returns:
            a dict with the key "loss" containing a tensor of shape [N].
            may also contain other keys for different terms.
        """
        noise = x_0.replace(torch.randn_like(x_0.feats))
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
        terms["mse"] = F.mse_loss(pred.feats, target.feats)
        terms["loss"] = terms["mse"]

        # log loss with time bins
        mse_per_instance = np.array(
            [F.mse_loss(pred.feats[x_0.layout[i]], target.feats[x_0.layout[i]]).item() for i in range(x_0.shape[0])]
        )
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean()}

        return terms, {}

    def validation_losses(
        self, x_0: sp.SparseTensor, cond=None, extrinsics=None, intrinsics=None, cond_batch_idx=None, **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.

        Args:
            x_0: The [N x ... x C] sparse tensor of the inputs.
            cond: The [N x ...] tensor of additional conditions.
            kwargs: Additional arguments to pass to the backbone.

        Returns:
            a dict with the key "loss" containing a tensor of shape [N].
            may also contain other keys for different terms.
        """
        noise = x_0.replace(torch.randn_like(x_0.feats))
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
        terms["mse"] = F.mse_loss(pred.feats, target.feats)
        terms["loss"] = terms["mse"]

        # log loss with time bins
        mse_per_instance = np.array(
            [F.mse_loss(pred.feats[x_0.layout[i]], target.feats[x_0.layout[i]]).item() for i in range(x_0.shape[0])]
        )
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean()}

        return terms, {}

    def _merge_snapshot_batches(self, batches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge per-batch snapshot payloads while preserving conditioning fields.
        """
        if len(batches) == 0:
            return {}

        def merge_values(values: List[Any]) -> Any:
            first = values[0]
            if isinstance(first, dict):
                return {key: merge_values([v[key] for v in values]) for key in first.keys()}
            if isinstance(first, sp.SparseTensor):
                return sp.sparse_cat(values)
            if isinstance(first, torch.Tensor):
                return torch.cat(values, dim=0)
            if isinstance(first, list):
                return sum(values, [])
            if isinstance(first, tuple):
                return type(first)(sum((list(v) for v in values), []))
            return values

        return {key: merge_values([batch[key] for batch in batches]) for key in batches[0].keys()}

    @torch.no_grad()
    def run_snapshot(
        self,
        dataloader,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        sampler = self.get_sampler()
        data_iter = iter(dataloader)

        sample_gt_batches = []
        sample_batches = []
        cond_vis = []
        collected = 0

        while collected < num_samples:
            try:
                data = next(data_iter)
            except StopIteration:
                break

            curr = min(batch_size, num_samples - collected, data["x_0"].shape[0])
            if "cond_batch_idx" in data:
                # flat-format: cond/extrinsics/intrinsics are indexed by view, not batch element
                keep = data["cond_batch_idx"] < curr
                flat_keys = {"cond", "extrinsics", "intrinsics", "cond_batch_idx"}
                batch_data = {k: (data[k][keep] if k in flat_keys else v[:curr]) for k, v in data.items()}
            else:
                batch_data = {k: v[:curr] for k, v in data.items()}
            batch_data = recursive_to_device(batch_data, "cuda")

            noise = batch_data["x_0"].replace(torch.randn_like(batch_data["x_0"].feats))
            sample_gt_batches.append({k: v for k, v in batch_data.items()})
            cond_vis.append(self.vis_cond(**batch_data))

            cond_data = {k: v for k, v in batch_data.items() if k != "x_0"}
            args = self.get_inference_cond(self.models["denoiser"], x_0=batch_data["x_0"], **cond_data)

            res = sampler.sample(
                self.models["denoiser"],
                noise=noise,
                **args,
                steps=12,
                guidance_strength=3.0,
                verbose=verbose,
            )
            sample_batches.append({"x_0": res.samples, **cond_data})
            collected += curr

        if len(sample_batches) == 0:
            return {}

        sample_gt = self._merge_snapshot_batches(sample_gt_batches)
        sample = self._merge_snapshot_batches(sample_batches)

        sample_dict = {
            "sample_gt": {"value": sample_gt, "type": "sample"},
            "sample": {"value": sample, "type": "sample"},
        }

        if len(cond_vis) > 0:
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


class SparseFlowMatchingCFGTrainer(ClassifierFreeGuidanceMixin, SparseFlowMatchingTrainer):
    """
    Trainer for sparse diffusion model with flow matching objective and classifier-free guidance.

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


class CameraConditionedSparseFlowMatchingCFGTrainer(CameraConditionedMixin, SparseFlowMatchingCFGTrainer):
    pass
