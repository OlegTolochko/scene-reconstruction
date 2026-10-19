import copy
import json
import os
import threading
import time
from abc import abstractmethod
from contextlib import nullcontext
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils

from ..utils import elastic_utils, grad_clip_utils
from ..utils.data_utils import ResumableSampler, cycle, recursive_to_device
from ..utils.dist_utils import *
from ..utils.general_utils import *
from .utils import *


class BasicTrainer:
    """
    Trainer for basic training loop.

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
        mix_precision_mode (str):
            - None: No mixed precision.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        mix_precision_dtype (str): Mixed precision dtype.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        parallel_mode (str): Parallel mode. Options are 'ddp'.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.
    """

    def __init__(
        self,
        models,
        dataset,
        *,
        output_dir,
        load_dir,
        step,
        max_steps,
        val_dataset=None,
        batch_size=None,
        batch_size_per_gpu=None,
        batch_split=None,
        num_workers_per_gpu=1,
        optimizer={},
        lr_scheduler=None,
        elastic=None,
        grad_clip=None,
        ema_rate=0.9999,
        fp16_mode=None,
        mix_precision_mode="inflat_all",
        mix_precision_dtype="float16",
        fp16_scale_growth=1e-3,
        parallel_mode="ddp",
        finetune_ckpt=None,
        log_param_stats=False,
        prefetch_data=True,
        snapshot_batch_size=4,
        i_val=10,
        i_print=1000,
        i_log=500,
        i_sample=10000,
        i_save=10000,
        i_ddpcheck=10000,
        **kwargs,
    ):
        assert (
            batch_size is not None or batch_size_per_gpu is not None
        ), "Either batch_size or batch_size_per_gpu must be specified."

        self.models = models
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.batch_split = batch_split if batch_split is not None else 1
        self.num_workers_per_gpu = num_workers_per_gpu
        self.max_steps = max_steps
        self.optimizer_config = optimizer
        self.lr_scheduler_config = lr_scheduler
        self.elastic_controller_config = elastic
        self.grad_clip = grad_clip
        self.ema_rate = [ema_rate] if isinstance(ema_rate, float) else ema_rate
        if fp16_mode is not None:
            mix_precision_dtype = "float16"
            mix_precision_mode = fp16_mode
        self.mix_precision_mode = mix_precision_mode
        self.mix_precision_dtype = str_to_dtype(mix_precision_dtype)
        self.fp16_scale_growth = fp16_scale_growth
        self.parallel_mode = parallel_mode
        self.log_param_stats = log_param_stats
        self.prefetch_data = prefetch_data
        self.snapshot_batch_size = snapshot_batch_size
        self.log = []
        self.pretrained_trainable_param_names = set()
        if self.prefetch_data:
            self._data_prefetched = None

        self.output_dir = output_dir
        self.i_val = i_val
        self.i_print = i_print
        self.i_log = i_log
        self.i_sample = i_sample
        self.i_save = i_save
        self.i_ddpcheck = i_ddpcheck

        if dist.is_initialized():
            # Multi-GPU params
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.local_rank = dist.get_rank() % torch.cuda.device_count()
            self.is_master = self.rank == 0
        else:
            # Single-GPU params
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0
            self.is_master = True

        self.batch_size = batch_size if batch_size_per_gpu is None else batch_size_per_gpu * self.world_size
        self.batch_size_per_gpu = (
            batch_size_per_gpu if batch_size_per_gpu is not None else batch_size // self.world_size
        )
        assert self.batch_size % self.world_size == 0, "Batch size must be divisible by the number of GPUs."
        assert self.batch_size_per_gpu % self.batch_split == 0, "Batch size per GPU must be divisible by batch split."

        self.init_models_and_more(**kwargs)
        self.prepare_dataloader(**kwargs)

        if load_dir is not None and step is not None and finetune_ckpt is not None:
            self.pretrained_trainable_param_names = self.infer_pretrained_trainable_param_names(finetune_ckpt)

        # Load checkpoint
        self.step = 0
        if load_dir is not None and step is not None:
            self.init_optimizer_and_more()
            self.load(load_dir, step)
        elif finetune_ckpt is not None:
            self.finetune_from(finetune_ckpt)
            self.init_optimizer_and_more()
        else:
            self.init_optimizer_and_more()

        if self.is_master:
            os.makedirs(os.path.join(self.output_dir, "ckpts"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "samples"), exist_ok=True)
            self.writer = SummaryWriter(os.path.join(self.output_dir, "tb_logs"))

        if self.parallel_mode == "ddp" and self.world_size > 1:
            self.check_ddp()

        if self.is_master:
            print("\n\nTrainer initialized.")
            print(self)

    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f"  - Models:")
        for name, model in self.models.items():
            lines.append(f"    - {name}: {model.__class__.__name__}")
        lines.append(f"  - Dataset: {indent(str(self.dataset), 2)}")
        lines.append(f"  - Dataloader:")
        lines.append(f"    - Sampler: {self.dataloader.sampler.__class__.__name__}")
        lines.append(f"    - Num workers: {self.dataloader.num_workers}")
        lines.append(f"  - Number of steps: {self.max_steps}")
        lines.append(f"  - Number of GPUs: {self.world_size}")
        lines.append(f"  - Batch size: {self.batch_size}")
        lines.append(f"  - Batch size per GPU: {self.batch_size_per_gpu}")
        lines.append(f"  - Batch split: {self.batch_split}")
        lines.append(f"  - Optimizer: {self.optimizer.__class__.__name__}")
        if len(self.optimizer.param_groups) == 1:
            lines.append(f'  - Learning rate: {self.optimizer.param_groups[0]["lr"]}')
            lines.append(f'  - Weight decay: {self.optimizer.param_groups[0].get("weight_decay", 0.0)}')
        else:
            lines.append("  - Optimizer groups:")
            for i, group in enumerate(self.optimizer.param_groups):
                group_name = group.get("name", f"group_{i}")
                lines.append(
                    f'    - {group_name}: lr={group["lr"]}, weight_decay={group.get("weight_decay", 0.0)}, params={len(group["params"])}'
                )
        if self.lr_scheduler_config is not None:
            lines.append(f"  - LR scheduler: {self.lr_scheduler.__class__.__name__}")
        if self.elastic_controller_config is not None:
            lines.append(f"  - Elastic memory: {indent(str(self.elastic_controller), 2)}")
        if self.grad_clip is not None:
            lines.append(f"  - Gradient clip: {indent(str(self.grad_clip), 2)}")
        lines.append(f"  - EMA rate: {self.ema_rate}")
        lines.append(f"  - Mixed precision dtype: {self.mix_precision_dtype}")
        lines.append(f"  - Mixed precision mode: {self.mix_precision_mode}")
        if self.mix_precision_mode == "amp" and self.mix_precision_dtype == torch.float16:
            lines.append(f"  - FP16 scale growth: {self.fp16_scale_growth}")
        lines.append(f"  - Parallel mode: {self.parallel_mode}")
        return "\n".join(lines)

    @property
    def device(self):
        for _, model in self.models.items():
            if hasattr(model, "device"):
                return model.device
        return next(list(self.models.values())[0].parameters()).device

    def init_models_and_more(self, **kwargs):
        """
        Initialize models and master parameter state.
        """
        if self.world_size > 1:
            # Prepare distributed data parallel
            self.training_models = {
                name: DDP(
                    model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    bucket_cap_mb=128,
                    find_unused_parameters=False,
                )
                for name, model in self.models.items()
            }
        else:
            self.training_models = self.models

        # Build master params
        self.model_params = []
        self.trainable_named_params = []
        for model_name, model in self.models.items():
            trainable_matrices = 0
            for param_name, p in model.named_parameters():
                if p.requires_grad:
                    self.model_params.append(p)
                    self.trainable_named_params.append((f"{model_name}.{param_name}", p))
                    trainable_matrices += 1
            trainable_number = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Trainable parameters in {model_name}: {trainable_number} in {trainable_matrices} matrices")

        if self.mix_precision_mode == "amp":
            self.master_params = self.model_params
            if self.mix_precision_dtype == torch.float16:
                self.scaler = torch.GradScaler()
        elif self.mix_precision_mode == "inflat_all":
            self.master_params = make_master_params(self.model_params)
            if self.mix_precision_dtype == torch.float16:
                self.log_scale = 20.0
        elif self.mix_precision_mode is None:
            self.master_params = self.model_params
        else:
            raise NotImplementedError(f"Mix precision mode {self.mix_precision_mode} is not implemented.")

        # Build EMA params
        if self.is_master:
            self.ema_params = [copy.deepcopy(self.master_params) for _ in self.ema_rate]

    def init_optimizer_and_more(self):
        """
        Initialize optimizer, schedulers, and training-time controllers.
        """

        # Initialize optimizer
        optimizer_args = dict(self.optimizer_config["args"])
        lr_cfg = optimizer_args.get("lr")
        wd_cfg = optimizer_args.get("weight_decay", 0.0)
        use_param_group_builder = isinstance(lr_cfg, dict) or isinstance(wd_cfg, dict)

        if use_param_group_builder:
            if self.mix_precision_mode == "inflat_all":
                raise ValueError("Optimizer param-group builder is not supported with mix_precision_mode='inflat_all'.")

            optimizer_params, group_stats = build_optimizer_param_groups(
                self.trainable_named_params,
                lr=lr_cfg,
                weight_decay=wd_cfg,
                pretrained_param_names=self.pretrained_trainable_param_names,
            )
            optimizer_args.pop("lr", None)
            optimizer_args.pop("weight_decay", None)

            if self.is_master:
                print("Optimizer param groups:")
                for group in optimizer_params:
                    group_name = group["name"]
                    print(
                        f'  - {group_name}: lr={group["lr"]}, weight_decay={group["weight_decay"]}, params={group_stats[group_name]:,}'
                    )
                print(f'  - total: {group_stats["total"]:,}')
        else:
            optimizer_params = self.master_params

        if hasattr(torch.optim, self.optimizer_config["name"]):
            self.optimizer = getattr(torch.optim, self.optimizer_config["name"])(optimizer_params, **optimizer_args)
        else:
            self.optimizer = globals()[self.optimizer_config["name"]](optimizer_params, **optimizer_args)

        # Initalize learning rate scheduler
        if self.lr_scheduler_config is not None:
            if hasattr(torch.optim.lr_scheduler, self.lr_scheduler_config["name"]):
                self.lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_config["name"])(
                    self.optimizer, **self.lr_scheduler_config["args"]
                )
            else:
                self.lr_scheduler = globals()[self.lr_scheduler_config["name"]](
                    self.optimizer, **self.lr_scheduler_config["args"]
                )

        # Initialize elastic memory controller
        if self.elastic_controller_config is not None:
            assert any(
                [
                    isinstance(model, (elastic_utils.ElasticModule, elastic_utils.ElasticModuleMixin))
                    for model in self.models.values()
                ]
            ), "No elastic module found in models, please inherit from ElasticModule or ElasticModuleMixin"
            self.elastic_controller = getattr(elastic_utils, self.elastic_controller_config["name"])(
                **self.elastic_controller_config["args"]
            )
            for model in self.models.values():
                if isinstance(model, (elastic_utils.ElasticModule, elastic_utils.ElasticModuleMixin)):
                    model.register_memory_controller(self.elastic_controller)

        # Initialize gradient clipper
        if self.grad_clip is not None:
            if isinstance(self.grad_clip, (float, int)):
                self.grad_clip = float(self.grad_clip)
            else:
                self.grad_clip = getattr(grad_clip_utils, self.grad_clip["name"])(**self.grad_clip["args"])

    def load_finetune_model_ckpt(self, path, model_name):
        if path.endswith(".pt"):
            if self.is_master:
                print(f"  [{model_name}] Loading local .pt checkpoint from {path}")
            return torch.load(read_file_dist(path), map_location=self.device, weights_only=True)

        if self.is_master:
            print(f"  [{model_name}] Loading Hugging Face checkpoint from {path}")

        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        path_parts = path.split("/")
        repo_id = f"{path_parts[0]}/{path_parts[1]}"
        model_path = "/".join(path_parts[2:])
        model_file = hf_hub_download(repo_id, f"{model_path}.safetensors")
        return load_file(model_file)

    def infer_pretrained_trainable_param_names(self, finetune_ckpt):
        pretrained_trainable_param_names = set()
        for model_name, model in self.models.items():
            if model_name not in finetune_ckpt:
                continue

            model_state_dict = model.state_dict()
            trainable_param_names = {param_name for param_name, p in model.named_parameters() if p.requires_grad}
            model_ckpt = self.load_finetune_model_ckpt(finetune_ckpt[model_name], model_name)

            for param_name, param in model_ckpt.items():
                if param_name not in trainable_param_names:
                    continue
                if param_name not in model_state_dict:
                    continue
                if param.shape != model_state_dict[param_name].shape:
                    continue
                pretrained_trainable_param_names.add(f"{model_name}.{param_name}")

        return pretrained_trainable_param_names

    def prepare_dataloader(self, **kwargs):
        """
        Prepare dataloader.
        """
        self.data_sampler = ResumableSampler(
            self.dataset,
            shuffle=True,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=self.num_workers_per_gpu,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            timeout=120,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, "collate_fn") else None,
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)

        if self.val_dataset is None:
            self.val_dataloader = None
        else:
            self.val_data_sampler = ResumableSampler(
                self.val_dataset,
                shuffle=False,
            )
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size_per_gpu // self.batch_split,
                num_workers=self.num_workers_per_gpu,
                pin_memory=True,
                drop_last=False,
                persistent_workers=True,
                timeout=120,
                collate_fn=self.val_dataset.collate_fn if hasattr(self.val_dataset, "collate_fn") else None,
                sampler=self.val_data_sampler,
            )

    def _master_params_to_state_dicts(self, master_params):
        """
        Convert master params to dict of state_dicts.
        """
        if self.mix_precision_mode == "inflat_all":
            master_params = unflatten_master_params(self.model_params, master_params)
        state_dicts = {name: model.state_dict() for name, model in self.models.items()}
        master_params_names = sum(
            [
                [(name, n) for n, p in model.named_parameters() if p.requires_grad]
                for name, model in self.models.items()
            ],
            [],
        )
        for i, (model_name, param_name) in enumerate(master_params_names):
            state_dicts[model_name][param_name] = master_params[i]
        return state_dicts

    def _state_dicts_to_master_params(self, master_params, state_dicts):
        """
        Convert a state_dict to master params.
        """
        master_params_names = sum(
            [
                [(name, n) for n, p in model.named_parameters() if p.requires_grad]
                for name, model in self.models.items()
            ],
            [],
        )
        params = [state_dicts[name][param_name] for name, param_name in master_params_names]
        if self.mix_precision_mode == "inflat_all":
            model_params_to_master_params(params, master_params)
        else:
            for i, param in enumerate(params):
                master_params[i].data.copy_(param.data)

    def load(self, load_dir, step=0):
        """
        Load a checkpoint.
        Should be called by all processes.
        """
        if self.is_master:
            print(f"\nLoading checkpoint from step {step}...", end="")

        model_ckpts = {}
        for name, model in self.models.items():
            model_ckpt = torch.load(
                read_file_dist(os.path.join(load_dir, "ckpts", f"{name}_step{step:07d}.pt")),
                map_location=self.device,
                weights_only=True,
            )
            model_ckpts[name] = model_ckpt
            model.load_state_dict(model_ckpt)
        self._state_dicts_to_master_params(self.master_params, model_ckpts)
        del model_ckpts

        if self.is_master:
            for i, ema_rate in enumerate(self.ema_rate):
                ema_ckpts = {}
                for name, model in self.models.items():
                    ema_ckpt = torch.load(
                        os.path.join(load_dir, "ckpts", f"{name}_ema{ema_rate}_step{step:07d}.pt"),
                        map_location=self.device,
                        weights_only=True,
                    )
                    ema_ckpts[name] = ema_ckpt
                self._state_dicts_to_master_params(self.ema_params[i], ema_ckpts)
                del ema_ckpts

        misc_ckpt = torch.load(
            read_file_dist(os.path.join(load_dir, "ckpts", f"misc_step{step:07d}.pt")),
            map_location=torch.device("cpu"),
            weights_only=False,
        )
        self.optimizer.load_state_dict(misc_ckpt["optimizer"])
        self.step = misc_ckpt["step"]
        self.data_sampler.load_state_dict(misc_ckpt["data_sampler"])
        if self.mix_precision_mode == "amp" and self.mix_precision_dtype == torch.float16:
            self.scaler.load_state_dict(misc_ckpt["scaler"])
        elif self.mix_precision_mode == "inflat_all" and self.mix_precision_dtype == torch.float16:
            self.log_scale = misc_ckpt["log_scale"]
        if self.lr_scheduler_config is not None:
            self.lr_scheduler.load_state_dict(misc_ckpt["lr_scheduler"])
        if self.elastic_controller_config is not None:
            self.elastic_controller.load_state_dict(misc_ckpt["elastic_controller"])
        if self.grad_clip is not None and not isinstance(self.grad_clip, float):
            self.grad_clip.load_state_dict(misc_ckpt["grad_clip"])
        del misc_ckpt

        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            print(" Done.")

        if self.world_size > 1:
            self.check_ddp()

    def save(self, non_blocking=True):
        """
        Save a checkpoint.
        Should be called only by the rank 0 process.
        """
        assert self.is_master, "save() should be called only by the rank 0 process."
        print(f"\nSaving checkpoint at step {self.step}...", end="")

        model_ckpts = self._master_params_to_state_dicts(self.master_params)
        for name, model_ckpt in model_ckpts.items():
            model_ckpt = {k: v.cpu() for k, v in model_ckpt.items()}  # Move to CPU for saving
            if non_blocking:
                threading.Thread(
                    target=torch.save,
                    args=(model_ckpt, os.path.join(self.output_dir, "ckpts", f"{name}_step{self.step:07d}.pt")),
                ).start()
            else:
                torch.save(model_ckpt, os.path.join(self.output_dir, "ckpts", f"{name}_step{self.step:07d}.pt"))

        for i, ema_rate in enumerate(self.ema_rate):
            ema_ckpts = self._master_params_to_state_dicts(self.ema_params[i])
            for name, ema_ckpt in ema_ckpts.items():
                ema_ckpt = {k: v.cpu() for k, v in ema_ckpt.items()}  # Move to CPU for saving
                if non_blocking:
                    threading.Thread(
                        target=torch.save,
                        args=(
                            ema_ckpt,
                            os.path.join(self.output_dir, "ckpts", f"{name}_ema{ema_rate}_step{self.step:07d}.pt"),
                        ),
                    ).start()
                else:
                    torch.save(
                        ema_ckpt, os.path.join(self.output_dir, "ckpts", f"{name}_ema{ema_rate}_step{self.step:07d}.pt")
                    )

        misc_ckpt = {
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
            "data_sampler": self.data_sampler.state_dict(),
        }
        if self.mix_precision_mode == "amp" and self.mix_precision_dtype == torch.float16:
            misc_ckpt["scaler"] = self.scaler.state_dict()
        elif self.mix_precision_mode == "inflat_all" and self.mix_precision_dtype == torch.float16:
            misc_ckpt["log_scale"] = self.log_scale
        if self.lr_scheduler_config is not None:
            misc_ckpt["lr_scheduler"] = self.lr_scheduler.state_dict()
        if self.elastic_controller_config is not None:
            misc_ckpt["elastic_controller"] = self.elastic_controller.state_dict()
        if self.grad_clip is not None and not isinstance(self.grad_clip, float):
            misc_ckpt["grad_clip"] = self.grad_clip.state_dict()
        if non_blocking:
            threading.Thread(
                target=torch.save,
                args=(misc_ckpt, os.path.join(self.output_dir, "ckpts", f"misc_step{self.step:07d}.pt")),
            ).start()
        else:
            torch.save(misc_ckpt, os.path.join(self.output_dir, "ckpts", f"misc_step{self.step:07d}.pt"))
        print(" Done.")

    def finetune_from(self, finetune_ckpt):
        """
        Finetune from a checkpoint.
        Should be called by all processes.
        """
        if self.is_master:
            print("\nFinetuning from:")
            for name, path in finetune_ckpt.items():
                print(f"  - {name}: {path}")

        model_ckpts = {}
        pretrained_trainable_param_names = set()
        for name, model in self.models.items():
            model_state_dict = model.state_dict()
            trainable_param_names = {param_name for param_name, p in model.named_parameters() if p.requires_grad}
            if name in finetune_ckpt:
                model_ckpt = self.load_finetune_model_ckpt(finetune_ckpt[name], name)

                filtered_ckpt = {}
                for k, v in model_ckpt.items():
                    if k not in model_state_dict:
                        if self.is_master:
                            print(f"Warning: {k} not in model, skipped")
                        continue
                    if v.shape != model_state_dict[k].shape:
                        if self.is_master:
                            print(f"Warning: shape mismatch {k}, skipped")
                        continue
                    filtered_ckpt[k] = v
                    if k in trainable_param_names:
                        pretrained_trainable_param_names.add(f"{name}.{k}")
                model.load_state_dict(filtered_ckpt, strict=False)
                model_ckpts[name] = model.state_dict()
            else:
                if self.is_master:
                    print(f"Warning: {name} not found in finetune_ckpt, skipped.")
                model_ckpts[name] = model_state_dict

        self.pretrained_trainable_param_names = pretrained_trainable_param_names
        self._state_dicts_to_master_params(self.master_params, model_ckpts)

        if self.is_master:
            for name, model in self.models.items():
                total = sum(p.numel() for p in model.parameters())
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print("AFTER loading pretrained checkpoint:")
                print(f"[{name}] total={total:,} trainable={trainable:,} ({trainable/total:.4%})")

            for i, ema_rate in enumerate(self.ema_rate):
                self._state_dicts_to_master_params(self.ema_params[i], model_ckpts)
        del model_ckpts

        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            print("Done.")

        if self.world_size > 1:
            self.check_ddp()

    @abstractmethod
    def run_snapshot(self, dataloader, num_samples, batch_size=4, verbose=False, **kwargs):
        """
        Run a snapshot of the model.
        """
        pass

    @torch.no_grad()
    def visualize_sample(self, sample):
        """
        Convert a sample to an image.
        """
        if hasattr(self.dataset, "visualize_sample"):
            return self.dataset.visualize_sample(sample)
        else:
            return sample

    @torch.no_grad()
    def snapshot_dataset(self, num_samples=100, batch_size=4):
        """
        Sample images from the dataset.
        """
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=True,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, "collate_fn") else None,
        )
        save_cfg = {}
        for i in range(0, num_samples, batch_size):
            data = next(iter(dataloader))
            data = {k: v[: min(num_samples - i, batch_size)] for k, v in data.items()}
            data = recursive_to_device(data, self.device)
            vis = self.visualize_sample(data)
            if isinstance(vis, dict):
                for k, v in vis.items():
                    if f"dataset_{k}" not in save_cfg:
                        save_cfg[f"dataset_{k}"] = []
                    save_cfg[f"dataset_{k}"].append(v)
            else:
                if "dataset" not in save_cfg:
                    save_cfg["dataset"] = []
                save_cfg["dataset"].append(vis)
        for name, image in save_cfg.items():
            utils.save_image(
                torch.cat(image, dim=0),
                os.path.join(self.output_dir, "samples", f"{name}.jpg"),
                nrow=int(np.sqrt(num_samples)),
                normalize=True,
                value_range=self.dataset.value_range,
            )

    @torch.no_grad()
    def get_snapshot_dataloader(self, split="train", batch_size=4):
        if split == "train":
            dataset = self.dataset
        elif split == "val":
            if self.val_dataset is None:
                return None
            dataset = self.val_dataset
        else:
            raise ValueError(f"Unknown split: {split}")

        return torch.utils.data.DataLoader(
            copy.deepcopy(dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=dataset.collate_fn if hasattr(dataset, "collate_fn") else None,
        )

    @torch.no_grad()
    def snapshot(self, suffix=None, num_samples=4, batch_size=4, verbose=False):
        """
        Sample images from the model.
        NOTE: This function should be called by all processes.
        """
        if self.is_master:
            print(f"\nSampling {num_samples} images...", end="")

        if suffix is None:
            suffix = f"step{self.step:07d}"

        # Assign tasks
        num_samples_per_process = int(np.ceil(num_samples / self.world_size))
        amp_context = (
            partial(torch.autocast, device_type="cuda", dtype=self.mix_precision_dtype)
            if self.mix_precision_mode == "amp"
            else nullcontext
        )

        samples = {}
        with amp_context():
            # ---- train snapshot ----
            train_loader = self.get_snapshot_dataloader("train", batch_size)
            train_samples = self.run_snapshot(
                train_loader,
                num_samples_per_process,
                batch_size=batch_size,
                verbose=verbose,
            )
            for k, v in train_samples.items():
                samples[f"train_{k}"] = v

            torch.cuda.empty_cache()

            # ---- val snapshot ----
            if self.val_dataset is not None:
                val_loader = self.get_snapshot_dataloader("val", batch_size)
                val_samples = self.run_snapshot(
                    val_loader,
                    num_samples_per_process,
                    batch_size=batch_size,
                    verbose=verbose,
                )
                for k, v in val_samples.items():
                    samples[f"val_{k}"] = v
            torch.cuda.empty_cache()

        # Preprocess images
        for key in list(samples.keys()):
            if samples[key]["type"] == "sample":
                vis = self.visualize_sample(samples[key]["value"])
                if isinstance(vis, dict):
                    for k, v in vis.items():
                        samples[f"{key}_{k}"] = {"value": v, "type": "image"}
                    del samples[key]
                else:
                    samples[key] = {"value": vis, "type": "image"}

        # Gather results
        if self.world_size > 1:
            backend = dist.get_backend() if dist.is_initialized() else None
            for key in samples.keys():
                value = samples[key]["value"].contiguous()
                original_device = value.device

                # Match tensor device to collective backend requirements:
                # - NCCL requires CUDA tensors.
                # - Gloo collectives are CPU-oriented here.
                if backend == "nccl" and value.device.type != "cuda":
                    value = value.to(self.device, non_blocking=True)
                elif backend != "nccl" and value.device.type == "cuda":
                    value = value.cpu()

                if self.is_master:
                    all_images = [torch.empty_like(value) for _ in range(self.world_size)]
                else:
                    all_images = []
                dist.gather(value, all_images, dst=0)
                if self.is_master:
                    gathered = torch.cat(all_images, dim=0)[:num_samples]
                    if original_device.type == "cpu" and gathered.device.type != "cpu":
                        gathered = gathered.cpu()
                    elif original_device.type == "cuda" and gathered.device.type != "cuda":
                        gathered = gathered.to(original_device, non_blocking=True)
                    samples[key]["value"] = gathered

        # Save images
        if self.is_master:
            os.makedirs(os.path.join(self.output_dir, "samples", suffix), exist_ok=True)
            for key in samples.keys():
                if samples[key]["type"] == "image":
                    utils.save_image(
                        samples[key]["value"],
                        os.path.join(self.output_dir, "samples", suffix, f"{key}_{suffix}.jpg"),
                        nrow=int(np.sqrt(num_samples)),
                        normalize=True,
                        value_range=self.dataset.value_range,
                    )
                elif samples[key]["type"] == "number":
                    min = samples[key]["value"].min()
                    max = samples[key]["value"].max()
                    images = (samples[key]["value"] - min) / (max - min)
                    images = utils.make_grid(
                        images,
                        nrow=int(np.sqrt(num_samples)),
                        normalize=False,
                    )
                    save_image_with_notes(
                        images,
                        os.path.join(self.output_dir, "samples", suffix, f"{key}_{suffix}.jpg"),
                        notes=f"{key} min: {min}, max: {max}",
                    )

        if self.is_master:
            print(" Snapshot Done.")

    def update_ema(self):
        """
        Update exponential moving average.
        Should only be called by the rank 0 process.
        """
        assert self.is_master, "update_ema() should be called only by the rank 0 process."
        for i, ema_rate in enumerate(self.ema_rate):
            for master_param, ema_param in zip(self.master_params, self.ema_params[i]):
                ema_param.detach().mul_(ema_rate).add_(master_param, alpha=1.0 - ema_rate)

    def check_ddp(self):
        """
        Check if DDP is working properly.
        Should be called by all process.
        """
        if self.is_master:
            print("\nPerforming DDP check...")

        if self.is_master:
            print("Checking if parameters are consistent across processes...")
        dist.barrier()
        try:
            for p in self.master_params:
                # split to avoid OOM
                for i in range(0, p.numel(), 10000000):
                    sub_size = min(10000000, p.numel() - i)
                    sub_p = p.detach().view(-1)[i : i + sub_size]
                    # gather from all processes
                    sub_p_gather = [torch.empty_like(sub_p) for _ in range(self.world_size)]
                    dist.all_gather(sub_p_gather, sub_p)
                    # check if equal
                    assert all(
                        [torch.equal(sub_p, sub_p_gather[i]) for i in range(self.world_size)]
                    ), "parameters are not consistent across processes"
        except AssertionError as e:
            if self.is_master:
                print(f"\n\033[91mError: {e}\033[0m")
                print("DDP check failed.")
            raise e

        dist.barrier()
        if self.is_master:
            print("Done.")

    @abstractmethod
    def training_losses(**mb_data):
        """
        Compute training losses.
        """
        pass

    @abstractmethod
    def validation_losses(**mb_data):
        """
        Compute training losses.
        """
        pass

    def load_data(self):
        """
        Load data.
        """
        if self.prefetch_data:
            if self._data_prefetched is None:
                self._data_prefetched = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
            data = self._data_prefetched
            self._data_prefetched = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
        else:
            data = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)

        # if the data is a dict, we need to split it into multiple dicts with batch_size_per_gpu
        if isinstance(data, dict):
            if self.batch_split == 1:
                data_list = [data]
            else:
                batch_size = list(data.values())[0].shape[0]
                data_list = [
                    {
                        k: v[i * batch_size // self.batch_split : (i + 1) * batch_size // self.batch_split]
                        for k, v in data.items()
                    }
                    for i in range(self.batch_split)
                ]
        elif isinstance(data, list):
            data_list = data
        else:
            raise ValueError("Data must be a dict or a list of dicts.")

        return data_list

    def run_step(self, data_list):
        """
        Run a training step.
        """
        step_log = {"loss": {}, "status": {}}
        amp_context = (
            partial(torch.autocast, device_type="cuda", dtype=self.mix_precision_dtype)
            if self.mix_precision_mode == "amp"
            else nullcontext
        )
        elastic_controller_context = (
            self.elastic_controller.record if self.elastic_controller_config is not None else nullcontext
        )

        # Train
        losses = []
        statuses = []
        elastic_controller_logs = []
        zero_grad(self.model_params)
        for i, mb_data in enumerate(data_list):
            ## sync at the end of each batch split
            sync_contexts = (
                [self.training_models[name].no_sync for name in self.training_models]
                if i != len(data_list) - 1 and self.world_size > 1
                else [nullcontext]
            )
            with nested_contexts(*sync_contexts), elastic_controller_context():
                with amp_context():
                    loss, status = self.training_losses(**mb_data)
                    l = loss["loss"] / len(data_list)
                ## backward
                if self.mix_precision_mode == "amp" and self.mix_precision_dtype == torch.float16:
                    self.scaler.scale(l).backward()
                elif self.mix_precision_mode == "inflat_all" and self.mix_precision_dtype == torch.float16:
                    scaled_l = l * (2**self.log_scale)
                    scaled_l.backward()
                else:
                    l.backward()
            ## log
            losses.append(dict_foreach(loss, lambda x: x.item() if isinstance(x, torch.Tensor) else x))
            statuses.append(dict_foreach(status, lambda x: x.item() if isinstance(x, torch.Tensor) else x))
            if self.elastic_controller_config is not None:
                elastic_controller_logs.append(self.elastic_controller.log())
        ## gradient clip
        if self.grad_clip is not None:
            if self.mix_precision_mode == "amp" and self.mix_precision_dtype == torch.float16:
                self.scaler.unscale_(self.optimizer)
            elif self.mix_precision_mode == "inflat_all":
                model_grads_to_master_grads(self.model_params, self.master_params)
                if self.mix_precision_dtype == torch.float16:
                    self.master_params[0].grad.mul_(1.0 / (2**self.log_scale))
            if isinstance(self.grad_clip, float):
                grad_norm = torch.nn.utils.clip_grad_norm_(self.master_params, self.grad_clip)
            else:
                grad_norm = self.grad_clip(self.master_params)
            if torch.isfinite(grad_norm):
                statuses[-1]["grad_norm"] = grad_norm.item()
        ## step
        if self.mix_precision_mode == "amp" and self.mix_precision_dtype == torch.float16:
            prev_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        elif self.mix_precision_mode == "inflat_all":
            if self.mix_precision_dtype == torch.float16:
                prev_scale = 2**self.log_scale
                if not any(not p.grad.isfinite().all() for p in self.model_params):
                    if self.grad_clip is None:
                        model_grads_to_master_grads(self.model_params, self.master_params)
                        self.master_params[0].grad.mul_(1.0 / (2**self.log_scale))
                    self.optimizer.step()
                    master_params_to_model_params(self.model_params, self.master_params)
                    self.log_scale += self.fp16_scale_growth
                else:
                    self.log_scale -= 1
            else:
                prev_scale = 1.0
                if self.grad_clip is None:
                    model_grads_to_master_grads(self.model_params, self.master_params)
                if not any(not p.grad.isfinite().all() for p in self.master_params):
                    self.optimizer.step()
                    master_params_to_model_params(self.model_params, self.master_params)
                else:
                    print("\n\033[93mWarning: NaN detected in gradients. Skipping update.\033[0m")
        else:
            prev_scale = 1.0
            if not any(not p.grad.isfinite().all() for p in self.model_params):
                self.optimizer.step()
            else:
                print("\n\033[93mWarning: NaN detected in gradients. Skipping update.\033[0m")
        ## log/adjust learning rate
        # Always log current optimizer LR(s), even when no scheduler is configured.
        current_lrs = [group["lr"] for group in self.optimizer.param_groups]
        statuses[-1]["lr"] = current_lrs[0]
        statuses[-1]["lr_groups"] = {
            self.optimizer.param_groups[i].get("name", f"group_{i}"): lr for i, lr in enumerate(current_lrs)
        }
        if self.lr_scheduler_config is not None:
            self.lr_scheduler.step()

        # Logs
        step_log["loss"] = dict_reduce(losses, lambda x: np.mean(x))
        step_log["status"] = dict_reduce(
            statuses, lambda x: np.mean(x), special_func={"min": lambda x: np.min(x), "max": lambda x: np.max(x)}
        )
        if self.elastic_controller_config is not None:
            step_log["elastic"] = dict_reduce(elastic_controller_logs, lambda x: np.mean(x))
        if self.grad_clip is not None:
            step_log["grad_clip"] = self.grad_clip if isinstance(self.grad_clip, float) else self.grad_clip.log()

        # Check grad and norm of each param
        if self.log_param_stats:
            param_norms = {}
            param_grads = {}
            for model_name, model in self.models.items():
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        param_norms[f"{model_name}.{name}"] = param.norm().item()
                        if param.grad is not None and torch.isfinite(param.grad).all():
                            param_grads[f"{model_name}.{name}"] = param.grad.norm().item() / prev_scale
            step_log["param_norms"] = param_norms
            step_log["param_grads"] = param_grads

        # Update exponential moving average
        if self.is_master:
            self.update_ema()
        step_log["vram"] = self.get_vram_stats()

        return step_log

    def run_val_step(self):
        if self.val_dataloader is None:
            return None

        for model in self.models.values():
            model.eval()

        amp_context = (
            partial(torch.autocast, device_type="cuda", dtype=self.mix_precision_dtype)
            if self.mix_precision_mode == "amp"
            else nullcontext
        )

        val_losses = []
        with torch.no_grad():
            for data in self.val_dataloader:
                data = recursive_to_device(data, self.device, non_blocking=True)

                with amp_context():
                    val_loss, _ = self.validation_losses(**data)

                val_losses.append(dict_foreach(val_loss, lambda x: x.item() if isinstance(x, torch.Tensor) else x))

        for model in self.models.values():
            model.train()

        val_log = dict_reduce(val_losses, lambda x: np.mean(x))
        return val_log

    def save_logs(self):
        log_str = "\n".join([f"{step}: {json.dumps(dict_foreach(log, lambda x: float(x)))}" for step, log in self.log])
        with open(os.path.join(self.output_dir, "log.txt"), "a") as log_file:
            log_file.write(log_str + "\n")

        # show with mlflow
        log_show = [l for _, l in self.log if not dict_any(l, lambda x: np.isnan(x))]
        log_show = dict_reduce(log_show, lambda x: np.mean(x))
        log_show = dict_flatten(log_show, sep="/")
        for key, value in log_show.items():
            self.writer.add_scalar(key, value, self.step)
        self.log = []

    def get_vram_stats(self, reset_peak=False):
        device = torch.device("cuda")

        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        max_allocated = torch.cuda.max_memory_allocated(device)

        stats = torch.tensor(
            [allocated, reserved, max_allocated],
            device=device,
            dtype=torch.float64,
        )

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.MAX)

        if reset_peak:
            torch.cuda.reset_peak_memory_stats(device)

        return {
            "alloc_mb": stats[0].item() / 1024**2,
            "reserved_mb": stats[1].item() / 1024**2,
            "peak_mb": stats[2].item() / 1024**2,
        }

    def check_abort(self):
        """
        Check if training should be aborted due to certain conditions.
        """
        # 1. If log_scale in inflat_all mode is less than 0
        if self.mix_precision_dtype == torch.float16 and self.mix_precision_mode == "inflat_all" and self.log_scale < 0:
            if self.is_master:
                print("\n\n\033[91m")
                print(f"ABORT: log_scale in inflat_all mode is less than 0 at step {self.step}.")
                print("This indicates that the model is diverging. You should look into the model and the data.")
                print("\033[0m")
                self.save(non_blocking=False)
                self.save_logs()
            if self.world_size > 1:
                dist.barrier()
            raise ValueError("ABORT: log_scale in inflat_all mode is less than 0.")

    def run(self):
        """
        Run training.
        """
        if self.is_master:
            print("\nStarting training...")
            self.snapshot_dataset(num_samples=25, batch_size=self.snapshot_batch_size)
            torch.cuda.empty_cache()
            print("Snapshot Dataset done")
        if self.step == 0:
            print("Starting init snapshot")
            self.snapshot(suffix="init", batch_size=self.snapshot_batch_size)
            torch.cuda.empty_cache()
        else:  # resume
            self.snapshot(suffix=f"resume_step{self.step:07d}", batch_size=self.snapshot_batch_size)

        time_last_print = 0.0
        time_elapsed = 0.0
        while self.step < self.max_steps:
            time_start = time.time()
            torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())

            data_list = self.load_data()
            step_log = self.run_step(data_list)

            time_end = time.time()
            time_elapsed += time_end - time_start

            self.step += 1

            if self.val_dataloader is not None and self.step % self.i_val == 0:
                val_log = self.run_val_step()

            # Print progress
            if self.is_master and self.step % self.i_print == 0:
                speed = self.i_print / (time_elapsed - time_last_print) * 3600
                columns = [
                    f"Step: {self.step}/{self.max_steps} ({self.step / self.max_steps * 100:.2f}%)",
                    f"Elapsed: {time_elapsed / 3600:.2f} h",
                    f"Speed: {speed:.2f} steps/h",
                    f"ETA: {(self.max_steps - self.step) / speed:.2f} h",
                ]
                print(" | ".join([c.ljust(25) for c in columns]), flush=True)
                time_last_print = time_elapsed

            # Check ddp
            if (
                self.parallel_mode == "ddp"
                and self.world_size > 1
                and self.i_ddpcheck is not None
                and self.step % self.i_ddpcheck == 0
            ):
                self.check_ddp()

            # Sample images
            if self.step % self.i_sample == 0:
                self.snapshot()

            if self.is_master:
                self.log.append((self.step, {}))

                # Log time
                self.log[-1][1]["time"] = {
                    "sample": (time_end - time_start) / self.batch_size,  # accounts for multi gpu and batch splits!
                    "step": time_end - time_start,
                    "elapsed": time_elapsed,
                }

                # Log losses
                if step_log is not None:
                    self.log[-1][1].update(step_log)

                if self.val_dataloader is not None and self.step % self.i_val == 0 and val_log is not None:
                    self.log[-1][1]["val"] = val_log
                    for k, v in dict_flatten(val_log, sep="/").items():
                        self.writer.add_scalar(f"val/{k}", v, self.step)

                # Log scale
                if self.mix_precision_dtype == torch.float16:
                    if self.mix_precision_mode == "amp":
                        self.log[-1][1]["scale"] = self.scaler.get_scale()
                    elif self.mix_precision_mode == "inflat_all":
                        self.log[-1][1]["log_scale"] = self.log_scale

                # Save log
                if self.step % self.i_log == 0:
                    self.save_logs()

                # Save checkpoint
                if self.step % self.i_save == 0:
                    self.save()

            if self.world_size > 1:
                dist.barrier()

            # Check abort
            self.check_abort()

        self.snapshot(suffix="final", batch_size=self.snapshot_batch_size)
        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            self.writer.close()
            print("Training finished.")

    def profile(self, wait=2, warmup=3, active=5):
        """
        Profile the training loop.
        """
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(self.output_dir, "profile")),
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for _ in range(wait + warmup + active):
                self.run_step()
                prof.step()
