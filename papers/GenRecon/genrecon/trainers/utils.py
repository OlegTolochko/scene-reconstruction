from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn

# FP16 utils
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


def str_to_dtype(dtype_str: str):
    return {
        "f16": torch.float16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "f32": torch.float32,
        "fp32": torch.float32,
        "float32": torch.float32,
    }[dtype_str]


def make_master_params(model_params):
    """
    Copy model parameters into a inflated tensor of full-precision parameters.
    """
    master_params = _flatten_dense_tensors([param.detach().float() for param in model_params])
    master_params = nn.Parameter(master_params)
    master_params.requires_grad = True
    return [master_params]


def unflatten_master_params(model_params, master_params):
    """
    Unflatten the master parameters to look like model_params.
    """
    return _unflatten_dense_tensors(master_params[0].detach(), model_params)


def model_params_to_master_params(model_params, master_params):
    """
    Copy the model parameter data into the master parameters.
    """
    master_params[0].detach().copy_(_flatten_dense_tensors([param.detach().float() for param in model_params]))


def master_params_to_model_params(model_params, master_params):
    """
    Copy the master parameter data back into the model parameters.
    """
    for param, master_param in zip(model_params, _unflatten_dense_tensors(master_params[0].detach(), model_params)):
        param.detach().copy_(master_param)


def model_grads_to_master_grads(model_params, master_params):
    """
    Copy the gradients from the model parameters into the master parameters
    from make_master_params().
    """
    master_params[0].grad = _flatten_dense_tensors([param.grad.data.detach().float() for param in model_params])


def zero_grad(model_params):
    for param in model_params:
        if param.grad is not None:
            if param.grad.grad_fn is not None:
                param.grad.detach_()
            else:
                param.grad.requires_grad_(False)
            param.grad.zero_()


def build_optimizer_param_groups(
    named_params: Iterable[Tuple[str, nn.Parameter]],
    lr: Union[float, Dict[str, float]],
    weight_decay: Union[float, Dict[str, float]],
    pretrained_param_names: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Build optimizer parameter groups with dedicated LoRA, new-parameter, and old-parameter settings.

    Grouping logic for trainable parameters:
    - `lora`: parameter names containing `lora_`.
    - `old_*`: parameters restored from the finetune checkpoint.
    - `new_*`: parameters not restored from the finetune checkpoint, e.g. newly added projections.
    - `*_weights`: matrix parameters (ndim >= 2) excluding norms.
    - `*_no_decay`: bias/norm/scalar params (forced weight_decay=0).
    """
    pretrained_param_names = pretrained_param_names or set()

    group_lora: List[nn.Parameter] = []
    group_old_weights: List[nn.Parameter] = []
    group_old_no_decay: List[nn.Parameter] = []
    group_new_weights: List[nn.Parameter] = []
    group_new_no_decay: List[nn.Parameter] = []

    for name, param in named_params:
        if not param.requires_grad:
            continue

        lname = name.lower()
        is_lora = "lora_" in lname
        is_bias = lname.endswith(".bias") or lname == "bias"
        is_norm = "norm" in lname
        is_scalar = param.ndim == 0 or param.numel() == 1

        if is_lora:
            group_lora.append(param)
            continue

        is_old = name in pretrained_param_names
        if is_bias or is_norm or is_scalar or param.ndim < 2:
            if is_old:
                group_old_no_decay.append(param)
            else:
                group_new_no_decay.append(param)
        else:
            if is_old:
                group_old_weights.append(param)
            else:
                group_new_weights.append(param)

    def _get_group_value(
        cfg: Union[float, Dict[str, float]],
        aliases: Tuple[str, ...],
        *,
        required: bool,
        name: str,
        default: Optional[float] = None,
    ) -> Optional[float]:
        if not isinstance(cfg, dict):
            return float(cfg)
        for alias in aliases:
            if alias in cfg:
                return float(cfg[alias])
        if default is not None:
            return float(default)
        if required:
            alias_str = ", ".join(f"'{alias}'" for alias in aliases)
            raise KeyError(f"optimizer args for {name} must contain one of: {alias_str}.")
        return None

    has_lora = len(group_lora) > 0
    has_old = len(group_old_weights) > 0 or len(group_old_no_decay) > 0
    has_new = len(group_new_weights) > 0 or len(group_new_no_decay) > 0

    lora_lr = _get_group_value(lr, ("lora",), required=has_lora, name="lora lr")
    old_lr = _get_group_value(lr, ("old",), required=has_old, name="old lr")
    new_lr = _get_group_value(lr, ("new",), required=has_new, name="new lr")

    lora_wd = _get_group_value(
        weight_decay,
        ("lora",),
        required=False,
        name="lora weight_decay",
        default=0.0,
    )
    old_weights_wd = _get_group_value(
        weight_decay,
        ("old", "old_weights"),
        required=has_old and len(group_old_weights) > 0,
        name="old weight_decay",
    )
    new_weights_wd = _get_group_value(
        weight_decay,
        ("new", "new_weights"),
        required=has_new and len(group_new_weights) > 0,
        name="new weight_decay",
    )

    optimizer_param_groups: List[Dict[str, Any]] = []
    if has_lora:
        optimizer_param_groups.append(
            {
                "name": "lora",
                "params": group_lora,
                "lr": lora_lr,
                "weight_decay": lora_wd,
            }
        )
    if len(group_old_weights) > 0:
        optimizer_param_groups.append(
            {
                "name": "old_weights",
                "params": group_old_weights,
                "lr": old_lr,
                "weight_decay": old_weights_wd,
            }
        )
    if len(group_old_no_decay) > 0:
        optimizer_param_groups.append(
            {
                "name": "old_no_decay",
                "params": group_old_no_decay,
                "lr": old_lr,
                "weight_decay": 0.0,
            }
        )
    if len(group_new_weights) > 0:
        optimizer_param_groups.append(
            {
                "name": "new_weights",
                "params": group_new_weights,
                "lr": new_lr,
                "weight_decay": new_weights_wd,
            }
        )
    if len(group_new_no_decay) > 0:
        optimizer_param_groups.append(
            {
                "name": "new_no_decay",
                "params": group_new_no_decay,
                "lr": new_lr,
                "weight_decay": 0.0,
            }
        )

    if len(optimizer_param_groups) == 0:
        raise ValueError("No trainable parameters found for optimizer param-group construction.")

    group_stats = {
        "lora": sum(p.numel() for p in group_lora),
        "old_weights": sum(p.numel() for p in group_old_weights),
        "old_no_decay": sum(p.numel() for p in group_old_no_decay),
        "new_weights": sum(p.numel() for p in group_new_weights),
        "new_no_decay": sum(p.numel() for p in group_new_no_decay),
    }
    group_stats["total"] = sum(group_stats.values())

    return optimizer_param_groups, group_stats


# LR Schedulers
from torch.optim.lr_scheduler import LambdaLR


class LinearWarmupLRScheduler(LambdaLR):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(LinearWarmupLRScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, current_step):
        if current_step < self.warmup_steps:
            return float(current_step + 1) / self.warmup_steps
        return 1.0
