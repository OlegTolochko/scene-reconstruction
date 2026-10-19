import numpy as np
import torch

from ....modules import sparse as sp
from ....pipelines import samplers


class ClassifierFreeGuidanceMixin:
    def __init__(self, *args, p_uncond: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_uncond = p_uncond

    def get_cond(self, cond, neg_cond=None, **kwargs):
        """
        Get the conditioning data.
        """
        assert neg_cond is not None, "neg_cond must be provided for classifier-free guidance"

        if self.p_uncond > 0:
            # randomly drop the class label
            def get_batch_size(cond):
                if isinstance(cond, torch.Tensor):
                    return cond.shape[0]
                elif isinstance(cond, sp.SparseTensor):
                    return cond.shape[0]
                elif isinstance(cond, list):
                    return len(cond)
                else:
                    raise ValueError(f"Unsupported type of cond: {type(cond)}")

            ref_cond = cond if not isinstance(cond, dict) else cond[list(cond.keys())[0]]
            B = get_batch_size(ref_cond)

            def select(cond, neg_cond, mask):
                if isinstance(cond, torch.Tensor):
                    mask_t = torch.tensor(mask, device=cond.device).reshape(-1, *[1] * (cond.ndim - 1))
                    return torch.where(mask_t, neg_cond, cond)
                elif isinstance(cond, sp.SparseTensor):
                    new_feats = cond.feats.clone()
                    for b, (should_zero, layout) in enumerate(zip(mask, cond.layout)):
                        if should_zero:
                            new_feats[layout] = 0.0
                    return cond.replace(new_feats)
                elif isinstance(cond, list):
                    return [nc if m else c for c, nc, m in zip(cond, neg_cond, mask)]
                else:
                    raise ValueError(f"Unsupported type of cond: {type(cond)}")

            mask = list(np.random.rand(B) < self.p_uncond)
            if not isinstance(cond, dict):
                cond = select(cond, neg_cond, mask)
            else:
                cond = {key: select(cond[key], neg_cond[key], mask) for key in cond.keys()}

        return cond

    def get_inference_cond(self, cond, neg_cond=None, **kwargs):
        """
        Get the conditioning data for inference.
        """
        assert neg_cond is not None, "neg_cond must be provided for classifier-free guidance"
        return {"cond": cond, "neg_cond": neg_cond, **kwargs}

    def get_sampler(self, **kwargs) -> samplers.FlowEulerCfgSampler:
        """
        Get the sampler for the diffusion process.
        """
        return samplers.FlowEulerCfgSampler(self.sigma_min)
