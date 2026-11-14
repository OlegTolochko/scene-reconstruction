"""The aggregator variants of the MLP bench. To test an idea, add a class here, try it (seconds, no GPU) and
submit it by its name:

    uv run --with torch experiments/genrecon_mlp_bench/variants.py my_idea
    sbatch experiments/genrecon_mlp_bench/run.sbatch my_idea

A variant subclasses Aggregator (aggregator.py, which documents the steps and the depth cues) and overrides what
it changes. Rules of thumb:
  - create new layers in __init__ with zero_linear, so the variant starts exactly at GenRecon (the job prints
    whether it does);
  - index the cues by name: cues[..., BEHIND], cues[..., KNOWN], ...;
  - once a variant has been submitted, give a changed version a new name: its runs keep the old code's results.
"""

import torch

from aggregator import N_CUES, Aggregator, register, zero_linear


@register("depth_inputs")
class DepthInputs(Aggregator):
    """The depth cues as extra inputs of the first layer of both MLPs (variant A of genrecon_depth_cue_finetuning)."""

    def __init__(self, base):
        super().__init__(base)
        self.cue_feature = zero_linear(N_CUES, self.channels)
        self.cue_weight = zero_linear(N_CUES, self.channels)

    def view_features(self, feats, mu, var, mask, cues):
        h = self.first_layer(self.feature_mlp, feats, mu, var) + self.cue_feature(cues)
        return self.rest(self.feature_mlp, h)

    def view_logits(self, feats, mu, var, mask, cues):
        h = self.first_layer(self.weight_mlp, feats, mu, var) + self.cue_weight(cues)
        return self.rest(self.weight_mlp, h)


@register("depth_inputs_only")
class DepthInputsOnly(DepthInputs):
    """depth_inputs where only the two new layers train and GenRecon's weights stay as released: a gain then
    comes from the depth, not from more finetuning."""

    train_only = ("cue_",)


@register("visible_mean")
class VisibleMean(Aggregator):
    """A learned weight per view in the mean and variance, from the depth cues: a view that sees the voxel behind
    a surface can count less. GenRecon's MLPs get the cleaner statistics but no depth themselves."""

    def __init__(self, base):
        super().__init__(base)
        self.visibility = zero_linear(N_CUES, 1)

    def statistics(self, feats, mask, cues):
        v = 2 * torch.sigmoid(self.visibility(cues))  # 1 at the start, so the statistics start as GenRecon's
        v = v.to(feats.dtype) * mask.unsqueeze(-1)  # only the views that see the voxel count
        total = v.sum(dim=1, keepdim=True).clamp_min(1e-6)
        mu = (feats * v).sum(dim=1, keepdim=True) / total
        var = ((feats * feats * v).sum(dim=1, keepdim=True) / total - mu * mu).clamp_min(0)
        return mu, var


if __name__ == "__main__":  # python variants.py [NAME ...]: tries the named variants, or all of them
    import sys

    from aggregator import VARIANTS, preflight

    for name in sys.argv[1:] or list(VARIANTS):
        print(preflight(name))
