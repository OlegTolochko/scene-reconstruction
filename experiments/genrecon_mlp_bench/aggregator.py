"""GenRecon's multi-view aggregator of the sparse-structure stage as a class whose steps a variant can override,
the registry of variants, and the depth cues every variant gets.

GenRecon is not modified: build() wraps the aggregator of a loaded GenRecon model. The wrapper holds GenRecon's
two MLPs under their own names, so GenRecon's checkpoint keys stay valid and a variant only adds keys.

The cue computation is the one of experiments/genrecon_depth_cue_finetuning (feat/depth-cue-finetuning). Only
torch is imported here.
"""

import hashlib
import inspect

import torch
import torch.nn.functional as F
from torch import nn

# The channels of `cues` [B, N, V, 4], per (view, voxel); all 0 where the view has no valid depth at the voxel.
BEHIND = 0  # clip(delta / voxel size, -8, 8), delta = voxel depth - observed depth: < 0 free space, > 0 occluded
BEHIND_REL = 1  # clip(delta / observed depth, -0.5, 0.5): depth error grows with the distance
CONFIDENCE = 2  # the depth model's confidence, standardised over the training rooms and clipped to [-3, 3]
KNOWN = 3  # 1 where the voxel projects onto a pixel with a valid depth
N_CUES = 4

VARIANTS: dict[str, type] = {}


def register(name: str):
    """Class decorator: makes the aggregator available as `sbatch run.sbatch <name>`."""

    def add(cls):
        assert name not in VARIANTS, f"two variants are called {name!r}"
        assert name.replace("_", "").isalnum(), f"variant names are letters, digits and _: {name!r}"
        VARIANTS[name] = cls
        cls.variant_name = name
        return cls

    return add


def zero_linear(n_in: int, n_out: int, bias: bool = False) -> nn.Linear:
    """A linear layer that outputs 0 until it is trained, so a variant built on it starts exactly at GenRecon."""
    layer = nn.Linear(n_in, n_out, bias=bias)
    nn.init.zeros_(layer.weight)
    if bias:
        nn.init.zeros_(layer.bias)
    return layer


@register("genrecon")
class Aggregator(nn.Module):
    """GenRecon's aggregator, unchanged: the baseline, and the class a variant subclasses.

    For every voxel it merges the DINOv3 features of the views. feats [B, N, V, D] holds the feature each of the
    N views has at each of the V voxels, mask [B, N, V] whether the view sees the voxel. The steps:

        statistics      mean and variance of the features over the views
        view_features   feature_mlp on [feature, mean, variance], per view
        view_logits     weight_mlp on the same input: one logit per view, softmaxed over the views
        output          mean + sum over the views of weight * view feature           [B, V, D]

    A variant overrides one or more of the three steps (or forward itself) and may add layers in __init__. Each
    step gets `cues` [B, N, V, 4], the depth cues listed at the top of this file, in the dtype of the weights.
    New layers should start at zero (zero_linear), so training starts exactly at GenRecon; build() checks it.
    """

    # Prefixes of this module's parameter names that train, e.g. ("cue_",). None: the whole aggregator trains.
    train_only = None

    def __init__(self, base: nn.Module):
        super().__init__()
        unsupported = [f for f in ("use_camera", "use_conv_net", "use_self_attention") if getattr(base, f, False)]
        assert not unsupported and getattr(base, "refinement_factor", 1) == 1, \
            f"the bench covers GenRecon's sparse-structure aggregator only, not one with {unsupported or 'refinement'}"
        self.channels = base.channels
        self.feature_mlp, self.weight_mlp = base.feature_mlp, base.weight_mlp  # GenRecon's weights, same names
        self.cues = None  # set before every forward by whoever projects the features (train.py, inject.py)

    @staticmethod
    def first_layer(mlp: nn.Sequential, feats, mu, var):
        """mlp[0] on [feats, mu, var] without building the concatenation, as GenRecon does it. [B, N, V, D]."""
        W, D = mlp[0].weight, feats.shape[-1]
        return F.linear(feats, W[:, :D]) + F.linear(mu, W[:, D : 2 * D]) + F.linear(var, W[:, 2 * D : 3 * D]) + mlp[0].bias

    @staticmethod
    def rest(mlp: nn.Sequential, h):
        """The layers of mlp after the first one."""
        for i in range(1, len(mlp)):
            h = mlp[i](h)
        return h

    def statistics(self, feats, mask, cues):
        """Mean and variance over the views that see the voxel, [B, 1, V, D] each."""
        mask_f = mask.unsqueeze(-1)
        n = mask_f.sum(dim=1, keepdim=True).clamp_min(1)
        masked = feats * mask_f
        mu = masked.sum(dim=1, keepdim=True) / n
        var = ((masked * feats).sum(dim=1, keepdim=True) / n - mu * mu).clamp_min(0)
        return mu, var

    def view_features(self, feats, mu, var, mask, cues):
        """What each view adds to the mean, [B, N, V, D]."""
        return self.rest(self.feature_mlp, self.first_layer(self.feature_mlp, feats, mu, var))

    def view_logits(self, feats, mu, var, mask, cues):
        """One logit per view, [B, N, V, 1]; forward masks them and takes the softmax over the views."""
        return self.rest(self.weight_mlp, self.first_layer(self.weight_mlp, feats, mu, var))

    def forward(self, feats, mask, camera_emb=None):
        B, N, V, D = feats.shape
        assert D == self.channels and mask.shape == (B, N, V) and mask.dtype == torch.bool
        cues = self.cues
        if cues is None:  # no depth given: every cue 0, as where the depth is unknown
            cues = feats.new_zeros(B, N, V, N_CUES)
        assert cues.shape == (B, N, V, N_CUES), f"cues {tuple(cues.shape)} do not match the features {(B, N, V)}"
        cues = cues.to(self.feature_mlp[0].weight.dtype)

        mu, var = self.statistics(feats, mask, cues)
        mu, var = mu.to(feats.dtype), var.to(feats.dtype)  # the MLPs take one dtype (bf16 at inference)
        view = self.view_features(feats, mu, var, mask, cues)
        logits = self.view_logits(feats, mu, var, mask, cues)

        mask_f = mask.unsqueeze(-1)
        w = torch.softmax(logits.masked_fill(~mask_f, -1e9), dim=1)
        w = torch.where((~mask).all(dim=1, keepdim=True).unsqueeze(-1), torch.zeros_like(w), w)
        return (mu + (view * w).sum(dim=1, keepdim=True)).squeeze(1)


@torch.no_grad()
def start_difference(base: nn.Module, agg: Aggregator) -> float:
    """Largest difference between the variant and GenRecon's aggregator on random inputs and cues, relative to the
    largest output. 0 for a variant that starts exactly at GenRecon."""
    w = base.feature_mlp[0].weight
    g = torch.Generator().manual_seed(0)  # its own generator: the global random streams stay as they are
    feats = torch.randn(1, 4, 9, base.channels, generator=g).to(w)
    mask = torch.rand(1, 4, 9, generator=g) < 0.7
    mask[:, :, 0] = False  # a voxel no view sees
    mask = mask.to(w.device)
    feats = feats * mask.unsqueeze(-1)  # as the projection delivers them
    cues = torch.randn(1, 4, 9, N_CUES, generator=g).to(w)
    training = base.training, agg.training
    base.eval(), agg.eval()
    agg.cues = cues
    try:
        ref, out = base(feats.clone(), mask, None), agg(feats.clone(), mask, None)
    finally:
        agg.cues = None
        base.train(training[0]), agg.train(training[1])
    return float((out.float() - ref.float()).abs().max() / ref.float().abs().max().clamp_min(1e-12))


def build(name: str, base: nn.Module) -> Aggregator:
    """The variant `name` around GenRecon's aggregator `base`, on its device and in its dtype."""
    if name not in VARIANTS:
        raise SystemExit(f"unknown variant {name!r}; variants.py has: {', '.join(VARIANTS)}")
    w = base.feature_mlp[0].weight
    agg = VARIANTS[name](base).to(device=w.device, dtype=w.dtype)
    agg.start_difference = start_difference(base, agg)
    print(f"variant {name}: " + ("starts exactly at GenRecon" if agg.start_difference == 0 else
                                 f"differs from GenRecon at initialisation by {agg.start_difference:.2e} (relative)"))
    return agg


def set_trainable(model: nn.Module) -> int:
    """Only the aggregator trains (GenRecon's transformer and LoRA stay frozen), and of it only train_only if the
    variant sets that. Returns the number of trained weights."""
    only = model.aggregator.train_only
    count = 0
    for name, p in model.named_parameters():
        inside = name.removeprefix("aggregator.")
        p.requires_grad_(name.startswith("aggregator.") and (only is None or inside.startswith(tuple(only))))
        count += p.numel() if p.requires_grad else 0
    assert count > 0, f"variant {model.aggregator.variant_name}: no parameter matches train_only = {only}"
    return count


def stand_in(channels: int = 16) -> nn.Module:
    """A small module with the layout of GenRecon's aggregator, to try a variant without GenRecon or a GPU."""
    base = nn.Module()
    base.channels = channels
    relu = lambda: nn.ReLU(inplace=True)  # noqa: E731
    base.feature_mlp = nn.Sequential(nn.Linear(3 * channels, channels), relu(), nn.Linear(channels, channels), relu(),
                                     nn.Linear(channels, channels))
    base.weight_mlp = nn.Sequential(nn.Linear(3 * channels, channels), relu(), nn.Linear(channels, 1))
    return base


def preflight(name: str) -> str:
    """Runs a variant on a small stand-in for GenRecon's aggregator the three ways it will be run: in float32,
    under bfloat16 autocast with a backward pass (training), and with bfloat16 weights (reconstruction). Raises
    on the first error, so a mistake in a variant shows up in seconds and not after the baseline has trained.
    Returns a summary line."""
    if name not in VARIANTS:
        raise SystemExit(f"unknown variant {name!r}; variants.py has: {', '.join(VARIANTS)}")
    torch.manual_seed(0)
    base = stand_in()
    agg = VARIANTS[name](base)
    difference = start_difference(Aggregator(base), agg)
    C = base.channels
    mask = torch.rand(2, 3, 5) < 0.7
    mask[:, :, 0] = False
    feats, cues = torch.randn(2, 3, 5, C) * mask.unsqueeze(-1), torch.randn(2, 3, 5, N_CUES)

    def run(module, feats):
        module.cues = cues
        out = module(feats, mask)
        assert out.shape == (2, 5, C), f"the output is {tuple(out.shape)}, expected [B, V, D] = {(2, 5, C)}"
        assert torch.isfinite(out.float()).all(), "the output has NaN or infinite values"
        return out

    run(agg, feats.clone())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        run(agg, feats.clone()).float().square().mean().backward()
    trained = {n: p for n, p in agg.named_parameters() if agg.train_only is None or n.startswith(tuple(agg.train_only))}
    assert trained, f"no parameter matches train_only = {agg.train_only}"
    dead = [n for n, p in trained.items() if p.grad is None]
    assert not dead, f"these parameters are to be trained but do not affect the output: {dead}"
    with torch.no_grad():
        run(VARIANTS[name](stand_in().to(torch.bfloat16)).to(torch.bfloat16), feats.to(torch.bfloat16))

    new = [n for n, _ in agg.named_parameters() if not n.startswith(("feature_mlp.", "weight_mlp."))]
    start = "starts exactly at GenRecon" if difference == 0 else f"differs from GenRecon at the start by {difference:.1e} (relative)"
    trains = "the whole aggregator" if agg.train_only is None else ", ".join(trained)
    return f"{name}: ok, {start}; new parameters: {', '.join(new) or 'none'}; trains: {trains}"


def source(name: str) -> str:
    """The code of a variant: its class and the variant classes it builds on (Aggregator itself only for the
    baseline)."""
    classes = [c for c in VARIANTS[name].__mro__ if issubclass(c, Aggregator)]
    classes = classes[:-1] or classes
    return "\n\n".join(inspect.getsource(c) for c in reversed(classes))


def source_hash(name: str) -> str:
    return hashlib.sha256(source(name).encode()).hexdigest()[:16]


def compute_cues(
    points: torch.Tensor,  # [V, 3] voxel centres, in the frame of w2c
    w2c: torch.Tensor,  # [B, N, 4, 4] OpenCV world-to-camera of the depth maps
    K: torch.Tensor,  # [B, N, 3, 3] intrinsics normalised to the depth map (u, v in [0, 1))
    depth: torch.Tensor,  # [B, N, H, W] z-depth in the units of w2c; <= 0 or non-finite = unknown
    conf: torch.Tensor,  # [B, N, H, W]
    voxel_size: float,
    conf_mean: float = 0.0,
    conf_std: float = 1.0,
) -> torch.Tensor:
    """The cues [B, N, V, 4] in float32 (channels: BEHIND, BEHIND_REL, CONFIDENCE, KNOWN)."""
    B, N, H, W = depth.shape
    w2c, K, points = w2c.float(), K.float(), points.float()

    cam = torch.einsum("bnij,vj->bnvi", w2c[..., :3, :3], points) + w2c[..., None, :3, 3]  # [B, N, V, 3]
    z = cam[..., 2]
    front = z > 1e-6
    uv = torch.einsum("bnij,bnvj->bnvi", K, cam / torch.where(front, z, torch.ones_like(z))[..., None])
    u, v = uv[..., 0], uv[..., 1]
    inside = front & (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
    # Nearest pixel: interpolating would mix depths across occlusion edges.
    px = torch.floor(u * W).long().clamp(0, W - 1)
    py = torch.floor(v * H).long().clamp(0, H - 1)
    idx = (py * W + px).reshape(B, N, -1)
    D = depth.reshape(B, N, -1).float().gather(2, idx).reshape(z.shape)
    c = conf.reshape(B, N, -1).float().gather(2, idx).reshape(z.shape)

    known = inside & torch.isfinite(D) & (D > 0) & torch.isfinite(c)
    D = torch.where(known, D, torch.ones_like(D))
    delta = z - D
    cues = torch.stack(
        [
            (delta / voxel_size).clamp(-8.0, 8.0),
            (delta / D).clamp(-0.5, 0.5),
            ((c - conf_mean) / conf_std).clamp(-3.0, 3.0),
            torch.ones_like(z),
        ],
        dim=-1,
    )
    return torch.where(known[..., None], cues, torch.zeros_like(cues))
