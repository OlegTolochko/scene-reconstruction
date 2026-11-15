# GenRecon with visibility-weighted aggregation

Does removing occluded views with depth improve GenRecon, without any training?

GenRecon projects every voxel into every view, samples a DINOv3 feature $\mathbf f_i$ there and aggregates
the features IBRNet-style: mean $\boldsymbol\mu$ and variance $\boldsymbol\sigma^2$ over the views, an MLP per
view on $[\mathbf f_i, \boldsymbol\mu, \boldsymbol\sigma^2]$, and a softmax over view logits $w_i$. Occlusion is
not handled. A voxel hidden behind an object in view $i$ gets the occluder's feature at full weight, which
contaminates $\boldsymbol\mu$, $\boldsymbol\sigma^2$ and the residual. This experiment computes a weight per
(view, voxel) from depth and uses it in the frozen aggregator. No trained weight changes, and neither do the
DiTs, chunking or voting.

$$v_i=\text{valid}_i\big[c_i\,g(\Delta_i)+1-c_i\big],\quad \Delta_i=z_i(\mathbf x)-D_i(\pi_i(\mathbf x)),\quad
g(\Delta)=\operatorname{sigmoid}\Big(\frac{\tau-\Delta}{s\tau}\Big),\quad \tau=\max(\tau_\text{vox}\,s_\text{vox},\ \rho D_i)$$

$$\boldsymbol\mu_v=\frac{\sum_i v_i\mathbf f_i}{\sum_i v_i},\quad
\boldsymbol\sigma_v^2=\frac{\sum_i v_i\mathbf f_i^2}{\sum_i v_i}-\boldsymbol\mu_v^2,\quad
\tilde w_i=w_i+\lambda\log\frac{v_i+\varepsilon}{1+\varepsilon}$$

and per voxel $\alpha=\min(1,\sum_i v_i/\kappa)$, $v\leftarrow\alpha v+(1-\alpha)\,\text{valid}$.

| Choice | Why |
| --- | --- |
| Only occluded views ($\Delta\gg0$) are removed, free-space views ($\Delta<0$) are kept | The frozen MLPs most likely recognise empty space by cross-view inconsistency. Dropping free-space views would make empty voxels look like surfaces. |
| Low confidence falls back to $v=1$ instead of scaling $v$ | Low confidence means we can't tell whether the voxel is occluded, so GenRecon's default is kept. Scaling by confidence would down-weight textureless surfaces whose DINO features are fine. |
| $\tau_\text{vox}=1.5$ voxels, $\rho=0.02$ | $1.5>\sqrt3/2$ covers the voxel discretisation. $\rho D$ covers depth error, which grows with distance. |
| The voxel centre only, nearest depth pixel | GenRecon samples the feature at the centre, so the weight describes that same sample. Interpolating depth would mix the two sides of an occlusion edge. |
| $\kappa$ fallback | A voxel hidden in (almost) every view must be filled in by the generative prior anyway, so it gets GenRecon's weights back. |
| $\log\frac{v+\varepsilon}{1+\varepsilon}$ instead of $\log(v+\varepsilon)$ | It is the same up to a constant shift, which the softmax ignores. It is exactly 0 at $v=1$, so $v=\text{valid}$ reproduces GenRecon bit for bit. |

$v=\text{valid}$ is GenRecon (its mean, variance and softmax already skip views the voxel projects outside
of). The method deviates from GenRecon only where there is confident evidence of occlusion.

### Which variance (`--var-mode`)

The first runs used $\boldsymbol\sigma_v^2$ (`weighted`) and did not move stage 1 towards the scan. One
explanation: removing the occluded views also removes their disagreement, so a voxel behind a surface gets a
small variance and looks like a surface to the frozen MLPs. Two alternatives keep the v-weighted mean
$\boldsymbol\mu_v$:

| `--var-mode` | Variance | Behind a surface |
| --- | --- | --- |
| `weighted` | $\sum_i v_i(\mathbf f_i-\boldsymbol\mu_v)^2/\sum_i v_i$ | shrinks: looks more like a surface |
| `original` | GenRecon's, over the valid views around $\boldsymbol\mu$ | unchanged |
| `around_vmean` | over the valid views around $\boldsymbol\mu_v$, $=\boldsymbol\sigma^2+(\boldsymbol\mu-\boldsymbol\mu_v)^2$ | grows by how far the weighting moved the mean: looks more empty |

`around_vmean` never lowers the variance. Its price is that a true surface that is partly occluded also gets
a larger variance. That fits the baseline, which has recall to spare (0.97 within 1 cell) but lacks
precision (0.75). A clean $\boldsymbol\mu_v$ with an inflated variance never occurred in training, so only a
run shows how strongly the MLPs react.

`variance_check.py` tests this before a run. It stops GenRecon after stage 1's conditioning, labels the
baseline's stage 1 cells with the scan (true: contains surface, false: 2 or more cells from it) and reports
how well each variance separates false from true cells (AUROC, over all cells and over cells with a
down-weighted view). A run is worth it if `around_vmean` separates better than `original`.

## Setting

This uses the unposed setting of `experiments/unposed_genrecon`: the cameras and depth of one model, so the
depth is consistent with the cameras GenRecon projects with, in the same units, without any alignment.
`--model` picks the model, as in `run_eval.py`:

| `--model` | Cameras and depth | Baseline | Outputs |
| --- | --- | --- | --- |
| `vggt_omega` (default) | VGGT-Omega, `vggt.npz` | `outputs/unposed_genrecon` | `outputs/genrecon_occlusion_weighting` |
| `solv3r` | Solv3R, `solv3r.npz` (depth 504 px on the long side) | `outputs/unposed_genrecon_solv3r` | `outputs/genrecon_occlusion_weighting_solv3r` |

Solv3R beats VGGT-Omega on unposed GenRecon, so it is the stronger baseline to improve on. Variants:

| Variant | Depth $D$, confidence $c$ | Output folder | Purpose |
| --- | --- | --- | --- |
| `depth` | the model's depth; $c$ ramps linearly from 0 at conf 2 to 1 at conf 5 | `vggt/`, `solv3r/` | the method |
| `gt` | the laser scan rendered at the same cameras, $c=1$ ($c=0$ where a ray hits nothing) | `gt/` | oracle: separates the mechanism from the cue quality |
| `identity` | none, $v=\text{valid}$ | `identity/` | must reproduce unposed GenRecon; checks the patched code path |

For `gt`, the cameras are placed in the scan with the similarity transform from their centres to the
ScanNet++ centres, as `run_eval.py` does for the mesh. The depth is divided by its scale. The cameras stay
the model's, so only the depth differs from `depth`. The `gt` cue therefore differs between the models.

The confidence ramp starts at convert.py's threshold (conf 2), which convert.py uses for both models. Check
`c_mean` in `check.json` before reading Solv3R results: if Solv3R's confidence has a different range, pass
`cues.py --conf-lo --conf-hi`.

Why this baseline matters even though we will finetune: it validates the cue pipeline, because frame, scale or
sign bugs show up here instead of being silently ignored by zero-initialised inputs. It is also the ablation
that finetuning has to beat ("would a hand-made visibility mask do the same?").

## Run

Requires the unposed GenRecon runs (`experiments/unposed_genrecon/run_eval.py [--model solv3r]`, for the
model's npz and the baseline meshes), the GenRecon env, and the scans from `scripts/download_scannetpp_eval.yml`.

```bash
uv run experiments/genrecon_occlusion_weighting/cues.py 32     # depth cues + check.json (optional: run.py builds them)
uv run experiments/genrecon_occlusion_weighting/run.py 32 --variant depth
uv run experiments/genrecon_occlusion_weighting/run.py 32 --variant gt
uv run experiments/genrecon_occlusion_weighting/run.py 32 --variant identity
uv run experiments/genrecon_occlusion_weighting/variance_check.py 32 [scene ...]  # --cue gt (default) or depth
uv run experiments/genrecon_occlusion_weighting/run.py 32 --variant gt --var-mode around_vmean --tag around_vmean
uv run scripts/evaluate_scannetpp.py outputs/genrecon_occlusion_weighting/vggt/32views   # and gt, identity
uv run scripts/evaluate_scannetpp.py outputs/unposed_genrecon/32views           # baseline

# Solv3R's cameras and depth: add --model solv3r to every command above
uv run experiments/genrecon_occlusion_weighting/run.py 32 --variant depth --model solv3r
uv run scripts/evaluate_scannetpp.py outputs/genrecon_occlusion_weighting_solv3r/solv3r/32views
uv run scripts/evaluate_scannetpp.py outputs/unposed_genrecon_solv3r/32views    # baseline
```

`--matched` uses `run_eval.py --matched`'s GenRecon settings (baseline `outputs/unposed_genrecon[_solv3r]_matched`).
Scene IDs after the view count restrict the run. `--stages ss,shape` weights only those stages (default all
three). `--var-mode` picks the variance (default `weighted`); `--var-mode zero` is an ablation that sets
$\boldsymbol\sigma^2=0$ everywhere, to measure how much the aggregator uses it. `--fusion mean` makes the
aggregator return only the mean over the views, skipping its MLPs and the softmax (default `mlp`). `--tau-vox --rho --softness --lam --eps --kappa` override the defaults in `visibility.Params`; use
`--tag` to give such a run its own output folder.

## Outputs

```text
outputs/genrecon_occlusion_weighting[_solv3r]/
├── cues/<N>views/<scene>/
│   ├── vggt.npz | solv3r.npz, gt.npz   names, depth [S, H, W], conf [S, H, W], K [S, 3, 3] (cues.py)
│   └── check.json                  the model's depth against the scan, per view and mean
├── variance_check/<cue>[_matched]/<N>views/<scene>/
│   ├── variances.npz               per cell of stage 1's grid: each mode's variance, sum of valid, sum of v
│   └── check.json                  AUROC per mode
└── <vggt | solv3r | gt | identity>[_matched][_<tag>]/<N>views/<scene>/
    ├── mesh.ply.zst, align.json    as unposed_genrecon/run_eval.py
    ├── scene/                      convert.py's GenRecon input
    └── genrecon/
        ├── visibility_args.json    inject.py's options
        ├── visibility_stats.json   per stage: pairs with v < 0.5, voxels in the kappa fallback,
        │                           mean sum of v, histogram of Delta / tau
        └── ...                     GenRecon's usual outputs
```

What to check before reading the mesh scores:

- `check.json`: `median_gt_over_pred` should be close to 1, and AbsRel small. Otherwise the cue has a frame or
  scale error, not just noise.
- `visibility_stats.json` for `gt`: the Delta / tau histogram should have a peak near 0 (surface voxels) and
  mass above 1 (occluded pairs). Pairs with v < 0.5 should be a minority, and voxels in the fallback rare.
- `identity` should match `outputs/unposed_genrecon[_solv3r]` up to GPU nondeterminism.

## Files

| File | Env | Role |
| --- | --- | --- |
| `visibility.py` | torch | $v$, the $\kappa$ fallback, the variance modes, and the v-weighted copies of both aggregator forwards |
| `inject.py` | GenRecon | patches `FullSceneImagesTo3DPipeline` (as `genrecon_oracle_occupancy/inject.py` does) and calls `reconstruct_scene.main()` |
| `cues.py` | root | builds the model's depth cue and the `gt` cue, reusing `evaluate_scannetpp.py`'s scan loading and depth rendering and `run_eval.py`'s `umeyama` |
| `run.py` | root | the scene loop. It reuses `run_eval.process()` (images, VGGT-Omega or Solv3R, convert.py, GenRecon, alignment with the scan), with `inject.py` as GenRecon's entry point |
| `variance_check.py` | root | AUROC of each variance mode for stage 1's true and false cells, from `inject.py --vis_dump` |

GenRecon is not modified. The only change outside this folder is in `run_eval.py`: a `script` argument
(defaulting to `reconstruct_scene.py`), and convert.py + GenRecon split out of `process()` as `genrecon()`.

Not part of this experiment yet: multi-point sampling per voxel, the posed (DSLR) setting, and view counts
other than 32 (`run.py` accepts any N that has unposed runs).
