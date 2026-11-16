# MLP bench: does a changed aggregator finetune better than GenRecon's?

GenRecon's multi-view aggregator (`papers/GenRecon/genrecon/modules/cond_3D/`) never sees depth. This bench
finetunes variants of it that get depth cues from Solv3R, and compares each with the unchanged aggregator finetuned
the same way. A variant is one small class. One `sbatch` trains it, logs its curves to Weights & Biases next
to the baseline's and uploads its meshes of the validation rooms.

It builds on `experiments/genrecon_depth_cue_finetuning` (branch `feat/depth-cue-finetuning`): the same data, depth
cues, cue cameras and training setup, copied here so this folder runs on its own. GenRecon is not modified.

## Write a variant

Add a class to `variants.py`. It subclasses `Aggregator` (`aggregator.py`), which is GenRecon's aggregator split
into three steps, and overrides what it changes:

```python
@register("my_idea")
class MyIdea(Aggregator):
    def __init__(self, base):
        super().__init__(base)                       # GenRecon's feature_mlp and weight_mlp, pretrained
        self.gate = zero_linear(N_CUES, 1)           # new layers start at zero: training starts at GenRecon

    def view_logits(self, feats, mu, var, mask, cues):
        return super().view_logits(feats, mu, var, mask, cues) + self.gate(cues)
```

| Step | Returns | GenRecon's version |
| --- | --- | --- |
| `statistics(feats, mask, cues)` | mean and variance over the views, `[B, 1, V, D]` each | plain mean and variance of the views that see the voxel |
| `view_features(feats, mu, var, mask, cues)` | what each view adds to the mean, `[B, N, V, D]` | `feature_mlp` on `[feature, mean, variance]` |
| `view_logits(feats, mu, var, mask, cues)` | one logit per view, `[B, N, V, 1]` | `weight_mlp` on the same input |

`cues` is `[B, N, V, 4]`, per view and voxel, all 0 where the view has no depth there: `BEHIND` (voxel depth minus
observed depth, in voxels, clipped to ±8; below 0 is free space, above 0 is occluded), `BEHIND_REL` (the same
relative to the observed depth, ±0.5), `CONFIDENCE` (Solv3R's, standardised) and `KNOWN`. By default the whole
aggregator trains and the rest of GenRecon is frozen; `train_only = ("gate.",)` restricts it to those parameters.

Try it in seconds, without a GPU (float32, and bfloat16 autocast with a backward pass):

```bash
uv run --with torch experiments/genrecon_mlp_bench/variants.py my_idea
```

Included:

- `depth_cue_first_layer`: the cues as extra inputs of the first layer of both MLPs (variant A of the other
  experiment). In the bench `2026-11-14_overfit10` its curve is the baseline's: its cue layers stayed close to zero.
- `depth_cue_first_layer_x10`: the same with the cues times 10, which is a 10 times larger learning rate for the
  cue layers alone.
- `depth_cue_only_x10`: that one with only the cue layers training. Nothing else can memorise the rooms, so what
  its validation loss falls below its value at step 0 is due to depth alone. Compare it with its own step 0:
  `vs_genrecon/*` compares it with a baseline that does memorise.

### Depth routes: where can frozen GenRecon take depth in?

After the cue modulator of `genrecon_mlp_finetuning` (branch `feat/depth-cue-finetuning`), which improves the
sparse structure on SAGE when only it trains. Its three outputs were trained together there; here each route
runs alone. Only small cue networks train (20k to 170k weights), GenRecon's two MLPs stay frozen. The networks
see only depth: the cues, bumps around "the voxel is on the surface this view sees", steps for in front of it
and behind it, and their mean and maximum over the views.

| Variant | Depth may |
| --- | --- |
| `depth_routes_all` | all three below |
| `depth_routes_control` | the same networks without the depth values (only which views have depth) |
| `depth_route_weights` | choose among the views: a logit offset and a weight in the mean and variance |
| `depth_route_gate` | scale a voxel's fused feature; towards 0 is the feature of a voxel no view sees |
| `depth_route_residual` | add a learned feature to a voxel, a direction the views' features do not have |

Read them on `vs_released/val_mse_percent` (and `val/bin_<k>/mse` for the noisy timesteps, where the structure
is decided), against the control and not against `genrecon`, which memorises. First `depth_routes_all` and
`depth_routes_control`: without a gap between them the single routes need not run. Then the single routes: the
claim behind the modulator is that weighting the views cannot separate a voxel in front of a wall from one on
it, since all views give the same wall feature, so `depth_route_weights` should gain nothing and
`depth_route_residual` most of it. Expect a few percent: in the other experiment, a validation loss 4 % lower
went with a sparse-structure IoU of 0.52 instead of 0.41 on the overfitted room.

The same with the true depth, as the limit of what the depth model can give (a bench of its own):

```bash
BENCH=2026-11-15_overfit10_true_depth CUES=depth_cues_true sbatch experiments/genrecon_mlp_bench/run.sbatch depth_routes_all
```

## Run it

Once, on the cluster: `uv pip install --python papers/GenRecon/.venv/bin/python wandb`, then
`papers/GenRecon/.venv/bin/wandb login`, and check the settings at the top of `run.sbatch` (`DATA`, `TRAIN`, `BENCH`).
Then, from the repo root:

```bash
sbatch experiments/genrecon_mlp_bench/run.sbatch my_idea          # that variant only
sbatch experiments/genrecon_mlp_bench/run.sbatch --mesh my_idea   # ... and its meshes
sbatch experiments/genrecon_mlp_bench/run.sbatch                  # the baseline
sbatch experiments/genrecon_mlp_bench/run.sbatch --mesh           # the baseline and all mesh references
sbatch experiments/genrecon_mlp_bench/run.sbatch genrecon 1       # the baseline alone, with another training seed
```

- A job does only what its command names. A variant job trains that variant, nothing else; with `--mesh` it
  also makes its meshes. Meshes are off by default.
- The command without a variant trains the baseline (the variant `genrecon`: the unchanged aggregator, finetuned
  on the same rooms with the same settings and cameras). With `--mesh` it first makes the other references of the
  bench, once: the true rooms as meshes (`ground_truth.py`) and the meshes of released GenRecon, without any
  finetuning; then the baseline's meshes. If one of the first two fails, the job goes on and ends with status 1.
- A bench is named in `run.sbatch` (`BENCH`). Its rooms and `config.json` are frozen in its folder when it is
  first set up, so all its runs are comparable. To change steps, learning rate or rooms, give `BENCH` a new name.
- Resubmitting a command resumes its run. Once a variant has been submitted, give a changed version a new name:
  a run refuses to continue with other code than it started with.
- Checkpoints hold only the aggregator (about 130 MB per saved step, with the EMA weights and the optimiser
  state), not the 1.3 B-weight model.
- With `--mesh`, after training the job reconstructs the bench's validation rooms with the run's aggregator
  (`mesh.py`) and uploads the meshes. For a run that is already trained, resubmitting with `--mesh` does only that.

## Read it

All runs of a bench are one group in the W&B project `genrecon-mlp-bench`, named by their variant.

| Chart | Meaning |
| --- | --- |
| `val/mse` | validation loss. Every run and step is scored on the same views, timesteps and noise, so curves can be compared directly, and all runs start at the same value (released GenRecon, before any update) |
| `vs_genrecon/val_mse_percent` | the run's `val/mse` relative to the baseline's at the same step; below 0 is better than the baseline |
| `vs_released/val_mse_percent` | the run's `val/mse` relative to its own step 0, released GenRecon: the chart for runs that train only cue layers |
| `val/bin_<k>/mse` | `val/mse` for the timesteps in `[k/10, (k+1)/10)`; 1 is pure noise |
| `val/occ/iou`, `val/occ/f1_tol1`, ... | occupancy of the sparse structure: each validation chunk sampled from pure noise with GenRecon's inference sampler (12 steps, CFG), decoded and compared with the true occupancy (64³ voxels); `tol1` counts a voxel one voxel off as right. Same chunks, views and noise for every run and step (`occ_seeds` noises per chunk, 1 by default) |
| `vs_genrecon/occ_iou_points`, `vs_released/occ_iou_points` | the run's `val/occ/iou` minus the baseline's at the same step, or minus its own step 0, in points (×100); above 0 is better. The same for `f1_tol1` |
| `loss/mse`, `status/*` | GenRecon's own training curves |

The run summary adds `val/best_mse`, `vs_genrecon/best_mse_percent` and `vs_genrecon/reaches_genrecon_best_at_step`
(how many steps the variant needs for the baseline's best loss). `genrecon_s1` against `genrecon` shows how much two
runs of the same model differ: a variant's gain means something once it is larger than that. If a node has no
connection, the run is logged offline and the job log prints the `wandb sync` command.

Each run also has `mesh/<room>`: GenRecon's full-scene reconstruction of every validation room (in an overfitting
bench: the rooms it trained on), with the run's final aggregator in the sparse-structure stage and released
GenRecon for the rest. The run `genrecon_released` of the same group has the same rooms from released GenRecon,
without any finetuning. All of them use Solv3R's cameras, the same views, chunks and seed, so the meshes of a room
differ only by the aggregator. The uploaded meshes are GenRecon's textured GLBs (as its `chunked_to_glb.py` makes
them: remeshed, with the colours baked into a 1024-pixel texture per chunk), simplified to 4 million faces and without
the ceiling (cut at 2.4 m), so the viewer looks into the room. The run `ground_truth` has the true rooms: Blender's
export of the rooms the views were rendered from, in the same frame and with the same cut, with textures shrunk to
1024 pixels. To compare runs, add a media panel for `mesh/<room>`.

## Files

| File | Env | Role |
| --- | --- | --- |
| `variants.py` | torch | the variants; `python variants.py NAME` tries one |
| `aggregator.py` | torch | `Aggregator`, the registry, the depth cues |
| `bench.py` | torch | repeatable validation, W&B, comparison with the baseline, aggregator-only checkpoints |
| `train.py` | GenRecon | dataset with depth maps and trainer; runs GenRecon's `train.py` unchanged |
| `mesh.py` | GenRecon | meshes of the validation rooms for a run or for released GenRecon, uploaded to W&B |
| `ground_truth.py`, `export_room_glb.py` | GenRecon, Blender | the true rooms as meshes, uploaded to W&B |
| `setup_run.py`, `config.json`, `rooms.csv` | root | freeze a bench's rooms and settings |
| `run.sbatch` | | the SLURM job: settings, training and meshes |

Outputs: `outputs/genrecon_mlp_bench/<bench>/<run>/` with `val.jsonl`, `summary.json`, `run.json` (variant and
its code), `ckpts/`, `meshes/<room>/mesh.glb`, GenRecon's `log.txt` and `tb_logs/`.
