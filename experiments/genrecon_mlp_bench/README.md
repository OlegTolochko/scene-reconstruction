# MLP bench: does a changed aggregator finetune better than GenRecon's?

GenRecon's multi-view aggregator (`papers/GenRecon/genrecon/modules/cond_3D/`) never sees depth. This bench
finetunes variants of it that get depth cues from Solv3R, and compares each with the unchanged aggregator finetuned
the same way. A variant is one small class. One `sbatch` trains it and logs its curves to Weights & Biases next
to the baseline's.

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

Included: `depth_inputs` (the cues as inputs of both MLPs, variant A of the other experiment), `depth_inputs_only`
(the same, only the new layers train) and `visible_mean` (a learned weight per view in the mean and variance).

## Run it

Once, on the cluster: `uv pip install --python papers/GenRecon/.venv/bin/python wandb`, then
`papers/GenRecon/.venv/bin/wandb login`, and check the settings at the top of `run.sbatch` (`DATA`, `TRAIN`, `BENCH`).
Then, from the repo root:

```bash
sbatch experiments/genrecon_mlp_bench/run.sbatch my_idea       # trains the baseline first if the bench has none
sbatch experiments/genrecon_mlp_bench/run.sbatch               # the baseline alone
sbatch experiments/genrecon_mlp_bench/run.sbatch genrecon 1    # the baseline with another training seed
```

- The baseline is the variant `genrecon`: the unchanged aggregator, finetuned on the same rooms with the same
  settings and cameras. A bench has one; later jobs find it finished and skip it.
- A bench is named in `run.sbatch` (`BENCH`). Its rooms and `config.json` are frozen in its folder when it is
  first set up, so all its runs are comparable. To change steps, learning rate or rooms, give `BENCH` a new name.
- Resubmitting a command resumes its run. Once a variant has been submitted, give a changed version a new name:
  a run refuses to continue with other code than it started with.
- Checkpoints hold only the aggregator (about 130 MB per saved step, with the EMA weights and the optimiser
  state), not the 1.3 B-weight model.

## Read it

All runs of a bench are one group in the W&B project `genrecon-mlp-bench`, named by their variant.

| Chart | Meaning |
| --- | --- |
| `val/mse` | validation loss. Every run and step is scored on the same views, timesteps and noise, so curves can be compared directly, and all runs start at the same value (released GenRecon, before any update) |
| `vs_genrecon/val_mse_percent` | the run's `val/mse` relative to the baseline's at the same step; below 0 is better than the baseline |
| `val/bin_<k>/mse` | `val/mse` for the timesteps in `[k/10, (k+1)/10)`; 1 is pure noise |
| `loss/mse`, `status/*` | GenRecon's own training curves |

The run summary adds `val/best_mse`, `vs_genrecon/best_mse_percent` and `vs_genrecon/reaches_genrecon_best_at_step`
(how many steps the variant needs for the baseline's best loss). `genrecon_s1` against `genrecon` shows how much two
runs of the same model differ: a variant's gain means something once it is larger than that. If a node has no
connection, the run is logged offline and the job log prints the `wandb sync` command.

## Files

| File | Env | Role |
| --- | --- | --- |
| `variants.py` | torch | the variants; `python variants.py NAME` tries one |
| `aggregator.py` | torch | `Aggregator`, the registry, the depth cues |
| `bench.py` | torch | repeatable validation, W&B, comparison with the baseline, aggregator-only checkpoints |
| `train.py` | GenRecon | dataset with depth maps and trainer; runs GenRecon's `train.py` unchanged |
| `setup_run.py`, `config.json`, `rooms.csv` | root | freeze a bench's rooms and settings |
| `run.sbatch` | | the SLURM job: settings and training |

Outputs: `outputs/genrecon_mlp_bench/<bench>/<run>/` with `val.jsonl`, `summary.json`, `run.json` (variant and
its code), `ckpts/`, GenRecon's `log.txt` and `tb_logs/`.
