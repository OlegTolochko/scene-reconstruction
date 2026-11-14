# GenRecon with depth cues learned by the aggregator

Status: variant A is implemented for the sparse-structure stage. A first pilot (10 rooms) was inconclusive (see
Results). Next: more rooms and steps, the oracle, variant B.

Does GenRecon improve if its multi-view aggregator is finetuned with depth cues?

GenRecon's aggregator (`papers/GenRecon/genrecon/modules/cond_3D/`) merges the DINOv3 features of all views a
voxel projects into: the mean and variance over the views, an MLP per view on `[f_i, mean, var]`, and a softmax
over the views. It never sees depth, so a voxel hidden behind an occluder gets the occluder's feature at full
weight. `experiments/genrecon_occlusion_weighting` (on branch `feat/depth-cues`) tests a hand-made,
training-free fix. This experiment lets the aggregator learn how to use the cues instead. The training-free
version is the ablation it has to beat.

## Method

- Depth source: Solv3R, whose depth beats VGGT-Omega's in `experiments/solv3r_depth`. It gives depth, a
  confidence per pixel and cameras fitted to its predicted rays, in the same format as VGGT-Omega's output.
- Cues per (view, voxel): the depth residual (voxel depth minus observed depth) and Solv3R's confidence,
  standardised with its mean and standard deviation on the training renders.
- Variant A: the cues are extra inputs to the first layer of both MLPs, with zero-initialised weights, so
  training starts exactly at GenRecon.
- Variant B (not implemented): the cues also weight the mean and variance, through a learnable version of the
  visibility weight `v` of `genrecon_occlusion_weighting`. Otherwise an occluder's feature still contaminates
  the mean before the MLP sees anything.
- Only the aggregator is trained, starting from the released checkpoints. The DiTs and LoRA are frozen, and the
  loss is GenRecon's flow-matching loss. The sparse-structure stage comes first, then shape, texture only if
  needed.
- GenRecon is not modified. The changes are patched in from this folder, as in the other experiments.

## Training data

SAGE-10k, prepared with GenRecon's toolkit (`papers/GenRecon/data_toolkit_scenes/`) by `rooms/prepare_rooms.sh`:

1. Rooms: the 257 we downloaded, every second zip of the first 514 in SAGE-10k's sorted list
   (`rooms/select_rooms.py`, listed in `rooms/sage_rooms.csv`). These may overlap GenRecon's training scenes, so
   SAGE val only monitors training and every decision is made on ScanNet++.
2. Splits, with a fixed seed: `overfit` (1 room), `train_1` (50), `val_1` (10), `reserve` (196).
3. Depth per rendered view:
   - The cues come from Solv3R, run on each room's renders, because the cue at test time is Solv3R's. Solv3R
     predicts at 504 pixels on the long side, so depth and K are rescaled to the render resolution before the
     dataset crops them.
   - The cameras are Solv3R's too, aligned to the true ones with a similarity transform (`prepare_cues.py`, as
     in `unposed_genrecon/run_eval.py`). Training then sees the same pose and depth errors as the unposed
     setting at test time (`run_eval.py --model solv3r`). Rooms where the alignment fails are dropped.
   - The targets stay in the true chunk frame, from `get_slats.py`'s sparse-structure latents.
   - The true depth (`rooms/export_room_mesh.py`, then `rooms/true_depth.py` ray-casts the room mesh) checks the
     alignment (`rooms/check_depth.py`) and is the cue of the oracle.

## Evaluation

| Set | Scenes | Use |
| --- | --- | --- |
| SAGE val | our split | training monitor only (loss, divergence), since it may overlap GenRecon's training data |
| ScanNet++ dev | 15–20 scenes from `nvs_sem_train`, fixed list | every decision: cues, steps, variants, checkpoint |
| ScanNet++ test | the 50 `nvs_sem_val` scenes | the final runs only, evaluated with `scripts/evaluate_scannetpp.py` |

| Run | Question it answers |
| --- | --- |
| GenRecon, unposed with Solv3R's cameras (`unposed_genrecon_solv3r`) | baseline |
| `genrecon_occlusion_weighting` with Solv3R's depth (`feat/depth-cues`) | does learning beat a hand-made mask? |
| finetuned without cues: same data, steps and Solv3R cameras | control: separates the cues from "more finetuning on SAGE" |
| finetuned with Solv3R cues | the method |
| finetuned with true depth | oracle: is the limit the cue or the mechanism? |

The main claim is the method against the control. Against released GenRecon, the comparison also includes the
extra data and steps.

## Variant A: per-view cues into both MLPs

GenRecon is not modified: `train.py` registers a dataset and a trainer with GenRecon's registries, wraps
`build_single_model`, and then runs `papers/GenRecon/train.py` unchanged.

- Cues per (view, voxel), 0 where the depth is unknown (`cues.py`): `clip(Δ/voxel, ±8)`, `clip(Δ/D, ±0.5)`,
  the confidence standardised with `conf_mean`/`conf_std` (set by `make_run.py` from the training rooms), and a
  `known` flag.
- Opt-in extra cues (`models.denoiser.cues.extra` in the config): `footprint`, the voxel's size in DINOv3
  patches in this view, and `similarity`, the cosine between this view's feature and the mean of the others.
- Two `Linear(N, 1024)` without bias, zero-initialised, add to the first-layer pre-activations of
  `feature_mlp` and `weight_mlp` through forward pre-hooks. GenRecon's parameters and checkpoint keys are
  untouched, and at initialisation the aggregator is GenRecon's. There is no unit test for this yet;
  `run_scannetpp.py --zero-init` checks it end to end against `unposed_genrecon_solv3r`.
- Trained: the aggregator only (`cues.train` in `config_ss.json`), starting from the released checkpoint
  (`cues.init_ckpt`). `--control` sets all cues to 0, so the cue layers stay at 0: same data, steps and cameras.
- `cue_cameras: true` replaces the true cameras by the cue's (with GenRecon's crop applied), as at test time.
- Opt-in `dataset.args.cue_noise` perturbs the cue depth by a per-view scale and a smooth multiplicative field,
  to mimic real-image depth errors.

Cue data per room, in `<depth_cues>/<scene_id>/`:

| File | Contents |
| --- | --- |
| `cameras.npz` | `K [F, 3, 3]` normalised to the full render; `c2w [F, 4, 4]` camera-to-room, Blender convention as in `transforms.json`, aligned to the true cameras; F = the frames of `transforms.json` in its order |
| `<i:04d>.npz` | `depth [H, W]` z-depth in room units, `conf [H, W]`, for frame `i`, covering the full render (any resolution) |

For the oracle, the same format with the true depth, the true cameras and `conf = 1`.

## Run

| File | Role |
| --- | --- |
| `rooms/prepare_rooms.sh` | SLURM job: toolkit, true depth, Solv3R, cue files and depth check for every room of a data root |
| `make_run.py` | a run's config, data folders restricted to its rooms, and the confidence standardisation |
| `run_train.sh` | SLURM job for one arm (`cues` or `control`); resubmitting resumes, for runs longer than 6 h |
| `train.py` | the trainer (from `papers/GenRecon`, GenRecon env) |
| `inject.py`, `run_scannetpp.py` | unposed GenRecon on ScanNet++ with a finetuned aggregator in stage 1 |

```bash
# from the repo root
sbatch ... experiments/genrecon_mlp_finetuning/run_train.sh OUT cues -- --name NAME --train ROOT:all --val ROOT:all
sbatch ... experiments/genrecon_mlp_finetuning/run_train.sh OUT control -- --name NAME --train ROOT:all --val ROOT:all
uv run experiments/genrecon_mlp_finetuning/run_scannetpp.py 32 --name NAME_cues --run OUT/cues
uv run scripts/evaluate_scannetpp.py outputs/genrecon_mlp_finetuning/scannetpp/solv3r/NAME_cues/32views
```

## Results

Pilot (2026-11-12): 10 SAGE training rooms, 2000 steps, lr 1e-4, on data that still had wall thinning (since
removed). ScanNet++: 10 scenes, 32 views, Solv3R cameras. Outputs (on the cluster) are in
`outputs/genrecon_depth_cue_finetuning/` (the experiment's old name).

| Run | SAGE val loss | F-score | Chamfer |
| --- | --- | --- | --- |
| baseline (`unposed_genrecon_solv3r`) | – | 0.810 | 0.0603 |
| control | 0.2119 | 0.823 | 0.0606 |
| cues | 0.2111 | 0.820 | 0.0614 |
| cues + footprint + similarity (no matching control) | 0.2175 | 0.832 | 0.0639 |

The finetuning itself helps a little, but the cues add nothing measurable over the control. The cue layers
moved only a little: column norms of 0.03–0.23 against about 1.0 for the feature columns of the same layer.
The residual cues into `weight_mlp` moved the most. With 10 rooms, the pilot is too small to tell whether the
cues work.

## Next

- More rooms (54 are prepared; the plan is 500–1000) and more steps, cues against control, without wall thinning.
- A control for the extra cues.
- The oracle with true depth.
- Variant B.
- Save only the aggregator in the checkpoints (each is the full 2.8 GB denoiser).

## Open questions

- Aggregator only, or the aggregator and LoRA?
- Solv3R's cameras from the rays (the paper's protocol, used so far) or from the camera head?
