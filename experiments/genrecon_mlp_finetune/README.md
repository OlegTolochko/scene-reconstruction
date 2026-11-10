# GenRecon with depth cues learned by the aggregator

Status: plan, nothing implemented yet. Comments welcome, especially on the open questions at the end.

Does GenRecon improve if its multi-view aggregator is finetuned with depth cues?

GenRecon's aggregator (`papers/GenRecon/genrecon/modules/cond_3D/`) merges the DINOv3 features of all views a
voxel projects into: the mean and variance over the views, an MLP per view on `[f_i, mean, var]`, and a softmax
over the views. It never sees depth, so a voxel hidden behind an occluder gets the occluder's feature at full
weight. `experiments/genrecon_occlusion_weighting` tests a hand-made, training-free fix. This experiment lets the
aggregator learn how to use the cues instead. The training-free version is the ablation it has to beat.

## Method

- Depth source: Solv3R, whose depth beats VGGT-Omega's in `experiments/solv3r_depth`. It gives depth, a
  confidence per pixel and cameras fitted to its predicted rays, in the same format as VGGT-Omega's output.
- Cues per (view, voxel): the depth residual `Δ/τ` (voxel depth minus observed depth, as in
  `genrecon_occlusion_weighting/visibility.py`) and Solv3R's confidence. The confidence's range differs from
  VGGT-Omega's, so the ramp in `visibility.py` (2 to 5) does not carry over; it is normalised from its
  distribution on the training renders. The camera-to-voxel distance may be added later.
- The cues are extra inputs to the first layer of both MLPs, with zero-initialised weights, so training starts
  exactly at GenRecon. A test checks that this is bit-identical to GenRecon.
- The cues also weight the mean and variance, through a learnable version of the visibility weight `v`.
  Otherwise an occluder's feature still contaminates the mean before the MLP sees anything. This is a second
  variant, so the two can be compared.
- Only the aggregator is trained, starting from the released checkpoints. The DiTs and LoRA are frozen, and the
  loss is GenRecon's flow-matching loss. The sparse-structure stage comes first, then shape, texture only if
  needed.
- GenRecon is not modified. The changes are patched in from this folder, as in the other experiments.

## Training data

SAGE-10k, prepared with GenRecon's toolkit (`papers/GenRecon/data_toolkit_scenes/`), with three changes:

1. Scenes from the end of the SAGE-10k list instead of the first N, which GenRecon most likely used. Overlap
   with GenRecon's training scenes is then unlikely, though not certain.
2. A train/val split per room with a fixed seed. The scene lists are committed. GenRecon publishes no split.
3. Depth per rendered view:
   - The cues come from Solv3R, run on each room's renders, because the cue at test time is Solv3R's. This also
     gives a real confidence. Solv3R predicts at 504 pixels on the long side, so depth and K are rescaled to the
     render resolution before the dataset crops them.
   - The cameras are Solv3R's too, aligned to the true ones with a similarity transform (as in
     `unposed_genrecon/run_eval.py`). Training then sees the same pose and depth errors as the unposed setting
     at test time (`run_eval.py --model solv3r`).
   - The targets stay in the true chunk frame. Rooms where the alignment fails are dropped.
   - A Z pass in `render_rooms.py` gives the true depth. It is used to check the alignment and the chunk frames,
     and for the oracle variant.

Only the sparse-structure latents are needed for the first stage.

## Evaluation

| Set | Scenes | Use |
| --- | --- | --- |
| SAGE val | our split | training monitor only (loss, divergence, zero-init check), since it may overlap GenRecon's training data |
| ScanNet++ dev | 15–20 scenes from `nvs_sem_train`, fixed list | every decision: cues, steps, variants, checkpoint |
| ScanNet++ test | the 50 `nvs_sem_val` scenes | the final runs only, evaluated with `scripts/evaluate_scannetpp.py` |

| Run | Question it answers |
| --- | --- |
| GenRecon, unposed with Solv3R's cameras (`unposed_genrecon_solv3r`) | baseline |
| `genrecon_occlusion_weighting` with Solv3R's depth | does learning beat a hand-made mask? Needs a Solv3R variant in its `cues.py`, which only reads VGGT-Omega so far |
| finetuned without cues: same data, steps and Solv3R cameras | control: separates the cues from "more finetuning on SAGE" |
| finetuned with Solv3R cues | the method |
| finetuned with true depth | oracle: is the limit the cue or the mechanism? |

The main claim is the method against the control. Against released GenRecon, the comparison also includes the
extra data and steps.

## Order

1. Data: scene selection, split, Z pass, Solv3R on the renders, depth sanity check on a few rooms.
2. Aggregator with cue inputs, and the zero-init test.
3. Dataset loading the cues, training for the sparse-structure stage, the control with the same settings.
4. ScanNet++ dev set, then the shape stage if the sparse-structure stage helps.

## Open questions

- How many rooms? About 500–1000 should be enough for an aggregator-only finetune.
- Which cluster, and where does `DATA_ROOT` live? The toolkit runs on Linux only (Blender 5.0.1).
- Aggregator only, or the aggregator and LoRA?
- Solv3R's cameras from the rays (the paper's protocol, used so far) or from the camera head?
