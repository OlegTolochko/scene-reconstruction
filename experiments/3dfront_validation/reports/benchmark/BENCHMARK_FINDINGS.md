# What the five-scene baseline shows

All five distinct rooms/houses completed reconstruction, raw-mesh evaluation,
held-out-view evaluation, and textured-preview comparison. The full tables,
definitions, per-object results and diagnostic images are in
[BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md); the machine-readable summary is
[BENCHMARK_METRICS.csv](BENCHMARK_METRICS.csv).

At a distance tolerance of **1% of each furniture-reference diagonal D**,
the macro scene mean furniture F-score is **0.644**, precision **55.5%**, and
recall **83.1%**. Mean held-out depth coverage is **97.0%** and depth AbsRel
is **0.0143**, while silhouette IoU is **0.512**. Each scene has equal weight.
These are source-relative diagnostic scores, not centimetre measurements or
numbers directly comparable with another paper's evaluation protocol.

## Observed failures by scene

| Scene | Evidence | Interpretation and useful next test |
| --- | --- | --- |
| 01 — MasterBedroom-2820 | Bed recall is 96.8% at 1% D; individual lighting assets score 53.6%, 73.7%, and 81.6%. One cabinet asset scores 76.1%. The held-out images show lamp/detail mismatches and additional low planar geometry. | Large-object success can hide errors on smaller assets. Inspect visible lamp surfaces and supports with the per-object table and error map; test a detail/coverage change using the same held-out cameras. |
| 02 — LivingRoom-11414 | Furniture recall is 97.0%, but precision is 58.5%; 65.0% of predicted held-out foreground lies on furniture-reference background. Native selection uses six of eight supplied RGB views and issues six closest-camera fallback warnings. Images show a broad added lower plane and an upper detached patch. | Preserve the good furniture coverage while checking unmatched scene completion. A projected-area selection ablation is justified, but the current scores do not establish that the two dropped views caused the errors. |
| 03 — Library-27940 | Held-out coverage is 100%, yet silhouette IoU is 0.280 and extra predicted foreground is 72.0%. Visible-surface recall is 90.8%, versus 63.4% for occluded reference surfaces. Images show added lower, upper and side planes around the furniture. | Coverage alone rewards filling rays. Pair depth errors with silhouette precision/extra coverage and whole-surface completeness. Inspect added geometry against the supplementary reference assets. |
| 04 — Bedroom-18956 | Held-out IoU is 0.981 and depth AbsRel is 0.00430. Visible reference-surface recall is 99.5%, versus 42.4% for occluded surfaces; whole-furniture recall is 71.8%. The cabinet asset dominates the reference surface area and scores 66.9%. | Accurate observed silhouettes and depths do not establish complete 3D geometry. Hidden reference surfaces can include internal asset geometry as well as backsides. Test view coverage or completion priors, and retain the visible/occluded split. |
| 05 — DiningRoom-27978 | Lighting recall is 69.9%, versus 88.9% for the cabinet asset. Held-out foreground is missing prediction on 9.5% of rays, while extra foreground reaches 80.1%. Images show missing/displaced hanging-light details and an added lower plane. | Thin visible structures and unconstrained added geometry both matter. Use the lamp-specific surface measurements alongside missing/extra masks; a low overlap-only depth error does not capture both failures. |

Object recall values in this table use the same 1% D tolerance. The room names
are release labels: the processed scenes contain only **2–6 primary furniture
assets**, so these labels do not guarantee a complete bedroom or dining-room
inventory. All visual observations above were checked against the saved
held-out diagnostic images.

## Added architecture needs a separate interpretation

The RGB/depth reference renders contain the primary furniture collection.
Additional released files can contain floor, wall, ceiling and unlabeled other
geometry. Matching their supplied geometry increases mean predicted-surface
precision from **55.5% to 80.1%** at the same tolerance. Thus, a substantial
part of the furniture-unmatched prediction is compatible with released scene
context; the furniture-only error must not all be called hallucination.

However, mean recall over **all released assets** is only **28.0%**, with mean
F-score **0.388**. For scene 05, precision rises from 29.9% against furniture
to 86.8% against all assets, but all-assets recall is only 7.9%. Matching some
added surfaces does not establish complete room reconstruction.

This secondary comparison measures completion from furniture-only images.
It does not measure performance on images that visibly capture the room shell.
The supplementary asset collection has not been independently established as
the complete original 3D-FRONT room. Corresponding standalone furniture assets
and combined-scene nodes have identical bounds in all five rooms, supporting
use of the released coordinate frame; no alignment is fitted to predictions.

## Export can change the score without large surface displacement

The library's raw furniture F-score is **0.500**, versus **0.457** after preview
export: a **4.33 percentage-point decrease**. Raw-to-preview P95 distance is
only **0.000446 D**, and reference recall changes from 77.242% to 77.233%.
Predicted-surface precision changes from 36.963% to 32.419%, and total preview
surface area is **90.5%** of raw surface area.

These measurements are consistent with changed area weighting after export
cleanup/simplification. The exact responsible operation has not been isolated.
Keep raw geometry as the baseline artifact and report preview results
separately. An export-only follow-up can compare global/pre-bake geometry dumps
with the final per-chunk meshes, using the saved `.pt` files without rerunning
GenRecon.

## First controlled improvement test

Use **scene 02** to compare the native projected-area gate (**0.4**) with a
lower gate, for example **0.1**. Hold the eight supplied views, three held-out
views, support points, seed 42, checkpoints, sampler settings and coordinate
frame fixed. Record the actual retained views and fallback warnings, then
compare raw geometry, both reference scopes, and held-out depth/silhouettes.

This tests whether input selection discards useful observations. It is a
hypothesis, not an improvement demonstrated by the five baseline runs. The
other four scenes already retain all eight supplied RGB views, so a selection
change would not explain all observed failures.

After that, prioritize a fixed-view-count coverage experiment on the visible
lamp failures and the scene 04 occluded-surface gap. Recover metric scale before
using centimetre thresholds. More scenes and a verified training-house split
are needed before making generalization claims.

## Execution and scope

Scenes 01–02 ran on the original RTX 5090 host; scenes 03–05 ran through the
user-launched SSH job on the RTX 5080 host using shared files and cached models.
The living-room inference took about 36.8 minutes under competing workloads;
these mixed-host/shared-device timings are not an isolated speed benchmark.
New-run process GPU allocations were sampled every two seconds and can miss
brief peaks. The pilot's earlier remeshing OOM is not evidence of an intrinsic
requirement for more than 32 GiB of VRAM.

The baseline uses supplied cameras and reference-depth-derived support points,
with eight supplied views and three held-out views per scene. It does not test
camera estimation, an RGB-only system, texture accuracy, or physical support
relationships. Reference surfaces include hidden/internal asset geometry.
Five convenience-selected houses and unverified GenRecon training overlap limit
claims beyond these diagnostic cases. The 192 × 192 held-out ray grid samples
the original images; thin image structures are not evaluated at full resolution.

All five scenes were retained because their source data total only about 47 MB.
Archive files were never saved, and all runs reused the model/environment cache.
Large reconstruction files remain under `outputs/`; compact metrics and figures
are retained in `benchmark_results/` for repository review.

Measured relevant installation/cache/data/output/experiment roots occupy **30.6 GB (28.5 GiB)**, below the requested **50 GB** budget. See [storage audit](benchmark_results/storage.json).
