# Five-scene GenRecon diagnostic benchmark

Evaluated **5/5** frozen scenes. All camera/depth consistency gates passed.

[Interpretation and next tests](BENCHMARK_FINDINGS.md) · [Frozen camera split](benchmark_results/frozen_camera_split.png)

Eight supplied-camera RGB inputs, seed 42, and 10,000 support points backprojected from **input views only** per room. Three other released views are held out. GenRecon’s native camera visibility filter can discard supplied RGB views; the effective count and dropped IDs are recorded below. This evaluates reconstruction with supplied calibration/support, not an RGB-only system. Original metric scale and training-house disjointness remain unverified. The five houses are an early-archive convenience sample spanning room types, not a representative benchmark.

D is the diagonal of each original furniture scene bounding box. Raw meshes are evaluated in the supplied coordinate frame without alignment fitting or cropping. Walls/floor/ceiling are excluded from reference geometry because released conditioning renders omit them. Added architectural-looking geometry therefore counts as unmatched under this furniture-only protocol; this alone does not establish physically implausible geometry.

## Surface accuracy and completeness

100,000 surface-area-uniform samples per mesh, seed 42; each sample is measured against the other mesh’s nearest **triangle**, rather than against a finite point cloud. Precision is the predicted-area fraction within a threshold; recall is the reference-area fraction; F-score is their harmonic mean. Symmetric mean is the average of the two unsquared directed means. Distances below are fractions of D.

| Scene | Room | Precision @ 1% D | Recall @ 1% D | F @ 0.5% D | F @ 1% D | F @ 2% D | Symmetric mean / D |
|---|---|---:|---:|---:|---:|---:|---:|
| scene_01 | MasterBedroom-2820 | 82.6% | 86.2% | 65.4% | 84.4% | 92.3% | 0.00897 |
| scene_02 | LivingRoom-11414 | 58.5% | 97.0% | 51.6% | 73.0% | 77.5% | 0.02806 |
| scene_03 | Library-27940 | 37.0% | 77.2% | 40.2% | 50.0% | 61.8% | 0.05089 |
| scene_04 | Bedroom-18956 | 69.6% | 71.8% | 61.8% | 70.7% | 84.6% | 0.00913 |
| scene_05 | DiningRoom-27978 | 29.9% | 83.0% | 34.7% | 43.9% | 52.0% | 0.08581 |
| Macro scene mean | | 55.5% | 83.1% | 50.8% | 64.4% | 73.7% | 0.03657 |

## Held-out views

Camera-axis depth and silhouette queries use a fixed 192 × 192 grid of pixel centers sampled from the original 768 × 768 images. Background depth 65504 is invalid. AbsRel, MAE and RMSE use pixels where both reference and prediction hit; coverage and missing/extra rates prevent those errors from hiding holes. Per-room values pool held-out pixels; the final mean weights rooms equally. These are sampled-view metrics, not full-resolution raster benchmarks.

| Scene | Silhouette IoU | Depth coverage | Missing / GT | Extra / prediction | Depth AbsRel | MAE / D | RMSE / D | Correct depth <5%, all GT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scene_01 | 76.0% | 97.8% | 2.2% | 22.7% | 0.01399 | 0.01802 | 0.04956 | 93.9% |
| scene_02 | 34.7% | 97.1% | 2.9% | 65.0% | 0.00675 | 0.00891 | 0.02500 | 94.6% |
| scene_03 | 28.0% | 100.0% | 0.0% | 72.0% | 0.02773 | 0.03699 | 0.06806 | 82.8% |
| scene_04 | 98.1% | 99.4% | 0.6% | 1.3% | 0.00430 | 0.00464 | 0.00909 | 99.3% |
| scene_05 | 19.5% | 90.5% | 9.5% | 80.1% | 0.01852 | 0.02022 | 0.06759 | 84.6% |
| Macro scene mean | 51.2% | 97.0% | 3.0% | 48.2% | 0.01426 | 0.01776 | 0.04386 | 91.0% |

## Visibility and spatial diagnostics

Reference samples are visible when projected depth agrees with at least one input reference depth within max(0.2% D, 0.5% depth). Occluded samples are behind a valid input reference surface in at least one view and visible in none; other samples remain unclassified. This is a depth-based approximation, not a semantic visibility label. The free-space column marks predicted samples significantly in front of a valid input depth in any view (tolerance max(1% D, 1% depth)). Background projection and bounds flags are diagnostics, not independent proof of hallucination.

| Scene | Visible recall, 8 supplied views | Occluded recall, 8 supplied views | Visible recall, effective RGB views | Occluded recall, effective RGB views | Input free-space violation | Projects onto input background | Outside GT bounds +1% D |
|---|---:|---:|---:|---:|---:|---:|---:|
| scene_01 | 88.7% | 83.7% | 88.7% | 83.7% | 22.3% | 14.8% | 8.3% |
| scene_02 | 96.4% | 97.7% | 96.4% | 97.7% | 9.8% | 42.7% | 16.0% |
| scene_03 | 90.8% | 63.4% | 90.8% | 63.4% | 16.0% | 54.9% | 33.5% |
| scene_04 | 99.5% | 42.4% | 99.5% | 42.4% | 10.6% | 15.9% | 0.0% |
| scene_05 | 82.2% | 85.2% | 82.2% | 85.2% | 4.5% | 67.5% | 39.1% |

All visibility recalls in this table use the 1% D distance threshold. GenRecon’s native camera gate requires a chunk cube’s clipped projected bounding box to occupy at least **40% of the image** (`genrecon/datasets/components.py:237`). It tests projected area, rather than reference-depth occlusion; a rejected camera may still contain useful observed surfaces. This gate and closest-camera fallback are preserved for the baseline.

## Supplementary scene completion geometry

A secondary comparison includes **all released room assets**: the primary furniture mesh plus available `ceil.glb`, `wall.glb`, `floor.glb` and `others.glb`. This explanatory analysis was added after inspecting the supplementary assets; primary geometry settings and room selection remained fixed. It helps distinguish added geometry consistent with the supplied scene from unmatched additions. `others.glb` has no reliable semantic labels, and this processed asset collection is not independently verified as a complete original 3D-FRONT room. The distance threshold remains 1% of the **furniture** diagonal D, so primary and secondary precision use the same physical tolerance. Full-reference area sampling uses 100,000 samples, seed 43. Held-out depth/silhouette scores above still use the released furniture-only renders.

| Scene | All-assets precision @1% D | All-assets recall @1% D | All-assets F-score @1% D |
|---|---:|---:|---:|
| scene_01 | 89.2% | 32.6% | 47.7% |
| scene_02 | 79.7% | 51.3% | 62.4% |
| scene_03 | 72.3% | 35.6% | 47.7% |
| scene_04 | 72.6% | 12.7% | 21.6% |
| scene_05 | 86.8% | 7.9% | 14.5% |

## Raw geometry versus textured preview

The preview uses no global remesh, a per-chunk target of 250,000 triangles, 2048² textures, and standard hole filling/cleanup. Area-weighted sampling can change when export changes surface-area distribution, even when nearest-surface displacement stays small. F-score deltas below include Monte Carlo sampling variation; no repeat-seed uncertainty interval was estimated. Primary baseline scores always use the raw mesh.

| Scene | Raw F @1% D | Preview F @1% D | Delta, percentage points | Raw → preview P95 / D | Preview / raw surface area |
|---|---:|---:|---:|---:|---:|
| scene_01 | 84.4% | 84.6% | 0.19498 | 0.00009 | 0.94406 |
| scene_02 | 73.0% | 72.4% | -0.58045 | 0.00003 | 0.96912 |
| scene_03 | 50.0% | 45.7% | -4.33108 | 0.00045 | 0.90469 |
| scene_04 | 70.7% | 70.6% | -0.12345 | 0.00007 | 0.95052 |
| scene_05 | 43.9% | 42.9% | -1.05483 | 0.00004 | 0.96155 |

## Per-scene failures and object detail

Screening flags below use thresholds fixed in the evaluator (precision/recall <90%; missing/extra silhouette >5%; object recall <80% with ≥30 samples). These thresholds prioritize inspection and do not define benchmark pass/fail. Object samples are area-weighted, so tiny objects have less stable estimates. Full per-object geometry IDs, extents, surface area, visibility and recalls are retained in each `metrics.json` and `objects.csv`.

### scene_01: fe6717d3-31e0-41a2-83e6-98c29c3461df/MasterBedroom-2820

Inputs: 0000, 0001, 0003, 0004, 0006, 0007, 0009, 0010. Held out: 0002, 0005, 0008.
Effective RGB conditioning: 0000, 0001, 0003, 0004, 0006, 0007, 0009, 0010. Native chunks: 4; image-selection fallback warnings: 0.
Reference: 59,391 triangles; raw prediction: 5,677,912 triangles.

[Held-out diagnostics](benchmark_results/scene_01/heldout_views.png) · [Surface error map](benchmark_results/scene_01/surface_errors.png) · [Full metrics](benchmark_results/scene_01/metrics.json) · [Objects CSV](benchmark_results/scene_01/objects.csv)

[Side-by-side textured GLB](../../../../outputs/3dfront_benchmark/scene_01/comparison_textured.glb) — reference furniture left, preview prediction right; display translation only. [All supplied reference assets](../../../../outputs/3dfront_benchmark/scene_01/reference/full_room.glb).

- Unmatched furniture-target geometry: 17.4% of sampled predicted surface lies farther than 1% diagonal from primary furniture reference; inspect supplementary-asset comparison before calling it hallucination.
- Incomplete/detail loss: 13.8% of sampled reference furniture surface lies farther than 1% diagonal from prediction.
- Held-out extra silhouettes: 22.7% of predicted foreground lies on reference background.
- Object 3 (Cabinet_Shelf_Desk_e314cd3c-e309-4614-98d2-13f99208ced8_3.obj): recall@1% diagonal 76.1% (44132 area samples).
- Object 4 (Lighting_ec122679-f54a-4344-84da-7de8bb4cd4ff_1.obj): recall@1% diagonal 53.6% (1530 area samples).
- Object 5 (Lighting_41dce00c-6a26-4480-a569-4ab8e6e74d67_5.obj): recall@1% diagonal 73.7% (1877 area samples).

| Object | Samples | Visible | Recall @0.5% D | Recall @1% D | Recall @2% D | P95 / D |
|---|---:|---:|---:|---:|---:|---:|
| Cabinet [0] | 3511 | 63.0% | 77.4% | 96.0% | 99.9% | 0.00851 |
| Lighting [1] | 1892 | 96.0% | 54.1% | 81.6% | 99.4% | 0.01445 |
| Bed [2] | 47058 | 38.2% | 85.6% | 96.8% | 100.0% | 0.00818 |
| Cabinet [3] | 44132 | 56.9% | 62.9% | 76.1% | 87.3% | 0.07424 |
| Lighting [4] | 1530 | 88.0% | 22.0% | 53.6% | 85.5% | 0.02506 |
| Lighting [5] | 1877 | 97.2% | 20.2% | 73.7% | 100.0% | 0.01563 |

Additional-asset context (ceil.glb): 6.6% of predicted surface is farther than 1% D from furniture but within 1% D of supplementary released assets in their supplied coordinates. These can include architecture and unlabeled `others.glb` geometry. This is a proximity diagnostic; the primary furniture score is unchanged.

Export check: preview has 987,942 triangles. Raw → preview mean distance / D = 0.00003; P95 / D = 0.00009. Preview precision/recall @1% D = 83.1%/86.2%. Primary scores above use raw geometry.

### scene_02: e9b1fe99-242b-46e5-85c5-59028015310d/LivingRoom-11414

Inputs: 0000, 0001, 0003, 0004, 0006, 0007, 0009, 0010. Held out: 0002, 0005, 0008.
Effective RGB conditioning: 0003, 0004, 0006, 0007, 0009, 0010. Native chunks: 9; image-selection fallback warnings: 6.
Reference: 32,352 triangles; raw prediction: 6,253,168 triangles.

[Held-out diagnostics](benchmark_results/scene_02/heldout_views.png) · [Surface error map](benchmark_results/scene_02/surface_errors.png) · [Full metrics](benchmark_results/scene_02/metrics.json) · [Objects CSV](benchmark_results/scene_02/objects.csv)

[Side-by-side textured GLB](../../../../outputs/3dfront_benchmark/scene_02/comparison_textured.glb) — reference furniture left, preview prediction right; display translation only. [All supplied reference assets](../../../../outputs/3dfront_benchmark/scene_02/reference/full_room.glb).

- Native camera filtering dropped 2 of eight supplied RGB views: 0000, 0001. Depth support still uses the eight supplied views.
- Native image selector reported 6 camera-selection fallback warnings; inspect reconstruction.log and cameras.json.
- Unmatched furniture-target geometry: 41.5% of sampled predicted surface lies farther than 1% diagonal from primary furniture reference; inspect supplementary-asset comparison before calling it hallucination.
- Held-out extra silhouettes: 65.0% of predicted foreground lies on reference background.

| Object | Samples | Visible | Recall @0.5% D | Recall @1% D | Recall @2% D | P95 / D |
|---|---:|---:|---:|---:|---:|---:|
| Sofa [0] | 29185 | 41.5% | 82.9% | 99.7% | 100.0% | 0.00746 |
| Cabinet [1] | 525 | 97.7% | 59.0% | 86.9% | 96.0% | 0.01406 |
| Cabinet [2] | 63112 | 59.6% | 64.7% | 95.6% | 100.0% | 0.00981 |
| Cabinet [3] | 571 | 99.8% | 67.4% | 95.4% | 100.0% | 0.00975 |
| Cabinet [4] | 6607 | 71.0% | 95.6% | 99.7% | 100.0% | 0.00474 |

Additional-asset context (others.glb): 21.2% of predicted surface is farther than 1% D from furniture but within 1% D of supplementary released assets in their supplied coordinates. These can include architecture and unlabeled `others.glb` geometry. This is a proximity diagnostic; the primary furniture score is unchanged.

Export check: preview has 1,989,269 triangles. Raw → preview mean distance / D = 0.00001; P95 / D = 0.00003. Preview precision/recall @1% D = 57.8%/97.0%. Primary scores above use raw geometry.

Execution telemetry: 2208.3 seconds including subprocess startup/model loading; sampled peak process GPU allocation 3870.0 MiB. Samples every 2 seconds may miss peaks; shared-device timings are not isolated performance measurements.

### scene_03: e6db34b3-c6a1-4e04-b95a-30c415767e82/Library-27940

Inputs: 0000, 0001, 0003, 0004, 0006, 0007, 0009, 0010. Held out: 0002, 0005, 0008.
Effective RGB conditioning: 0000, 0001, 0003, 0004, 0006, 0007, 0009, 0010. Native chunks: 6; image-selection fallback warnings: 0.
Reference: 31,478 triangles; raw prediction: 5,010,190 triangles.

[Held-out diagnostics](benchmark_results/scene_03/heldout_views.png) · [Surface error map](benchmark_results/scene_03/surface_errors.png) · [Full metrics](benchmark_results/scene_03/metrics.json) · [Objects CSV](benchmark_results/scene_03/objects.csv)

[Side-by-side textured GLB](../../../../outputs/3dfront_benchmark/scene_03/comparison_textured.glb) — reference furniture left, preview prediction right; display translation only. [All supplied reference assets](../../../../outputs/3dfront_benchmark/scene_03/reference/full_room.glb).

- Unmatched furniture-target geometry: 63.0% of sampled predicted surface lies farther than 1% diagonal from primary furniture reference; inspect supplementary-asset comparison before calling it hallucination.
- Incomplete/detail loss: 22.8% of sampled reference furniture surface lies farther than 1% diagonal from prediction.
- Held-out extra silhouettes: 72.0% of predicted foreground lies on reference background.
- Object 0 (Table_08bd9700-ae85-4722-8a3c-bf394a31799f_3.obj): recall@1% diagonal 74.0% (82327 area samples).

| Object | Samples | Visible | Recall @0.5% D | Recall @1% D | Recall @2% D | P95 / D |
|---|---:|---:|---:|---:|---:|---:|
| Table [0] | 82327 | 47.3% | 61.6% | 74.0% | 88.5% | 0.02677 |
| Chair [1] | 4760 | 68.6% | 54.8% | 88.4% | 100.0% | 0.01223 |
| Sofa [2] | 9181 | 57.9% | 83.1% | 92.3% | 98.6% | 0.01451 |
| Lighting [3] | 3732 | 84.5% | 56.2% | 97.8% | 100.0% | 0.00909 |

Additional-asset context (ceil.glb, wall.glb, floor.glb, others.glb): 35.3% of predicted surface is farther than 1% D from furniture but within 1% D of supplementary released assets in their supplied coordinates. These can include architecture and unlabeled `others.glb` geometry. This is a proximity diagnostic; the primary furniture score is unchanged.

Export check: preview has 1,472,835 triangles. Raw → preview mean distance / D = 0.00006; P95 / D = 0.00045. Preview precision/recall @1% D = 32.4%/77.2%. Primary scores above use raw geometry.

Execution telemetry: 331.0 seconds including subprocess startup/model loading; sampled peak process GPU allocation 3538.0 MiB. Samples every 2 seconds may miss peaks; shared-device timings are not isolated performance measurements.

### scene_04: fd987ee8-eeac-4d95-a947-306db2feada5/Bedroom-18956

Inputs: 0000, 0001, 0003, 0004, 0006, 0007, 0009, 0010. Held out: 0002, 0005, 0008.
Effective RGB conditioning: 0000, 0001, 0003, 0004, 0006, 0007, 0009, 0010. Native chunks: 2; image-selection fallback warnings: 0.
Reference: 3,288 triangles; raw prediction: 1,146,258 triangles.

[Held-out diagnostics](benchmark_results/scene_04/heldout_views.png) · [Surface error map](benchmark_results/scene_04/surface_errors.png) · [Full metrics](benchmark_results/scene_04/metrics.json) · [Objects CSV](benchmark_results/scene_04/objects.csv)

[Side-by-side textured GLB](../../../../outputs/3dfront_benchmark/scene_04/comparison_textured.glb) — reference furniture left, preview prediction right; display translation only. [All supplied reference assets](../../../../outputs/3dfront_benchmark/scene_04/reference/full_room.glb).

- Unmatched furniture-target geometry: 30.4% of sampled predicted surface lies farther than 1% diagonal from primary furniture reference; inspect supplementary-asset comparison before calling it hallucination.
- Incomplete/detail loss: 28.2% of sampled reference furniture surface lies farther than 1% diagonal from prediction.
- Object 1 (Cabinet_Shelf_Desk_06d5595f-679e-423c-b011-9619edaaac60_1.obj): recall@1% diagonal 66.9% (83976 area samples).

| Object | Samples | Visible | Recall @0.5% D | Recall @1% D | Recall @2% D | P95 / D |
|---|---:|---:|---:|---:|---:|---:|
| Table [0] | 9932 | 57.2% | 87.8% | 98.4% | 100.0% | 0.00779 |
| Cabinet [1] | 83976 | 49.1% | 56.3% | 66.9% | 79.2% | 0.04015 |
| Lighting [2] | 6092 | 75.0% | 69.4% | 96.7% | 100.0% | 0.00891 |

Additional-asset context (ceil.glb, wall.glb, floor.glb, others.glb): 3.0% of predicted surface is farther than 1% D from furniture but within 1% D of supplementary released assets in their supplied coordinates. These can include architecture and unlabeled `others.glb` geometry. This is a proximity diagnostic; the primary furniture score is unchanged.

Export check: preview has 474,395 triangles. Raw → preview mean distance / D = 0.00002; P95 / D = 0.00007. Preview precision/recall @1% D = 69.4%/71.8%. Primary scores above use raw geometry.

Execution telemetry: 139.9 seconds including subprocess startup/model loading; sampled peak process GPU allocation 3384.0 MiB. Samples every 2 seconds may miss peaks; shared-device timings are not isolated performance measurements.

### scene_05: e1d87fdc-c05f-4cca-8b11-c7d002fbc36b/DiningRoom-27978

Inputs: 0000, 0001, 0003, 0004, 0006, 0007, 0009, 0010. Held out: 0002, 0005, 0008.
Effective RGB conditioning: 0000, 0001, 0003, 0004, 0006, 0007, 0009, 0010. Native chunks: 2; image-selection fallback warnings: 0.
Reference: 15,022 triangles; raw prediction: 1,157,466 triangles.

[Held-out diagnostics](benchmark_results/scene_05/heldout_views.png) · [Surface error map](benchmark_results/scene_05/surface_errors.png) · [Full metrics](benchmark_results/scene_05/metrics.json) · [Objects CSV](benchmark_results/scene_05/objects.csv)

[Side-by-side textured GLB](../../../../outputs/3dfront_benchmark/scene_05/comparison_textured.glb) — reference furniture left, preview prediction right; display translation only. [All supplied reference assets](../../../../outputs/3dfront_benchmark/scene_05/reference/full_room.glb).

- Unmatched furniture-target geometry: 70.1% of sampled predicted surface lies farther than 1% diagonal from primary furniture reference; inspect supplementary-asset comparison before calling it hallucination.
- Incomplete/detail loss: 17.0% of sampled reference furniture surface lies farther than 1% diagonal from prediction.
- Held-out holes: 9.5% of reference foreground has no predicted depth.
- Held-out extra silhouettes: 80.1% of predicted foreground lies on reference background.
- Object 0 (Lighting_85a68385-4bef-4aa2-9469-eedff668a91a_2.obj): recall@1% diagonal 69.9% (30915 area samples).

| Object | Samples | Visible | Recall @0.5% D | Recall @1% D | Recall @2% D | P95 / D |
|---|---:|---:|---:|---:|---:|---:|
| Lighting [0] | 30915 | 66.2% | 41.6% | 69.9% | 94.0% | 0.02092 |
| Cabinet [1] | 69085 | 74.9% | 73.9% | 88.9% | 100.0% | 0.01082 |

Additional-asset context (ceil.glb, wall.glb, floor.glb, others.glb): 56.9% of predicted surface is farther than 1% D from furniture but within 1% D of supplementary released assets in their supplied coordinates. These can include architecture and unlabeled `others.glb` geometry. This is a proximity diagnostic; the primary furniture score is unchanged.

Export check: preview has 482,390 triangles. Raw → preview mean distance / D = 0.00002; P95 / D = 0.00004. Preview precision/recall @1% D = 28.9%/83.0%. Primary scores above use raw geometry.

Execution telemetry: 103.2 seconds including subprocess startup/model loading; sampled peak process GPU allocation 3368.0 MiB. Samples every 2 seconds may miss peaks; shared-device timings are not isolated performance measurements.

## Reproduction and limitations

Run `python -m experiments.3dfront_validation.preparation.prepare_benchmark`, prepare each eight-view input with `python -m experiments.3dfront_validation.preparation.prepare_genrecon_input`, then `python -m experiments.3dfront_validation.inference.run_benchmark` in the `genrecon` environment. `data/3dfront/benchmark/protocol.json` freezes room selection and settings. Per-scene progress histories, source/checkpoint fingerprints, inference/export logs and full JSON metrics are saved under `outputs/`. `BENCHMARK_METRICS.csv` retains the summary in `reports/benchmark/`.

The reference triangle/depth consistency gate checks every released view (≥20 paired rays, IoU ≥0.95, median relative depth error <0.01). No additional scene normalization or registration is fitted to predictions. There are no centimetre claims. Surface distances and held-out geometry do not measure texture quality, semantic correctness or physical support relationships. High recall does not by itself prove object identity or separation.

This baseline holds the model and eight-view selection rule fixed across rooms. It does not yet establish causality for observed failures. Targeted view-coverage ablations and chunk-boundary attribution require separate controlled runs. Shared GPU workloads confound wall-clock timing; sampled per-process GPU memory telemetry is diagnostic and may miss brief peaks.
