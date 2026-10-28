# GT latent baseline — 3D-FRONT five-room selection

Completed on all five supplied 3D-FRONT benchmark rooms at **512 geometry resolution**. Original geometry was voxelized and encoded with GenRecon’s `get_slats.process_instance`, then reconstructed with its pretrained shape decoder using **raw saved features and their exact sparse coordinates**. No latent denormalization, DiT, image conditioning, camera estimation, texture, training, or ICP was used.

The working example was scene_04/chunk_000, containing shelving and room structure. Its saved transforms, numerical gate, original geometry, latent NPZ, voxel control and decoded mesh were established before processing the other chunks.

## Dataset and scene selection

Dataset: `huanngzh/3D-Front` (processed MIDI test release), revision
`156fc5312ea613d4e92cbf5da1fddfa61bbad3b2`.
These are the supplied full-room references from the five-house convenience
sample; the local aliases below are specific to this run. Inputs include the
available furniture, floor, wall, ceiling and `others` assets.

| Run alias | Source scene ID (house UUID / room ID) |
| --- | --- |
| scene_01 | `fe6717d3-31e0-41a2-83e6-98c29c3461df/MasterBedroom-2820` |
| scene_02 | `e9b1fe99-242b-46e5-85c5-59028015310d/LivingRoom-11414` |
| scene_03 | `e6db34b3-c6a1-4e04-b95a-30c415767e82/Library-27940` |
| scene_04 | `fd987ee8-eeac-4d95-a947-306db2feada5/Bedroom-18956` |
| scene_05 | `e1d87fdc-c05f-4cca-8b11-c7d002fbc36b/DiningRoom-27978` |

Reference files: `outputs/3dfront_benchmark/<alias>/reference/full_room.glb`;
furniture-only subsets use `furniture_only.glb` in the same directories.
The sanity check used `scene_04/chunk_000`, from house
`fd987ee8-eeac-4d95-a947-306db2feada5`, room `Bedroom-18956`.

| Scene | Chunks | Precision @1 voxel | Recall @1 voxel | F-score @1 voxel | Mean symmetric error, voxels | Furniture recall @1 voxel |
|---|---:|---:|---:|---:|---:|---:|
| scene_01 | 4 | 99.983% | 99.987% | 99.985% | 0.01441 | 99.973% |
| scene_02 | 6 | 99.994% | 99.989% | 99.991% | 0.02307 | 99.967% |
| scene_03 | 4 | 99.999% | 99.991% | 99.995% | 0.00900 | 99.982% |
| scene_04 | 4 | 100.000% | 99.926% | 99.963% | 0.01099 | 99.984% |
| scene_05 | 4 | 100.000% | 99.988% | 99.994% | 0.01921 | 99.742% |

Across the five rooms, F-score at one voxel ranges from 99.963% to 99.995%; mean symmetric error ranges from 0.00900 to 0.02307 voxels. These measurements show high surface fidelity for the supplied geometry when the decoder receives oracle shape latents and sparse coordinates. They do not establish equivalent fidelity for generated latents or predicted coordinates.

Means average the two directed mean unsigned nearest-surface distances. Each direction uses 100,000 area-weighted samples and exact triangle queries. One voxel is each scene’s recorded chunk edge divided by 512. Physical metric scale is unverified; no centimetre interpretation is claimed.

| Scene | Original → voxel control symmetric mean, voxels | Voxel control → decoded symmetric mean, voxels | GT → decoded P95, voxels | Furniture → decoded P95, voxels |
|---|---:|---:|---:|---:|
| scene_01 | 0.00570 | 0.01268 | 0.05789 | 0.14204 |
| scene_02 | 0.01376 | 0.01482 | 0.18871 | 0.20622 |
| scene_03 | 0.00433 | 0.00822 | 0.03476 | 0.11967 |
| scene_04 | 0.00488 | 0.00949 | 0.03787 | 0.16111 |
| scene_05 | 0.01357 | 0.00981 | 0.15421 | 0.28382 |

The voxel control uses the exact quantized encoder inputs and bypasses the VAE. It separates preprocessing from compression/decoding approximately; the control and decoder use different quad triangulation rules, and these distances are not additive.

Small gaps are visible at chunk seams. The following diagnostic checks reference samples within two voxel widths of internal chunk planes; the original full-room scores retain all gaps.

| Scene | Recall @1 voxel away from seams | Share of missed GT samples near seams |
|---|---:|---:|
| scene_01 | 99.987% | 0.0% |
| scene_02 | 99.989% | 0.0% |
| scene_03 | 99.991% | 0.0% |
| scene_04 | 99.925% | 0.0% |
| scene_05 | 99.991% | 25.0% |

This is a proximity diagnostic, not a causal correction. [Full boundary diagnostics](../../../outputs/gt_latent_baseline/boundary_diagnostics.json) include the voxel-control comparison.

Comparison GLBs below contain the original on the left and decoded geometry on the right. Only a display translation is applied; all metrics use the original common frame. Vertex colors show distance to the opposite surface: dark blue = zero; red = four or more voxel widths. The overlay files preserve the exact common frame, with GT blue and reconstruction orange.

- scene_01: [comparison GLB](../../../outputs/gt_latent_baseline/scene_01/comparison.glb) · [same-frame overlay](../../../outputs/gt_latent_baseline/scene_01/overlay.glb) · [cutaway preview](../../../outputs/gt_latent_baseline/scene_01/comparison.png) · [metrics](../../../outputs/gt_latent_baseline/scene_01/metrics.json)
- scene_02: [comparison GLB](../../../outputs/gt_latent_baseline/scene_02/comparison.glb) · [same-frame overlay](../../../outputs/gt_latent_baseline/scene_02/overlay.glb) · [cutaway preview](../../../outputs/gt_latent_baseline/scene_02/comparison.png) · [metrics](../../../outputs/gt_latent_baseline/scene_02/metrics.json)
- scene_03: [comparison GLB](../../../outputs/gt_latent_baseline/scene_03/comparison.glb) · [same-frame overlay](../../../outputs/gt_latent_baseline/scene_03/overlay.glb) · [cutaway preview](../../../outputs/gt_latent_baseline/scene_03/comparison.png) · [metrics](../../../outputs/gt_latent_baseline/scene_03/metrics.json)
- scene_04: [comparison GLB](../../../outputs/gt_latent_baseline/scene_04/comparison.glb) · [same-frame overlay](../../../outputs/gt_latent_baseline/scene_04/overlay.glb) · [cutaway preview](../../../outputs/gt_latent_baseline/scene_04/comparison.png) · [metrics](../../../outputs/gt_latent_baseline/scene_04/metrics.json)
- scene_05: [comparison GLB](../../../outputs/gt_latent_baseline/scene_05/comparison.glb) · [same-frame overlay](../../../outputs/gt_latent_baseline/scene_05/overlay.glb) · [cutaway preview](../../../outputs/gt_latent_baseline/scene_05/comparison.png) · [metrics](../../../outputs/gt_latent_baseline/scene_05/metrics.json)

[Working-example preview](../../../outputs/gt_latent_baseline/scene_04/chunk_000/comparison.png) · [Working-example latents](../../../outputs/gt_latent_baseline/scene_04/chunk_000/shape_raw.npz) · [Checkpoint/source provenance](../../../outputs/gt_latent_baseline/provenance.json) · [Independent artifact verification](../../../outputs/gt_latent_baseline/verification.json) · [CSV](../../../outputs/gt_latent_baseline/metrics.csv)

The independent verifier checked all source/checkpoint fingerprints, latent hashes and schemas, exact decoder input coordinates and features, source-to-normalized axis/scale transforms, raw decoded mesh-to-GLB vertex transforms and face identity, and comparison GLB triangle counts. No simplification, remeshing, alignment fitting, or separate occupancy prediction was applied.

This measures an empirical representation ceiling conditional on these supplied targets, chunk scales, and oracle sparse coordinates. Architectural area dominates some rooms; furniture-only recall is included to expose detail loss. Unsigned surface distances do not establish topology or watertightness. Chunks cover all released full-room geometry using deterministic triangle clipping; training uses randomized Blender Boolean crops with solidification. This test reuses target encoding exactly after cropping, but does not reproduce that upstream room-building augmentation. The earlier image-conditioned benchmark used different chunk selection and a furniture-only primary target, so its numbers are not a matched causal ablation against this experiment.

See [README.md](../README.md) for reproduction commands.

## How GenRecon constructs its targets

1. [`create_chunks.py`](../../../papers/GenRecon/data_toolkit_scenes/blender_scripts/create_chunks.py)
   rotates rooms, crops them with Blender booleans, and normalizes the crop cube
   to `[-0.5, 0.5]³`. It solidifies input surfaces by 0.001 before the boolean.
2. [`get_slats.py`](../../../papers/GenRecon/data_toolkit_scenes/get_slats.py)
   loads GLB node transforms and maps glTF Y-up to Blender Z-up with +90° around X.
   `mesh_to_flexible_dual_grid` uses a fixed unit AABB and QEF weights
   `face=1`, `boundary=0.2`, `regularization=0.01`. It sorts voxel sites using
   O-Voxel serialization, converts dual vertices to within-cell offsets,
   clamps them to `[0,1]`, and quantizes them to uint8. The three edge intersection
   flags are packed then unpacked as boolean features.
3. The pretrained `FlexiDualGridVaeEncoder` consumes these six features and
   downsamples spatially by 16. At geometry resolution 512, the saved coordinates
   belong to a **32³** sparse grid, with **32 floating point features per site**.
   The default is the posterior mean, `sample_posterior=False`. Saved NPZ fields
   `feats` (float32) and `coords` (uint8) are the raw representation.
4. [`SLat.get_instance`](../../../papers/GenRecon/genrecon/datasets/structured_latent.py)
   normalizes these raw features only when reading them for DiT training.
   Generated DiT latents are denormalized before decoding. This experiment reads
   the saved encoder features directly, **without either operation**.
5. [`FlexiDualGridVaeDecoder`](../../../papers/GenRecon/genrecon/models/sc_vaes/fdg_vae.py)
   runs in evaluation mode with `set_resolution(512)`. Its learned subdivision
   and edge reconstruction are intrinsic to shape decoding and remain enabled.
   There is no separate occupancy or sparse-structure generation stage. Only
   a zero batch-index column is prepended to the saved coordinates; features,
   coordinate order, and values are checked for exact equality at decoder input.

The encoder and decoder are the TRELLIS.2 shape VAE checkpoints used by GenRecon:
`microsoft/TRELLIS.2-4B/ckpts/shape_{enc,dec}_next_dc_f16c32_fp16`, pinned to
Hugging Face revision `af44b45f2e35a493886929c6d786e563ec68364d`. We instantiate
GenRecon's existing model classes and load checkpoint state dictionaries with
`strict=True`. No architecture or model math is changed.
