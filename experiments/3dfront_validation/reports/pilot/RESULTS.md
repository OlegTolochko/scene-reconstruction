# Pilot validation results — 2026-10-27

For the five-scene GenRecon reconstruction evaluation, see
[BENCHMARK_RESULTS.md](../benchmark/BENCHMARK_RESULTS.md). This file records the initial
dataset and camera/depth validation.

## Outcome

The isolated furniture views, cameras, and combined furniture mesh are
geometrically consistent after resolving the camera/image indexing and axis
conventions. This is useful data for furniture reconstruction diagnostics.
It is not yet a ready-to-run metric indoor-room benchmark: the supplied views
omit the room shell, and conversion from dataset coordinates to original
metres has not been established.

No GenRecon inference, view selection experiment, or benchmark evaluation was
run. This completes the one-room inspection in step 1, with the limitations
below carried forward for review before step 2.

## Source and selection

- Dataset: `huanngzh/3D-Front`, the MIDI-3D processed release.
- Revision: `156fc5312ea613d4e92cbf5da1fddfa61bbad3b2`.
- Room: `fe6717d3-31e0-41a2-83e6-98c29c3461df/MasterBedroom-2820`.
- Selected for cheap access near the start of both test archives, before any
  reconstruction results were available.
- Membership in `midi_test_room_ids.json` verified. Disjointness from
  GenRecon's training houses remains unknown.

## Contents

There are 11 RGB views at 768 × 768, corresponding depth EXRs, normals,
semantic PNGs, alternative RGB `rerender` files, and `meta.json`.

The combined room GLB has 76,209 vertices and 59,391 triangles, with six
furniture/light geometries: one bed, two cabinet/shelf/desk assets, and three
lighting assets. Its stored bounds span approximately
`1.33116 × 1.31765 × 1.90000` dataset units; the second axis is up.

The component directory also contains a separate `ceil.glb` (462 triangles),
which is absent from the combined GLB. Visual inspection shows a room shell
with wall, floor/ceiling surfaces and an opening, rather than just a ceiling.
It is not watertight and has no useful per-surface semantic labels in this
export. Its surfaces would need further inspection before scoring individual
architectural elements. The supplied furniture renders omit this shell;
indoor camera views including it would need to be rendered separately.

## Camera and depth consistency

The passing interpretation is:

1. Pair camera `locations[i]` using its explicit `index`: for example,
   index `0000` maps to `render_0000.webp` and `depth_0000.exr`.
2. Treat the camera transform as Blender/OpenGL camera-to-world: camera
   right is +x, up is +y, and forward is -z.
3. Transform stored mesh vertices from glTF y-up to camera-world z-up:
   `(x, y, z) -> (x, -z, y)`.
4. Derive horizontal focal length as `W / (2 * tan(camera_angle_x / 2))`;
   `camera_angle_x = 0.6911111611634243` radians. Square pixels and a centered
   principal point agree with the sampled depth checks.
5. Interpret EXR depth as camera-axis depth, not Euclidean ray distance.
6. Mask background value `65504` (float16 maximum); it is not valid depth.

Each view was checked at a 32 × 32 pixel grid using mesh ray intersections.
Across all views, 2,944 rays had both a valid released depth and a mesh hit.

| Hypothesis | Views available | Mean sampled silhouette IoU | Median of per-view median relative depth errors |
| --- | ---: | ---: | ---: |
| Sequential index, converted axes, camera-axis depth | 11 | 0.99699 | 0.000352 (0.0352%) |
| Sequential index, converted axes, ray distance | 11 | 0.99699 | 0.014718 (1.47%) |
| Literal metadata frame names, converted axes, camera-axis depth | 7 | 0.52189 | 0.076713 (7.67%) |
| Sequential index, stored axes, camera-axis depth | 11 | 0.41650 | 0.225712 (22.6%) |

The consistency check requires every view to have at least 20 paired rays,
silhouette IoU >= 0.95, and median relative depth error < 1%. All 11 pass.
This is a sparse dataset-consistency check; it does not establish subpixel
accuracy, material correctness, or reconstruction quality.

### Metadata issue

The metadata frame names differ from the actual sequential file numbering.
For example, the first camera has index `0000` but lists `depth_0003.exr`.
Four referenced depth filenames do not exist (`0011`, `0013`, `0015`, `0017`),
and all referenced `render_opaque` files are absent. There are nevertheless
11 complete sequential RGB/depth pairs. The numerical checks above support
the sequential pairing. Source metadata has not been modified.

## Scale and evaluation scope

The centered bounds and maximum extent of 1.9 suggest normalized geometry.
The retained metadata and GLB do not establish a conversion to original
metric scale. Camera/depth/mesh agreement verifies consistency in the same
coordinate system, not that these units are metres.

Do not report centimetre thresholds or metre errors for this sample yet.
Recover original scale from source scene information, or explicitly define
a normalized-coordinate protocol. An assumed room height is an assumption,
not verified metric ground truth.

The supplied views support furniture reconstruction/occlusion tests. Room
layout and opening tests require assembling the separate shell and furniture,
validating that combination, and producing interior views. This qualifies the
earlier recommendation to use the processed release without rendering work.

## Storage and reproducibility

- Retained dataset: 64 files, **14,762,477 bytes (14.08 MiB)**.
- Compressed archive bytes read: **37,457,920 bytes (35.72 MiB)** combined.
- Full archives stored: **none**. No Hugging Face cache was created.
- New Miniforge installation plus validation environment/cache: approximately
  **1.4 GiB allocated**, measured with `du -sh` after installation.
- Validation uses CPU tools only; no PyTorch, CUDA toolkit, or checkpoints.
- File hashes and download statistics: `data/3dfront/pilot/manifest.json`.
- Exact installed environment: `environment-resolved.yml` (Linux x86_64).

Generated artifacts (ignored by Git):

- `outputs/3dfront_validation/report.json`: inventory and all numerical checks.
- `outputs/3dfront_validation/rgb_contact_sheet.png`: all 11 RGB views.
- `outputs/3dfront_validation/camera_alignment.png`: RGB, depth, and ray overlay.
- `outputs/3dfront_validation/structural_asset.png`: separate room shell.
- `outputs/3dfront_validation/validation.log`: validator output.

Run the commands in [the experiment guide](../../README.md) to reproduce the preparation and validation.
