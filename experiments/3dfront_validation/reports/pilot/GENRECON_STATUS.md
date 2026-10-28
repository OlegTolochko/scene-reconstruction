# GenRecon pilot status — 2026-10-27

The subsequent five-scene evaluation is documented in
[BENCHMARK_RESULTS.md](../benchmark/BENCHMARK_RESULTS.md) and
[BENCHMARK_FINDINGS.md](../benchmark/BENCHMARK_FINDINGS.md).

## Completed: one supplied-camera reconstruction

GenRecon ran successfully on the pilot with **eight RGB views, four chunks,
seed 42, and 10,000 depth-derived support points**. All three generation stages
and mesh decoding completed. The raw result has **2,763,485 vertices and
5,677,912 triangles**. This is a smoke run; reconstruction accuracy has not
been benchmarked.

Artifacts under `outputs/genrecon/3dfront_pilot/`:

| File | Purpose |
| --- | --- |
| `comparison_textured_preview.glb` | Reference furniture left, simplified textured prediction right |
| `scene_preview_reference_frame.glb` | Textured prediction alone: 987,942 triangles, about 54 MiB |
| `geometry_only_reference_frame.glb` | Exact raw generated geometry, untextured: about 107 MiB |
| `comparison_geometry_only.glb` | Reference furniture left, exact raw prediction right |
| `mesh.ply` | Raw generated geometry in inference z-up coordinates |
| `to_glb_inputs.pt`, `chunk_inputs.pt` | Saved geometry/material inputs for re-export without rerunning inference |

The inspection GLBs use the reference glTF y-up coordinates and dataset scale.
The native textured exporter already converts axes; raw PLY needs the z-up to
y-up conversion. No additional registration or scale fitting was applied.
Only the comparison files translate the prediction horizontally for display.

## Export settings and resource observations

Default global remeshing at resolution 896 ran out of available GPU memory.
A competing workload was observed holding **24,874 MiB** on the shared RTX 5090.
This does **not** establish that export requires more than an isolated 32 GiB
GPU. The other workload was left untouched.

Export without global remeshing proceeded, but dense-mesh UV unwrapping was
slow. That attempt was interrupted in favor of a separately labeled preview:

- Global remeshing disabled.
- Per-chunk simplification target: 250,000 triangles.
- Four 2048 × 2048 base-color and metallic/roughness texture sets.
- Standard per-chunk cleanup and global hole filling retained.
- Preview conversion completed in approximately 98 seconds, excluding the
  earlier attempts. The raw geometry is retained independently of preview edits.

The preview is for inspection. Use the raw mesh for subsequent geometry
evaluation. Robust peak VRAM was not measured: shared workloads changed during
the run, and the printed PyTorch memory counters report zero with the selected
non-caching allocator, despite real GPU use.

## Verification and provenance

- GLBs reopened successfully; geometry is nonempty and finite.
- Preview and raw bounds differ by at most `0.000297` dataset units, a frame
  sanity check rather than an accuracy measurement.
- Comparison export preserves face counts, bounds after display translation,
  and generated base-color / metallic-roughness texture pixels.
- Checkpoint SHA-256 hashes and Hugging Face model revisions are recorded in
  `run_manifest.json`.
- Settings, input image IDs, checks and output metadata are recorded alongside
  the result. Logs retain the remeshing OOM and interrupted dense UV attempt.
- Source metric scale remains unknown. Input cameras and the support cloud are
  supplied references; this is not RGB-only calibration.
- Supplied RGB views omit the architectural shell. The comparison uses the
  reference furniture, not the separate structural asset.

Storage after completion is approximately **28 GiB** across the Miniforge
installation/environments (11 GiB), checkpoints (13 GiB), Hugging Face cache
(3.1 GiB), pilot data (17 MiB), and outputs (690 MiB), using rounded `du` values.
This is below the requested 50 GB budget.

## Completed setup

- Separate Conda environment: `genrecon`.
- PyTorch 2.9.1 + CUDA 13.0; torchvision 0.24.1.
- CUDA 13.0 compiler installed inside the environment; system CUDA unchanged.
- Pinned FlashAttention, nvdiffrast, nvdiffrec_render, CuMesh, FlexGEMM, and
  o-voxel installed; all extension imports passed.
- GPU execution checks passed: PyTorch matrix multiplication, FlashAttention,
  CUDA rasterization, and CuMesh initialization on the RTX 5090.
- Eight input images/cameras and 10,000 visible depth-derived support points
  prepared at `data/3dfront/genrecon_pilot/`.
- GenRecon's actual input reader verified: eight conditioning images, four
  retained chunks, maximum camera matrix discrepancy `4.77e-7`.
- Source metric scale remains unknown. This run uses supplied cameras and
  depth-derived support points, rather than estimating calibration from RGB.

All three GenRecon checkpoints are cached. Their total size is
13,684,232,356 bytes (approximately 13.7 GB). Completed ranges are
written directly to the checkpoint files and tracked in small sidecars;
interrupted transfers resume without keeping duplicate model archives.

DINOv3 is a separate frozen image encoder. GenRecon's generation checkpoints
do not replace it: `genrecon/modules/image_feature_extractor.py` explicitly
loads it via `DINOv3ViTModel.from_pretrained(...)`.

## Repeat with the preview settings

From the repository root:

```bash
source ~/.local/share/miniforge3/etc/profile.d/conda.sh
conda activate genrecon
python -m experiments.3dfront_validation.inference.run_genrecon \
    --output outputs/genrecon/3dfront_pilot_repeat \
    --skip-remesh --simplify-threshold 250000 --texture-size 2048
```

Cached weights and the saved login are reused. A new output directory avoids
mixing results with previous runs. The runner exports `scene_reference_frame.glb`
inside that directory. Standalone re-export can use `chunked_to_glb.py` with
the saved `.pt` inputs; see `run_manifest.json` for the successful export command.

Setup versions, checks, and installation logs are saved under
`outputs/genrecon_setup/`, including `setup_report.json` and
`input_check/check.json`. These are ignored by Git.
