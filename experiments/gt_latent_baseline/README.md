# Ground-truth latent reconstruction baseline

Encode and decode reference scene geometry with the pretrained GenRecon VAEs;
optionally reconstruct PBR materials. Dataset selections and measured results
belong in [`reports/`](reports/RESULTS.md).

Run from the repository root in the
[GenRecon environment](../3dfront_validation/environment/README.md).

## Inputs and settings

The current runner uses the 3D-FRONT reference export at
`outputs/3dfront_benchmark/<scene>/reference/{full_room,furniture_only}.glb`.
Generate these with the [reconstruction benchmark](../3dfront_validation/README.md).
See [settings](configs/README.md) for scene selection, checkpoints and output paths.
Other selections, including ScanNet/ScanNet++, require adapting input paths,
coordinate transforms, chunking and optional semantic/material inputs; there is
currently no dataset-selection CLI.

## Run

```bash
python -m experiments.gt_latent_baseline.reconstruction.run_baseline run
python -m experiments.gt_latent_baseline.evaluation.verify_artifacts
python -m experiments.gt_latent_baseline.evaluation.analyze_boundaries
python -m experiments.gt_latent_baseline.evaluation.summarize_results
```

Optional PBR reconstruction:

```bash
python -m experiments.gt_latent_baseline.reconstruction.run_pbr_baseline run
python -m experiments.gt_latent_baseline.evaluation.verify_pbr
python -m experiments.gt_latent_baseline.evaluation.summarize_results
```

Render any scene or chunk output directory:

```bash
python -m experiments.gt_latent_baseline.visualization.render_comparison outputs/gt_latent_baseline/<scene>
python -m experiments.gt_latent_baseline.visualization.render_pbr_comparison outputs/gt_latent_baseline/<scene>
```

Outputs: `outputs/gt_latent_baseline/` (meshes, latents, transforms, metrics and
logs). The summarizer writes `reports/RESULTS.md`. Completed stages are reused;
choose a fresh `OUT` in `common.py` when inputs or settings change. Preserve each
selection's report before generating another.
