"""Write the measured baseline report and CSV from completed artifacts."""
import csv
from pathlib import Path

from ..common import ROOT, OUT, fingerprint, read, write


def scene_selection(scene_ids):
    protocol = read(ROOT / 'experiments/3dfront_validation/configs/BENCHMARK_PROTOCOL.json')
    scenes = {scene['id']: scene['room'] for scene in protocol['scenes']}
    lines = [
        '## Dataset and scene selection', '',
        f"Dataset: `huanngzh/3D-Front` (processed MIDI test release), revision `{protocol['revision']}`.",
        'Supplied full-room references from the five-house convenience sample; aliases are local to this run. '
        'Inputs include available furniture, floor, wall, ceiling and `others` assets.', '',
        '| Run alias | Source scene ID (house UUID / room ID) |',
        '| --- | --- |',
    ]
    for scene_id in scene_ids:
        lines.append(f'| {scene_id} | `{scenes[scene_id]}` |')
    lines += ['', 'References: `outputs/3dfront_benchmark/<alias>/reference/full_room.glb`; '
              'furniture subsets: `furniture_only.glb` in the same directory.', '']
    return lines


TARGET_GENERATION = """## How GenRecon constructs its targets

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
`strict=True`. No architecture or model math is changed."""


def main():
    summaries = read(OUT / 'summary.json')
    verification = read(OUT / 'verification.json')
    assert verification['status'] == 'passed'
    has_pbr = (OUT / 'pbr_summary.json').exists()
    rows = []
    for scene_id, m in summaries.items():
        edge = read(OUT / scene_id / 'chunks.json')['edge_source_units']
        voxel = edge / 512
        p = m['prediction_to_reference']
        r = m['reference_to_prediction']
        rows.append(dict(scene=scene_id, chunks=m['chunk_count'],
            decoded_triangles=m['prediction_faces'],
            voxel_size_source_units=voxel,
            precision_1_voxel=m['scores']['1_voxels']['precision'],
            recall_1_voxel=m['scores']['1_voxels']['recall'],
            fscore_1_voxel=m['scores']['1_voxels']['fscore'],
            mean_symmetric_error_voxels=m['symmetric_mean_source_units'] / voxel,
            decoded_to_gt_p95_voxels=p['p95_voxels'],
            gt_to_decoded_p95_voxels=r['p95_voxels'],
            voxel_control_mean_symmetric_error_voxels=m['preprocessing_control']['symmetric_mean_source_units'] / voxel,
            voxel_to_decoded_mean_symmetric_error_voxels=m['voxel_to_decoded']['symmetric_mean_source_units'] / voxel,
            furniture_recall_1_voxel=m['furniture_recall_at_1_2_4_voxels']['1'],
            furniture_to_decoded_mean_voxels=m['furniture_reference_to_prediction']['mean_voxels'],
            furniture_to_decoded_p95_voxels=m['furniture_reference_to_prediction']['p95_voxels']))
    with (OUT / 'metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    text = ['# GT latent baseline — 3D-FRONT five-room selection', '',
        'Completed on all five supplied 3D-FRONT benchmark rooms at **512 geometry resolution**. '
        'Original geometry was voxelized and encoded with GenRecon’s `get_slats.process_instance`, '
        'then reconstructed with its pretrained shape decoder using **raw saved features and their exact sparse coordinates**. '
        'The shape measurements used no latent denormalization, DiT, image conditioning, camera estimation, texture, training, or ICP.', '',
        'The working example was scene_04/chunk_000, containing shelving and room structure. '
        'Its saved transforms, numerical gate, original geometry, latent NPZ, voxel control and decoded mesh '
        'were established before processing the other chunks.', '',
        *scene_selection(summaries),
        '| Scene | Chunks | Precision @1 voxel | Recall @1 voxel | F-score @1 voxel | Mean symmetric error, voxels | Furniture recall @1 voxel |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        text.append(f"| {row['scene']} | {row['chunks']} | {row['precision_1_voxel']:.3%} | {row['recall_1_voxel']:.3%} | "
                    f"{row['fscore_1_voxel']:.3%} | {row['mean_symmetric_error_voxels']:.5f} | {row['furniture_recall_1_voxel']:.3%} |")
    text += ['', f"Across the five rooms, F-score at one voxel ranges from {min(r['fscore_1_voxel'] for r in rows):.3%} "
        f"to {max(r['fscore_1_voxel'] for r in rows):.3%}; mean symmetric error ranges from "
        f"{min(r['mean_symmetric_error_voxels'] for r in rows):.5f} to {max(r['mean_symmetric_error_voxels'] for r in rows):.5f} voxels. "
        'These measurements show high surface fidelity for the supplied geometry when the decoder receives oracle shape latents and sparse coordinates. '
        'They do not establish equivalent fidelity for generated latents or predicted coordinates.']
    text += ['', 'Means average the two directed mean unsigned nearest-surface distances. '
        'Each direction uses 100,000 area-weighted samples and exact triangle queries. '
        'One voxel is each scene’s recorded chunk edge divided by 512. '
        'Physical metric scale is unverified; no centimetre interpretation is claimed.', '',
        '| Scene | Original → voxel control symmetric mean, voxels | Voxel control → decoded symmetric mean, voxels | GT → decoded P95, voxels | Furniture → decoded P95, voxels |',
        '|---|---:|---:|---:|---:|']
    for row in rows:
        text.append(f"| {row['scene']} | {row['voxel_control_mean_symmetric_error_voxels']:.5f} | "
                    f"{row['voxel_to_decoded_mean_symmetric_error_voxels']:.5f} | {row['gt_to_decoded_p95_voxels']:.5f} | "
                    f"{row['furniture_to_decoded_p95_voxels']:.5f} |")
    text += ['', 'The voxel control uses the exact quantized encoder inputs and bypasses the VAE. '
        'It separates preprocessing from compression/decoding approximately; the control and decoder use different quad triangulation rules, '
        'and these distances are not additive.']
    boundary = read(OUT / 'boundary_diagnostics.json')
    text += ['', 'Small gaps are visible at chunk seams. The following diagnostic checks reference samples '
        'within two voxel widths of internal chunk planes; the original full-room scores retain all gaps.', '',
        '| Scene | Recall @1 voxel away from seams | Share of missed GT samples near seams |',
        '|---|---:|---:|']
    for sid, diagnostic in boundary.items():
        d = diagnostic['stages']['decoded']
        share = d['fraction_of_misses_near_internal_seams']
        text.append(f"| {sid} | {d['recall_1_voxel_away_from_internal_seams']:.3%} | {share:.1%} |" if share is not None
                    else f"| {sid} | {d['recall_1_voxel_away_from_internal_seams']:.3%} | no missed samples |")
    text += ['', 'This is a proximity diagnostic, not a causal correction. '
        '[Full boundary diagnostics](../../../outputs/gt_latent_baseline/boundary_diagnostics.json) include the voxel-control comparison.', '']
    if has_pbr:
        text += ['The main comparison GLBs show the **original source materials on the left** and '
            '**decoded ground-truth PBR materials on the exact reconstructed geometry on the right**. '
            'This later visualization extension encodes raw GT texture latents with the pretrained PBR encoder and '
            'decodes them using the saved shape subdivision guides, without DiTs or latent denormalization. '
            'Base-color/alpha and metallic/roughness maps are baked at 4096². '
            'Every triangle is preserved exactly through UV export; only UV seam vertex duplication and face reindexing are allowed. '
            'No simplification, hole filling or remeshing is applied. The shape scores above are unchanged.', '',
            '[PBR provenance](../../../outputs/gt_latent_baseline/pbr_provenance.json) · '
            '[PBR artifact verification](../../../outputs/gt_latent_baseline/pbr_verification.json)', '']
    text += ['Only a display translation is applied in comparisons; all metrics use the original common frame. '
        'Surface-error colors are available separately in `comparison_surface_error.glb`: dark blue = zero; red = four or more voxel widths. '
        'The overlay files preserve the exact common frame, with GT blue and reconstruction orange.', '']
    for row in rows:
        sid = row['scene']
        text.append(f'- {sid}: [comparison GLB](../../../outputs/gt_latent_baseline/{sid}/comparison.glb) · '
                    f'[surface-error GLB](../../../outputs/gt_latent_baseline/{sid}/comparison_surface_error.glb) · '
                    f'[same-frame overlay](../../../outputs/gt_latent_baseline/{sid}/overlay.glb) · '
                    f'[cutaway preview](../../../outputs/gt_latent_baseline/{sid}/comparison{"_pbr" if has_pbr else ""}.png) · '
                    f'[metrics](../../../outputs/gt_latent_baseline/{sid}/metrics.json)')
    text += ['', f'[Working-example preview](../../../outputs/gt_latent_baseline/scene_04/chunk_000/comparison{"_pbr" if has_pbr else ""}.png) · '
        '[Working-example latents](../../../outputs/gt_latent_baseline/scene_04/chunk_000/shape_raw.npz) · '
        '[Checkpoint/source provenance](../../../outputs/gt_latent_baseline/provenance.json) · '
        '[Independent artifact verification](../../../outputs/gt_latent_baseline/verification.json) · '
        '[CSV](../../../outputs/gt_latent_baseline/metrics.csv)', '',
        'The independent verifier checked all source/checkpoint fingerprints, latent hashes and schemas, '
        'exact decoder input coordinates and features, source-to-normalized axis/scale transforms, '
        'raw decoded mesh-to-GLB vertex transforms and face identity, and comparison GLB triangle counts. '
        'No simplification, remeshing, alignment fitting, or separate occupancy prediction was applied.', '',
        'This measures an empirical representation ceiling conditional on these supplied targets, chunk scales, '
        'and oracle sparse coordinates. Architectural area dominates some rooms; furniture-only recall is included '
        'to expose detail loss. Unsigned surface distances do not establish topology or watertightness. '
        'Chunks cover all released full-room geometry using deterministic triangle clipping; training uses randomized '
        'Blender Boolean crops with solidification. This test reuses target encoding exactly after cropping, '
        'but does not reproduce that upstream room-building augmentation. '
        'The earlier image-conditioned benchmark used different chunk selection and a furniture-only primary target, '
        'so its numbers are not a matched causal ablation against this experiment.', '',
        'See [README.md](../README.md) for reproduction commands.', '']
    text += [TARGET_GENERATION, '']
    report = ROOT / 'experiments/gt_latent_baseline/reports/RESULTS.md'
    report.write_text('\n'.join(text))
    write(OUT / 'implementation_files.json', {str(p.relative_to(ROOT)): fingerprint(p) for p in (ROOT / 'experiments/gt_latent_baseline').rglob('*.py')})
    print('\n'.join(text[:12]), flush=True)


if __name__ == '__main__':
    main()
