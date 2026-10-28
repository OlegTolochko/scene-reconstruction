"""Produce tracked Markdown and CSV summaries from per-scene JSON metrics."""
import csv
import json
import shutil
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/3dfront_benchmark'
SOURCE=ROOT/'experiments/3dfront_validation/reports/benchmark'

def pct(value):return 'n/a' if value is None else f'{100*value:.1f}%'
def num(value):return 'n/a' if value is None else f'{value:.5f}'

def main():
    protocol=json.loads((ROOT/'data/3dfront/benchmark/protocol.json').read_text())
    rows=[]; reports=[]; incomplete=[]
    for scene in protocol['scenes']:
        path=OUT/scene['id']/'metrics.json'
        if not path.exists():
            incomplete.append(scene['id']);continue
        r=json.loads(path.read_text());reports.append((scene,r))
        row={'scene':scene['id'],'room':r['room'],'camera_gate_passed':r['camera_consistency_passed'],
             'effective_rgb_views':len(r.get('effective_conditioning_ids',r['input_ids'])),
             'chunks':r.get('chunk_count'),'camera_selection_fallbacks':len(r.get('camera_selection_fallbacks',[]))}
        for t in ('0.005','0.01','0.02'):
            for k,v in r['threshold_metrics'][t].items():row[f'{k}_{t}']=v
        row.update({'pred_to_gt_mean_D':r['prediction_to_reference']['mean_fraction_diagonal'],
            'gt_to_pred_mean_D':r['reference_to_prediction']['mean_fraction_diagonal'],
            'symmetric_mean_D':r['symmetric_mean_distance_fraction_diagonal'],
            'pred_to_gt_p95_D':r['prediction_to_reference']['p95_fraction_diagonal'],
            'gt_to_pred_p95_D':r['reference_to_prediction']['p95_fraction_diagonal'],
            'visible_recall_01':r['visibility_stratified_recall']['visible']['within_threshold']['0.01'],
            'occluded_recall_01':r['visibility_stratified_recall']['occluded']['within_threshold']['0.01'],
            'effective_rgb_visible_recall_01':r.get('visibility_stratified_recall_effective_conditioning',r['visibility_stratified_recall'])['visible']['within_threshold']['0.01'],
            'effective_rgb_occluded_recall_01':r.get('visibility_stratified_recall_effective_conditioning',r['visibility_stratified_recall'])['occluded']['within_threshold']['0.01'],
            'free_space_violation':r['predicted_surface_input_free_space_violation_fraction'],
            'background_projection':r['predicted_surface_input_background_projection_fraction'],
            'outside_gt_bounds':r['predicted_surface_outside_expanded_gt_bounds_fraction'],
            'heldout_iou':r['heldout']['silhouette_iou'],'heldout_depth_absrel':r['heldout']['depth_absrel'],
            'heldout_depth_mae_D':r['heldout']['depth_mae_fraction_diagonal'],
            'heldout_depth_rmse_D':r['heldout']['depth_rmse_fraction_diagonal'],
            'heldout_depth_coverage':r['heldout']['depth_coverage'],
            'heldout_missing_gt':r['heldout']['missing_fraction_gt'],
            'heldout_extra_pred':r['heldout']['extra_fraction_pred'],
            'correct_depth_gt_1pct':r['heldout']['correct_depth_fraction_all_gt']['0.01'],
            'correct_depth_gt_5pct':r['heldout']['correct_depth_fraction_all_gt']['0.05'],
            'correct_depth_gt_10pct':r['heldout']['correct_depth_fraction_all_gt']['0.1'],
            'raw_triangles':r['prediction_faces']})
        full=r.get('architectural_context',{}).get('all_released_assets_geometry') if r.get('architectural_context') else None
        row.update({'all_assets_precision_01':full['threshold_metrics']['0.01']['precision'] if full else None,
                    'all_assets_recall_01':full['threshold_metrics']['0.01']['recall'] if full else None,
                    'all_assets_fscore_01':full['threshold_metrics']['0.01']['fscore'] if full else None})
        preview=r['preview_comparison']
        if preview:
            pp=preview['prediction_to_reference']['within_threshold']['0.01']
            pr=preview['reference_to_prediction']['within_threshold']['0.01']
            pf=2*pp*pr/(pp+pr) if pp+pr else 0.
        else:pf=None
        row.update({'preview_fscore_01':pf,
            'preview_fscore_delta_percentage_points':100*(pf-r['threshold_metrics']['0.01']['fscore']) if pf is not None else None,
            'raw_to_preview_p95_D':preview['raw_to_preview']['p95_fraction_diagonal'] if preview else None,
            'preview_to_raw_area_ratio':preview.get('surface_area',0)/r['prediction_surface_area'] if preview and preview.get('surface_area') else None})
        rows.append(row)
    OUT.mkdir(parents=True,exist_ok=True)
    if rows:
        with (OUT/'summary.csv').open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
        with (SOURCE/'BENCHMARK_METRICS.csv').open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    macro={k:float(np.mean([r[k] for r in rows if isinstance(r[k],(int,float))]))
        for k in rows[0] if k not in ('scene','room','camera_gate_passed') and any(isinstance(r[k],(int,float)) for r in rows)} if rows else {}
    geometry_complete=len(rows)==5 and all(r['camera_gate_passed'] for r in rows)
    preview_count=sum(r['preview_comparison'] is not None for scene,r in reports)
    summary={'complete':geometry_complete and preview_count==5,'geometry_complete':geometry_complete,
        'preview_comparison_count':preview_count,'scene_count':len(rows),
        'incomplete':incomplete,'macro_scene_mean':macro,'scenes':rows}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    text=['# Five-scene GenRecon diagnostic benchmark','',
        f"Evaluated **{len(rows)}/5** frozen scenes. {'All camera/depth consistency gates passed.' if rows and all(r['camera_gate_passed'] for r in rows) else 'Check camera validation status before interpreting scores.'}",'',
        '[Interpretation and next tests](BENCHMARK_FINDINGS.md) · [Frozen camera split](benchmark_results/frozen_camera_split.png)','',
        'Eight supplied-camera RGB inputs, seed 42, and 10,000 support points backprojected from **input views only** per room. Three other released views are held out. GenRecon’s native camera visibility filter can discard supplied RGB views; the effective count and dropped IDs are recorded below. This evaluates reconstruction with supplied calibration/support, not an RGB-only system. Original metric scale and training-house disjointness remain unverified. The five houses are an early-archive convenience sample spanning room types, not a representative benchmark.','',
        'D is the diagonal of each original furniture scene bounding box. Raw meshes are evaluated in the supplied coordinate frame without alignment fitting or cropping. Walls/floor/ceiling are excluded from reference geometry because released conditioning renders omit them. Added architectural-looking geometry therefore counts as unmatched under this furniture-only protocol; this alone does not establish physically implausible geometry.','',
        '## Surface accuracy and completeness','',
        '100,000 surface-area-uniform samples per mesh, seed 42; each sample is measured against the other mesh’s nearest **triangle**, rather than against a finite point cloud. Precision is the predicted-area fraction within a threshold; recall is the reference-area fraction; F-score is their harmonic mean. Symmetric mean is the average of the two unsquared directed means. Distances below are fractions of D.','',
        '| Scene | Room | Precision @ 1% D | Recall @ 1% D | F @ 0.5% D | F @ 1% D | F @ 2% D | Symmetric mean / D |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        text.append(f"| {r['scene']} | {r['room'].split('/')[-1]} | {pct(r['precision_0.01'])} | {pct(r['recall_0.01'])} | {pct(r['fscore_0.005'])} | {pct(r['fscore_0.01'])} | {pct(r['fscore_0.02'])} | {num(r['symmetric_mean_D'])} |")
    if macro:
        text.append(f"| Macro scene mean | | {pct(macro['precision_0.01'])} | {pct(macro['recall_0.01'])} | {pct(macro['fscore_0.005'])} | {pct(macro['fscore_0.01'])} | {pct(macro['fscore_0.02'])} | {num(macro['symmetric_mean_D'])} |")
    text.extend(['','## Held-out views','',
        'Camera-axis depth and silhouette queries use a fixed 192 × 192 grid of pixel centers sampled from the original 768 × 768 images. Background depth 65504 is invalid. AbsRel, MAE and RMSE use pixels where both reference and prediction hit; coverage and missing/extra rates prevent those errors from hiding holes. Per-room values pool held-out pixels; the final mean weights rooms equally. These are sampled-view metrics, not full-resolution raster benchmarks.','',
        '| Scene | Silhouette IoU | Depth coverage | Missing / GT | Extra / prediction | Depth AbsRel | MAE / D | RMSE / D | Correct depth <5%, all GT |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
    for r in rows+([{'scene':'Macro scene mean',**macro}] if macro else []):
        text.append(f"| {r['scene']} | {pct(r['heldout_iou'])} | {pct(r['heldout_depth_coverage'])} | {pct(r['heldout_missing_gt'])} | {pct(r['heldout_extra_pred'])} | {num(r['heldout_depth_absrel'])} | {num(r['heldout_depth_mae_D'])} | {num(r['heldout_depth_rmse_D'])} | {pct(r['correct_depth_gt_5pct'])} |")
    text.extend(['','## Visibility and spatial diagnostics','',
        'Reference samples are visible when projected depth agrees with at least one input reference depth within max(0.2% D, 0.5% depth). Occluded samples are behind a valid input reference surface in at least one view and visible in none; other samples remain unclassified. This is a depth-based approximation, not a semantic visibility label. The free-space column marks predicted samples significantly in front of a valid input depth in any view (tolerance max(1% D, 1% depth)). Background projection and bounds flags are diagnostics, not independent proof of hallucination.','',
        '| Scene | Visible recall, 8 supplied views | Occluded recall, 8 supplied views | Visible recall, effective RGB views | Occluded recall, effective RGB views | Input free-space violation | Projects onto input background | Outside GT bounds +1% D |',
        '|---|---:|---:|---:|---:|---:|---:|---:|'])
    for r in rows:
        text.append(f"| {r['scene']} | {pct(r['visible_recall_01'])} | {pct(r['occluded_recall_01'])} | {pct(r['effective_rgb_visible_recall_01'])} | {pct(r['effective_rgb_occluded_recall_01'])} | {pct(r['free_space_violation'])} | {pct(r['background_projection'])} | {pct(r['outside_gt_bounds'])} |")
    text.extend(['','All visibility recalls in this table use the 1% D distance threshold. GenRecon’s native camera gate requires a chunk cube’s clipped projected bounding box to occupy at least **40% of the image** (`genrecon/datasets/components.py:237`). It tests projected area, rather than reference-depth occlusion; a rejected camera may still contain useful observed surfaces. This gate and closest-camera fallback are preserved for the baseline.',''])
    text.extend(['## Supplementary scene completion geometry','',
        'A secondary comparison includes **all released room assets**: the primary furniture mesh plus available `ceil.glb`, `wall.glb`, `floor.glb` and `others.glb`. This explanatory analysis was added after inspecting the supplementary assets; primary geometry settings and room selection remained fixed. It helps distinguish added geometry consistent with the supplied scene from unmatched additions. `others.glb` has no reliable semantic labels, and this processed asset collection is not independently verified as a complete original 3D-FRONT room. The distance threshold remains 1% of the **furniture** diagonal D, so primary and secondary precision use the same physical tolerance. Full-reference area sampling uses 100,000 samples, seed 43. Held-out depth/silhouette scores above still use the released furniture-only renders.','',
        '| Scene | All-assets precision @1% D | All-assets recall @1% D | All-assets F-score @1% D |',
        '|---|---:|---:|---:|'])
    for r in rows:
        text.append(f"| {r['scene']} | {pct(r['all_assets_precision_01'])} | {pct(r['all_assets_recall_01'])} | {pct(r['all_assets_fscore_01'])} |")
    text.extend(['','## Raw geometry versus textured preview','',
        'The preview uses no global remesh, a per-chunk target of 250,000 triangles, 2048² textures, and standard hole filling/cleanup. Area-weighted sampling can change when export changes surface-area distribution, even when nearest-surface displacement stays small. F-score deltas below include Monte Carlo sampling variation; no repeat-seed uncertainty interval was estimated. Primary baseline scores always use the raw mesh.','',
        '| Scene | Raw F @1% D | Preview F @1% D | Delta, percentage points | Raw → preview P95 / D | Preview / raw surface area |',
        '|---|---:|---:|---:|---:|---:|'])
    for r in rows:
        text.append(f"| {r['scene']} | {pct(r['fscore_0.01'])} | {pct(r['preview_fscore_01'])} | {num(r['preview_fscore_delta_percentage_points'])} | {num(r['raw_to_preview_p95_D'])} | {num(r['preview_to_raw_area_ratio'])} |")
    text.extend(['','## Per-scene failures and object detail','',
        'Screening flags below use thresholds fixed in the evaluator (precision/recall <90%; missing/extra silhouette >5%; object recall <80% with ≥30 samples). These thresholds prioritize inspection and do not define benchmark pass/fail. Object samples are area-weighted, so tiny objects have less stable estimates. Full per-object geometry IDs, extents, surface area, visibility and recalls are retained in each `metrics.json` and `objects.csv`.',''])
    for scene,r in reports:
        sid=scene['id']
        retained=SOURCE/'benchmark_results'/sid
        retained.mkdir(parents=True,exist_ok=True)
        for name in ('metrics.json','objects.csv','heldout_views.png','surface_errors.png'):
            shutil.copyfile(OUT/sid/name,retained/name)
        run_manifest=Path(scene['output'])/'run_manifest.json'
        if run_manifest.exists():
            shutil.copyfile(run_manifest,retained/'run_manifest.json')
        parsed_args=Path(scene['output'])/'args.json'
        if parsed_args.exists():
            shutil.copyfile(parsed_args,retained/'args.json')
        text.extend([f"### {sid}: {r['room']}",'',
            f"Inputs: {', '.join(r['input_ids'])}. Held out: {', '.join(r['heldout_ids'])}.",
            f"Effective RGB conditioning: {', '.join(r.get('effective_conditioning_ids',r['input_ids']))}. Native chunks: {r.get('chunk_count','unrecorded')}; image-selection fallback warnings: {len(r.get('camera_selection_fallbacks',[]))}.",
            f"Reference: {r['reference_faces']:,} triangles; raw prediction: {r['prediction_faces']:,} triangles.",'',
            f"[Held-out diagnostics](benchmark_results/{sid}/heldout_views.png) · [Surface error map](benchmark_results/{sid}/surface_errors.png) · [Full metrics](benchmark_results/{sid}/metrics.json) · [Objects CSV](benchmark_results/{sid}/objects.csv)",''])
        comparison=OUT/sid/'comparison_textured.glb'
        if comparison.exists():
            text.extend([f'[Side-by-side textured GLB](../../../../outputs/3dfront_benchmark/{sid}/comparison_textured.glb) — reference furniture left, preview prediction right; display translation only. [All supplied reference assets](../../../../outputs/3dfront_benchmark/{sid}/reference/full_room.glb).',''])
        text.extend('- '+flag for flag in r['failure_flags'])
        text.extend(['','| Object | Samples | Visible | Recall @0.5% D | Recall @1% D | Recall @2% D | P95 / D |',
            '|---|---:|---:|---:|---:|---:|---:|'])
        for obj in r['per_object']:
            name=obj['node'].split('_')[0]+f" [{obj['object_id']}]"
            text.append(f"| {name} | {obj['count']} | {pct(obj['visible_sample_fraction'])} | {pct(obj['within_threshold']['0.005'])} | {pct(obj['within_threshold']['0.01'])} | {pct(obj['within_threshold']['0.02'])} | {num(obj['p95_fraction_diagonal'])} |")
        preview=r['preview_comparison']
        text.append('')
        context=r.get('architectural_context')
        if context:
            text.append(f"Additional-asset context ({', '.join(context.get('asset_files',['ceil.glb']))}): {pct(context.get('prediction_near_additional_assets_and_far_from_furniture_at_1pct_D',context.get('prediction_near_shell_and_far_from_furniture_at_1pct_D')))} of predicted surface is farther than 1% D from furniture but within 1% D of supplementary released assets in their supplied coordinates. These can include architecture and unlabeled `others.glb` geometry. This is a proximity diagnostic; the primary furniture score is unchanged.")
            text.append('')
        if preview:
            text.append(f"Export check: preview has {preview['faces']:,} triangles. Raw → preview mean distance / D = {num(preview['raw_to_preview']['mean_fraction_diagonal'])}; P95 / D = {num(preview['raw_to_preview']['p95_fraction_diagonal'])}. Preview precision/recall @1% D = {pct(preview['prediction_to_reference']['within_threshold']['0.01'])}/{pct(preview['reference_to_prediction']['within_threshold']['0.01'])}. Primary scores above use raw geometry.")
        else:text.append('Preview export was not evaluated; see progress/export logs for status.')
        text.append('')
        if run_manifest.exists():
            run=json.loads(run_manifest.read_text())
            telemetry=run.get('inference_telemetry')
            if telemetry:
                text.append(f"Execution telemetry: {telemetry['elapsed_seconds']:.1f} seconds including subprocess startup/model loading; sampled peak process GPU allocation {telemetry['sampled_peak_process_gpu_mib']} MiB. Samples every 2 seconds may miss peaks; shared-device timings are not isolated performance measurements.")
                text.append('')
    text.extend(['## Reproduction and limitations','',
        'Run `python -m experiments.3dfront_validation.preparation.prepare_benchmark`, prepare each eight-view input with `python -m experiments.3dfront_validation.preparation.prepare_genrecon_input`, then `python -m experiments.3dfront_validation.inference.run_benchmark` in the `genrecon` environment. `data/3dfront/benchmark/protocol.json` freezes room selection and settings. Per-scene progress histories, source/checkpoint fingerprints, inference/export logs and full JSON metrics are saved under `outputs/`. `BENCHMARK_METRICS.csv` retains the summary in `reports/benchmark/`.','',
        'The reference triangle/depth consistency gate checks every released view (≥20 paired rays, IoU ≥0.95, median relative depth error <0.01). No additional scene normalization or registration is fitted to predictions. There are no centimetre claims. Surface distances and held-out geometry do not measure texture quality, semantic correctness or physical support relationships. High recall does not by itself prove object identity or separation.','',
        'This baseline holds the model and eight-view selection rule fixed across rooms. It does not yet establish causality for observed failures. Targeted view-coverage ablations and chunk-boundary attribution require separate controlled runs. Shared GPU workloads confound wall-clock timing; sampled per-process GPU memory telemetry is diagnostic and may miss brief peaks.',''])
    if incomplete:text.extend(['Incomplete scenes: '+', '.join(incomplete),''])
    report='\n'.join(text)
    (SOURCE/'BENCHMARK_RESULTS.md').write_text(report)
    (OUT/'RESULTS.md').write_text(report.replace('(../../../../outputs/', '(../../').replace('(benchmark_results/', '(../../experiments/3dfront_validation/reports/benchmark/benchmark_results/').replace('(BENCHMARK_FINDINGS.md)', '(../../experiments/3dfront_validation/reports/benchmark/BENCHMARK_FINDINGS.md)'))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
