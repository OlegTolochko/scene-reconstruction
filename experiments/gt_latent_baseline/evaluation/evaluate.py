"""Evaluate reconstructed chunks and aggregate scene geometry."""
import numpy as np
import trimesh

from ..common import OUT, ROOT, RESOLUTION, read, write, geometry, export
from .surface import compare, distances, query, sample, stats
from ..visualization.export_comparison import comparisons


def evaluate_chunk(chunk):
    edge = read(chunk / 'transform.json')['edge_source_units']
    reference, prediction = geometry(chunk / 'original.glb'), geometry(chunk / 'decoded.glb')
    control = geometry(chunk / 'voxel_control.glb')
    result = comparisons(chunk, reference, prediction, edge)
    result['preprocessing_control'] = compare(reference, control, edge)[0]
    result['voxel_to_decoded'] = compare(control, prediction, edge)[0]
    write(chunk / 'metrics.json', result)
    return result


def evaluate_scene(scene_id):
    dest = OUT / scene_id
    plan = read(dest / 'chunks.json')
    predictions, controls = [], []
    for entry in plan['chunks']:
        chunk = dest / entry['id']
        if not (chunk / 'metrics.json').exists():
            evaluate_chunk(chunk)
        predictions.append(geometry(chunk / 'decoded.glb'))
        controls.append(geometry(chunk / 'voxel_control.glb'))
    reference = geometry(dest / 'original.glb')
    prediction = trimesh.util.concatenate(predictions)
    control = trimesh.util.concatenate(controls)
    export(prediction, dest / 'decoded.glb')
    export(control, dest / 'voxel_control.glb')
    result = comparisons(dest, reference, prediction, plan['edge_source_units'])
    result['preprocessing_control'] = compare(reference, control, plan['edge_source_units'])[0]
    result['voxel_to_decoded'] = compare(control, prediction, plan['edge_source_units'])[0]
    # Furniture-only recall exposes detail loss which architectural area can hide.
    furniture = geometry(ROOT / 'outputs/3dfront_benchmark' / scene_id / 'reference/furniture_only.glb')
    d = distances(query(prediction), sample(furniture, 44))
    result['furniture_reference_to_prediction'] = stats(d, plan['edge_source_units'], np.linalg.norm(furniture.extents))
    result['furniture_recall_at_1_2_4_voxels'] = {str(n): float((d <= n * plan['edge_source_units'] / RESOLUTION).mean()) for n in (1, 2, 4)}
    result.update(scene=scene_id, chunk_count=len(predictions), voxel_size_source_units=plan['edge_source_units'] / RESOLUTION)
    write(dest / 'metrics.json', result)
    return result


