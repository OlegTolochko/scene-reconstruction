"""Attribute reference-surface errors near deterministic chunk boundaries."""
import numpy as np
import trimesh

from ..common import OUT, Y_TO_Z, geometry, read, write
from .surface import distances, query, sample


def main():
    result = {}
    for i in range(1, 6):
        scene = f'scene_{i:02d}'
        dest = OUT / scene
        plan = read(dest / 'chunks.json')
        edge = plan['edge_source_units']
        voxel = edge / 512
        original = geometry(dest / 'original.glb')
        points = sample(original, 42)
        points_z = trimesh.transform_points(points, Y_TO_Z)
        grid_low = np.array([c['low_zup'] for c in plan['chunks']]).min(axis=0)
        seam_distance = np.full(len(points), np.inf)
        planes = []
        for axis, count in enumerate(plan['grid_counts']):
            for j in range(1, count):
                position = grid_low[axis] + j * edge
                planes.append(dict(axis_zup=axis, position_source_units=float(position)))
                seam_distance = np.minimum(seam_distance, np.abs(points_z[:, axis] - position))
        near = seam_distance <= 2 * voxel
        stages = {}
        for label, mesh_path in [('decoded', 'decoded.glb'), ('voxel_control', 'voxel_control.glb')]:
            d = distances(query(geometry(dest / mesh_path)), points)
            missed = d > voxel
            stages[label] = dict(recall_1_voxel=float((~missed).mean()),
                missed_samples=int(missed.sum()),
                missed_samples_near_internal_seams=int((missed & near).sum()),
                fraction_of_misses_near_internal_seams=float((missed & near).sum() / missed.sum()) if missed.any() else None,
                recall_1_voxel_away_from_internal_seams=float((~missed[~near]).mean()),
                mean_error_voxels_away_from_internal_seams=float(d[~near].mean() / voxel))
        result[scene] = dict(planes=planes, boundary_band_voxels=2,
            fraction_reference_samples_near_seams=float(near.mean()), samples=100000, seed=42,
            stages=stages, interpretation='proximity attribution, not a causal correction; all original scores retained')
        write(dest / 'boundary_diagnostics.json', result[scene])
        print(scene, stages, flush=True)
    write(OUT / 'boundary_diagnostics.json', result)


if __name__ == '__main__':
    main()
