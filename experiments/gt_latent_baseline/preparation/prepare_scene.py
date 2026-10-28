"""Prepare deterministic chunks for the frozen 3D-FRONT latent baseline."""
import itertools

import numpy as np
import trimesh

from ..common import OUT, ROOT, RESOLUTION, Y_TO_Z, read, write, fingerprint, geometry, clip, export


def prepare(scene_id):
    dest = OUT / scene_id
    if (dest / 'chunks.json').exists():
        return read(dest / 'chunks.json')
    reference = ROOT / 'outputs/3dfront_benchmark' / scene_id / 'reference'
    full = geometry(reference / 'full_room.glb')
    furniture = geometry(reference / 'furniture_only.glb')
    full.apply_transform(Y_TO_Z)
    furniture.apply_transform(Y_TO_Z)
    # Match existing inference cube scale when present, extending only to contain
    # the full architecture height. No reference-to-prediction transform fitting.
    native_path = ROOT / 'outputs/3dfront_benchmark' / scene_id / 'native_preflight/chunk_transforms.json'
    native_edge = None
    if native_path.exists():
        native_edge = float(read(native_path)['chunks'][0]['M_chunk_to_original'][0][0])
    edge = max(float(full.extents[2]) * 1.02, native_edge or 0)
    counts = np.maximum(1, np.ceil(full.extents / edge).astype(int))
    center = full.bounds.mean(axis=0)
    grid_min = center - counts * edge / 2
    dest.mkdir(parents=True, exist_ok=True)
    source_y = full.copy().apply_transform(Y_TO_Z.T)
    export(source_y, dest / 'original.glb')
    chunks = []
    for index in itertools.product(*(range(int(n)) for n in counts)):
        low = grid_min + np.array(index) * edge
        high = low + edge
        original = clip(full, low, high)
        if not len(original.faces) or original.area < 1e-12:
            continue
        furniture_chunk = clip(furniture, low, high)
        chunks.append(dict(grid_index=list(index), low_zup=low.tolist(), high_zup=high.tolist(),
                           furniture_area=float(furniture_chunk.area), mesh=original))
    # First chunk has the most furniture surface, rather than selecting by score.
    chunks.sort(key=lambda c: -c['furniture_area'])
    records = []
    for i, entry in enumerate(chunks):
        original = entry.pop('mesh')
        chunk = dest / f'chunk_{i:03d}'
        chunk.mkdir(exist_ok=True)
        crop_center = (np.array(entry['low_zup']) + entry['high_zup']) / 2
        z_to_chunk = np.eye(4)
        z_to_chunk[:3, :3] /= edge
        z_to_chunk[:3, 3] = -crop_center / edge
        to_chunk = z_to_chunk @ Y_TO_Z
        to_source = np.linalg.inv(to_chunk)
        normalized = original.copy().apply_transform(z_to_chunk)
        assert np.max(np.abs(normalized.vertices)) <= .500001
        # normalized.glb is glTF Y-up. get_slats applies +X90 once on loading.
        normalized_y = normalized.copy().apply_transform(Y_TO_Z.T)
        export(normalized_y, chunk / 'normalized.glb')
        original_y = original.copy().apply_transform(Y_TO_Z.T)
        export(original_y, chunk / 'original.glb')
        roundtrip = trimesh.transform_points(normalized.vertices, to_source)
        error = float(np.max(np.abs(roundtrip - original_y.vertices)))
        assert error < 1e-8
        entry.update(id=f'chunk_{i:03d}', edge_source_units=edge,
                     voxel_size_source_units=edge / RESOLUTION,
                     M_source_yup_to_chunk_zup=to_chunk.tolist(),
                     M_chunk_zup_to_source_yup=to_source.tolist(),
                     original_faces=len(original.faces), original_area=float(original.area),
                     frame_roundtrip_max_abs_error=error,
                     determinant_rotation=float(np.linalg.det(Y_TO_Z[:3, :3])),
                     normalized_zup_bounds=normalized.bounds.tolist())
        write(chunk / 'transform.json', entry)
        records.append(entry)
    assert abs(sum(c['original_area'] for c in records) - full.area) <= full.area * 1e-5
    report = dict(scene=scene_id, edge_source_units=edge, native_edge_source_units=native_edge,
                  grid_counts=counts.tolist(), chunks=records, source_bounds_yup=source_y.bounds.tolist(),
                  source_faces=len(full.faces), source_area=float(full.area),
                  clipped_area=sum(c['original_area'] for c in records),
                  reference=fingerprint(reference / 'full_room.glb'),
                  furniture_reference=fingerprint(reference / 'furniture_only.glb'))
    write(dest / 'chunks.json', report)
    print(f'Prepared {scene_id}: {len(records)} chunks, edge={edge:.6f}', flush=True)
    return report


