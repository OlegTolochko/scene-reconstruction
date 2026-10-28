"""Verify PBR checkpoint/latent provenance and exact comparison export contents."""
import hashlib
from pathlib import Path

import numpy as np
import trimesh

from ..common import OUT, ROOT, SOURCE, Y_TO_Z, fingerprint, read, write


def material_signature(mesh):
    visual = mesh.visual
    material = getattr(visual, 'material', None)
    if material is None:
        return None
    if not isinstance(material, trimesh.visual.material.PBRMaterial):
        material = material.to_pbr()
    values = {}
    for name in ['baseColorFactor', 'metallicFactor', 'roughnessFactor', 'emissiveFactor',
                 'alphaMode', 'alphaCutoff', 'doubleSided']:
        value = getattr(material, name, None)
        values[name] = np.asarray(value).tolist() if isinstance(value, (np.ndarray, np.generic)) else value
    for name in ['baseColorTexture', 'metallicRoughnessTexture', 'emissiveTexture', 'normalTexture', 'occlusionTexture']:
        image = getattr(material, name, None)
        values[name] = None if image is None else dict(mode=image.mode, size=list(image.size),
            pixels_sha256=hashlib.sha256(np.asarray(image).tobytes()).hexdigest())
    return values


def compare_nodes(source, comparison, prefix, display_x=0.):
    for node in source.graph.nodes_geometry:
        transform, name = source.graph[node]
        exported_transform, exported_name = comparison.graph[f'{prefix}_{node}']
        display = np.eye(4)
        display[0, 3] = display_x
        assert np.allclose(exported_transform, display @ transform, atol=1e-12)
        before, after = source.geometry[name], comparison.geometry[exported_name]
        assert np.array_equal(before.vertices.astype(np.float32), after.vertices.astype(np.float32))
        assert np.array_equal(before.faces, after.faces)
        assert material_signature(before) == material_signature(after)
        uv_before, uv_after = getattr(before.visual, 'uv', None), getattr(after.visual, 'uv', None)
        if uv_before is not None:
            assert np.array_equal(uv_before.astype(np.float32), uv_after.astype(np.float32))


def main(scene_ids=None):
    provenance = read(OUT / 'pbr_provenance.json')
    for model in provenance['checkpoints'].values():
        for record in model['files'].values():
            assert fingerprint(record['path'])['sha256'] == record['sha256']
    for name, record in provenance['sources'].items():
        assert fingerprint(SOURCE / name)['sha256'] == record['sha256']
    results = {}
    for sid in scene_ids or [f'scene_{i:02d}' for i in range(1, 6)]:
        dest = OUT / sid
        plan = read(dest / 'chunks.json')
        chunks = []
        for entry in plan['chunks']:
            chunk = dest / entry['id']
            shape, pbr = np.load(chunk / 'shape_raw.npz'), np.load(chunk / 'pbr_raw.npz')
            assert np.array_equal(shape['coords'], pbr['coords'])
            assert pbr['feats'].shape == shape['feats'].shape
            assert np.isfinite(pbr['feats']).all()
            checksum = fingerprint(chunk / 'pbr_raw.npz')['sha256']
            assert checksum == read(chunk / 'pbr_encoding.json')['latents']['sha256']
            assert checksum == read(chunk / 'pbr_decoding.json')['raw_latents']['sha256']
            assert read(chunk / 'pbr_subdivisions.json')['original_shape_redecode_exact']
            shape_sha = fingerprint(chunk / 'shape_raw.npz')['sha256']
            assert shape_sha == read(chunk / 'encoding.json')['latents']['sha256']
            assert fingerprint(chunk / 'decoded.glb')['sha256'] == read(chunk / 'decoding.json')['decoded']['sha256']
            output = read(chunk / 'pbr_export.json')
            assert output['exact_triangles_verified']
            assert output['faces'] == read(chunk / 'decoding.json')['raw_faces']
            assert output['decoded_pbr']['sha256'] == fingerprint(chunk / 'decoded_pbr.glb')['sha256']
            source = trimesh.load_scene(chunk / 'normalized_pbr.glb', process=False)
            for g in source.geometry.values():
                assert np.isfinite(g.visual.uv).all()
            source.apply_transform(np.array(entry['M_chunk_zup_to_source_yup']) @ Y_TO_Z)
            decoded = trimesh.load_scene(chunk / 'decoded_pbr.glb', process=False)
            comparison = trimesh.load_scene(chunk / 'comparison.glb', process=False)
            record = read(chunk / 'pbr_comparison.json')
            compare_nodes(source, comparison, 'ORIGINAL')
            compare_nodes(decoded, comparison, 'DECODED_PBR', record['display_translation_x'])
            assert comparison.metadata['error_colors'] is False
            for g in decoded.geometry.values():
                if not len(g.faces) or g.area == 0:
                    continue
                assert isinstance(g.visual, trimesh.visual.TextureVisuals)
                assert g.visual.material.baseColorTexture is not None
                assert g.visual.material.metallicRoughnessTexture is not None
            chunks.append(dict(id=entry['id'], faces=output['faces'], exact_triangles_verified=True,
                               pbr_latent_sha256=checksum, material_pixels_and_uvs_preserved=True))
        source = trimesh.load_scene(ROOT / 'outputs/3dfront_benchmark' / sid / 'reference/full_room.glb', process=False)
        decoded = trimesh.load_scene(dest / 'decoded_pbr.glb', process=False)
        comparison = trimesh.load_scene(dest / 'comparison.glb', process=False)
        record = read(dest / 'pbr_comparison.json')
        assert fingerprint(dest / 'comparison.glb')['sha256'] == record['comparison']['sha256']
        compare_nodes(source, comparison, 'ORIGINAL')
        compare_nodes(decoded, comparison, 'DECODED_PBR', record['display_translation_x'])
        assert sum(len(g.faces) for g in decoded.geometry.values()) == sum(c['faces'] for c in chunks)
        assert comparison.metadata['error_colors'] is False
        results[sid] = dict(status='passed', chunks=chunks, comparison=fingerprint(dest / 'comparison.glb'),
                            source_materials_preserved=True, decoded_PBR_maps_preserved=True)
        print(f'PBR verification passed: {sid} ({len(chunks)} chunks)', flush=True)
    write(OUT / ('pbr_verification_partial.json' if scene_ids else 'pbr_verification.json'), dict(status='passed', scenes=results,
        checkpoints_and_sources_verified=True, exact_shape_geometry_retained=True,
        original_and_decoded_material_pixels_and_uvs_verified=True))
    if scene_ids:
        return
    # Retain the original shape verification and point it at the updated exports.
    original_verification = read(OUT / 'verification.json')
    for sid, record in original_verification['scenes'].items():
        record['surface_error_comparison'] = fingerprint(OUT / sid / 'comparison_surface_error.glb')
        record['comparison'] = results[sid]['comparison']
    original_verification['pbr_extension'] = str(OUT / 'pbr_verification.json')
    write(OUT / 'verification.json', original_verification)
    provenance['implementation_at_completion'] = {str(p.relative_to(ROOT)): fingerprint(p) for p in (ROOT / 'experiments/gt_latent_baseline').rglob('*.py')}
    provenance['uv_export'] = '4096-square atlases; per-chunk records distinguish global UVs and 8-cubed centroid partitions; exact shape triangles retained'
    provenance['execution'] = 'separate encoder/decoder processes; CUDA allocator caching disabled after first room to fit shared GPU memory'
    write(OUT / 'pbr_provenance.json', provenance)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scenes', nargs='*')
    main(parser.parse_args().scenes)
