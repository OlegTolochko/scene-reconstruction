"""Independently verify saved coordinate transforms, latent files and GLBs."""
from pathlib import Path

import numpy as np
import trimesh

from ..common import OUT, ROOT, SOURCE, Y_TO_Z, fingerprint, geometry, read, write


def main():
    provenance = read(OUT / 'provenance.json')
    for path, record in provenance['source_files'].items():
        assert fingerprint(SOURCE / path)['sha256'] == record['sha256'], path
    for model in provenance['checkpoints'].values():
        for record in model['files'].values():
            assert fingerprint(record['path'])['sha256'] == record['sha256']
    scenes = {}
    for i in range(1, 6):
        scene_id = f'scene_{i:02d}'
        dest = OUT / scene_id
        plan = read(dest / 'chunks.json')
        ref = ROOT / 'outputs/3dfront_benchmark' / scene_id / 'reference/full_room.glb'
        assert fingerprint(ref)['sha256'] == plan['reference']['sha256']
        source, original = geometry(ref), geometry(dest / 'original.glb')
        assert len(source.faces) == len(original.faces)
        assert np.allclose(source.bounds, original.bounds, atol=1e-6)
        chunks = []
        for entry in plan['chunks']:
            chunk = dest / entry['id']
            matrix = np.array(entry['M_source_yup_to_chunk_zup'])
            inverse = np.array(entry['M_chunk_zup_to_source_yup'])
            assert np.allclose(matrix @ inverse, np.eye(4), atol=1e-12)
            rotation = matrix[:3, :3] * entry['edge_source_units']
            assert np.allclose(rotation, Y_TO_Z[:3, :3], atol=1e-12)
            assert np.isclose(np.linalg.det(rotation), 1)
            original = geometry(chunk / 'original.glb')
            normalized = geometry(chunk / 'normalized.glb')
            encoded_vertices = trimesh.transform_points(normalized.vertices, Y_TO_Z)
            expected = trimesh.transform_points(original.vertices, matrix)
            normalization_error = float(np.abs(expected - encoded_vertices).max())
            assert normalization_error < 1e-6
            assert np.abs(encoded_vertices).max() <= .500001
            latent = np.load(chunk / 'shape_raw.npz')
            assert latent['feats'].dtype == np.float32
            assert latent['coords'].dtype == np.uint8
            assert latent['feats'].shape == (len(latent['coords']), 32)
            assert latent['coords'].shape[1] == 3
            assert np.isfinite(latent['feats']).all()
            assert latent['coords'].max() < 32
            assert len(np.unique(latent['coords'], axis=0)) == len(latent['coords'])
            enc, dec = read(chunk / 'encoding.json'), read(chunk / 'decoding.json')
            checksum = fingerprint(chunk / 'shape_raw.npz')['sha256']
            assert checksum == enc['latents']['sha256'] == dec['shape_latents']['sha256']
            assert dec['input_feats_exact'] and dec['input_coords_exact']
            assert dec['decoder_resolution'] == 512 and dec['latent_denormalization'] is False
            assert fingerprint(chunk / 'decoded.glb')['sha256'] == dec['decoded']['sha256']
            raw = trimesh.load(chunk / 'decoded_normalized_zup.ply', process=False)
            decoded = geometry(chunk / 'decoded.glb')
            output_vertices = trimesh.transform_points(raw.vertices, inverse)
            decoding_error = float(np.abs(decoded.vertices - output_vertices).max())
            assert decoding_error < 1e-6
            assert np.array_equal(raw.faces, decoded.faces)
            assert len(decoded.faces) == dec['raw_faces']
            for name in ['comparison.glb', 'overlay.glb']:
                restored = trimesh.load_scene(chunk / name, process=False)
                assert sum(len(g.faces) for g in restored.geometry.values()) == len(original.faces) + len(decoded.faces)
            chunks.append(dict(id=entry['id'], normalization_max_abs_error=normalization_error,
                               decoding_export_max_abs_error=decoding_error,
                               latent_tokens=len(latent['coords']), decoded_faces=len(decoded.faces),
                               latent_sha256=checksum))
        decoded = geometry(dest / 'decoded.glb')
        assert len(decoded.faces) == sum(c['decoded_faces'] for c in chunks)
        original = geometry(dest / 'original.glb')
        for name in ['comparison.glb', 'overlay.glb']:
            restored = trimesh.load_scene(dest / name, process=False)
            assert sum(len(g.faces) for g in restored.geometry.values()) == len(original.faces) + len(decoded.faces)
        metrics = read(dest / 'metrics.json')
        assert metrics['alignment'] == 'none' and metrics['comparison_export_roundtrip_verified']
        scenes[scene_id] = dict(status='passed', chunks=chunks, decoded_faces=len(decoded.faces),
                                comparison=fingerprint(dest / 'comparison.glb'))
        print(f'Verified {scene_id}: {len(chunks)} chunks, {len(decoded.faces):,} faces', flush=True)
    write(OUT / 'verification.json', dict(status='passed', geometry_resolution=512,
        provenance_checkpoints_and_sources_verified=True, scenes=scenes))


if __name__ == '__main__':
    main()
