"""Shape VAE stages and orchestration for the frozen GT latent baseline.

Run with python -m experiments.gt_latent_baseline.reconstruction.run_baseline.
GPU stages use separate processes to bound simultaneous model allocations.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import trimesh

from ..common import (
    ROOT, SOURCE, OUT, UPSTREAM, HF_REPO, HF_REVISION, ENCODER, DECODER,
    Y_TO_Z, RESOLUTION, write, read, fingerprint, checkpoint_files,
    geometry, clip, mesh_valid, export,
)
from ..preparation.prepare_scene import prepare
from ..evaluation.evaluate import evaluate_chunk, evaluate_scene


def provenance():
    records = {}
    for label, name in [('encoder', ENCODER), ('decoder', DECODER)]:
        records[label] = dict(id=f'{HF_REPO}/ckpts/{name}', revision=HF_REVISION,
                             files={ext: fingerprint(p) for ext, p in checkpoint_files(name).items()})
    paths = ['data_toolkit_scenes/get_slats.py', 'genrecon/models/sc_vaes/fdg_vae.py',
             'genrecon/models/sc_vaes/sparse_unet_vae.py', 'genrecon/datasets/structured_latent.py',
             'data_toolkit_scenes/blender_scripts/create_chunks.py']
    report = dict(experiment='Ground-truth latent reconstruction baseline',
        genrecon_upstream_repository='https://github.com/kasothaphie/GenRecon',
        genrecon_imported_commit=UPSTREAM,
        workspace_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_files={p: fingerprint(SOURCE / p) for p in paths}, checkpoints=records,
        geometry_resolution=RESOLUTION, shape_latent_grid_resolution=RESOLUTION // 16,
        posterior='mean (get_slats.py default sample_posterior=False)',
        latent_space='raw encoder output; no normalization or denormalization',
        decoder_coordinates='saved encoder coords unchanged; prepend batch index zero only',
        sparse_structure_prediction=False, texture=False, dit=False, image_conditioning=False,
        alignment='none; no ICP, fitted rotation, translation, or scale',
        metric_scale_verified=False,
        target='all supplied benchmark full_room assets: furniture, floor, wall, ceil, others when available',
        chunking='deterministic nonoverlapping cubes covering full reference; open triangle clipping, no caps or solidification',
        preprocessing_difference='training toolkit uses randomized Blender boolean crops and 0.001 solidification; this test preserves supplied triangles with deterministic clipping instead',
        frame='glTF Y-up -> +90 degrees about X -> Z-up -> fixed crop translation/uniform scale; invert for export',
        script=fingerprint(Path(__file__)))
    write(OUT / 'provenance.json', report)


def load_model(name):
    import torch
    sys.path.insert(0, str(SOURCE))
    import genrecon.models as models
    from safetensors.torch import load_file
    files = checkpoint_files(name)
    config = read(files['json'])
    model = getattr(models, config['name'])(**config['args'])
    # Strict loading prevents a partially initialized VAE from producing a
    # seemingly valid baseline (from_pretrained uses strict=False).
    model.load_state_dict(load_file(str(files['safetensors'])), strict=True)
    model.eval().cuda()
    torch.manual_seed(42)
    return model


def telemetry(start):
    import torch
    torch.cuda.synchronize()
    return dict(elapsed_seconds=time.monotonic() - start,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name())


def encode(chunk):
    import torch
    start = time.monotonic()
    model = load_model(ENCODER)
    spec = importlib.util.spec_from_file_location('gt_get_slats', SOURCE / 'data_toolkit_scenes/get_slats.py')
    toolkit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(toolkit)
    normalized = geometry(chunk / 'normalized.glb').apply_transform(Y_TO_Z)
    assert np.abs(normalized.vertices).max() < .500001
    counts = {}

    def capture_encoder(vertices, intersected):
        # Capture the exact 8-bit-quantized FDG features fed to the official
        # encoder, before any VAE compression. This is a preprocessing control.
        coords = vertices.coords[:, 1:].cpu().numpy().astype(np.int32)
        dual = vertices.feats.cpu().numpy()
        flags = intersected.feats.cpu().numpy()
        np.savez_compressed(chunk / 'voxels.npz', coords=coords,
                            dual_vertices_uint8=np.rint(dual * 255).astype(np.uint8),
                            intersected=flags)
        counts['voxel_count'] = len(coords)
        with torch.no_grad():
            z = model(vertices, intersected)
        assert z.coords[:, 1:].min() >= 0 and z.coords[:, 1:].max() < RESOLUTION // 16
        counts['latent_tokens'] = len(z.coords)
        return z

    with torch.no_grad():
        record, _, _ = toolkit.process_instance(
            root=str(chunk), sha256=chunk.name, local_path='normalized.glb',
            resolution_ovoxel=RESOLUTION, resolution_slat=RESOLUTION // 16,
            shape_encoder=capture_encoder, pbr_encoder=None, ss_encoder=None,
            shape_npz=str(chunk / 'shape_raw.npz'), pbr_npz=None, ss_npz=None,
            axis_fix_mode='auto')
    write(chunk / 'encoding.json', dict(status='complete', record=record, **counts,
                                        **telemetry(start), latents=fingerprint(chunk / 'shape_raw.npz')))
    print(f'Encoded {chunk}: {counts}', flush=True)


def decode(chunk):
    import torch
    start = time.monotonic()
    model = load_model(DECODER)
    model.set_resolution(RESOLUTION)
    from genrecon.modules.sparse import SparseTensor
    import o_voxel
    saved = np.load(chunk / 'shape_raw.npz')
    features = torch.from_numpy(saved['feats'].copy()).cuda()
    coords = torch.from_numpy(saved['coords'].astype(np.int32)).cuda()
    coords_batch = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1)
    latent = SparseTensor(feats=features, coords=coords_batch)
    assert np.array_equal(latent.coords[:, 1:].cpu().numpy(), saved['coords'])
    assert np.array_equal(latent.feats.cpu().numpy(), saved['feats'])
    with torch.no_grad():
        mesh = model(latent)[0]
    raw = trimesh.Trimesh(vertices=mesh.vertices.cpu().numpy(), faces=mesh.faces.cpu().numpy(), process=False)
    mesh_valid(raw)
    export(raw, chunk / 'decoded_normalized_zup.ply')
    transform = np.array(read(chunk / 'transform.json')['M_chunk_zup_to_source_yup'])
    original_frame = raw.copy().apply_transform(transform)
    export(original_frame, chunk / 'decoded.glb')
    # Drop the decoder before extracting the FDG control mesh.
    del model, mesh, latent, features, coords, coords_batch
    torch.cuda.empty_cache()
    voxels = np.load(chunk / 'voxels.npz')
    with torch.no_grad():
        v, f = o_voxel.convert.flexible_dual_grid_to_mesh(
            torch.from_numpy(voxels['coords']).cuda(),
            torch.from_numpy(voxels['dual_vertices_uint8'].astype(np.float32) / 255).cuda(),
            torch.from_numpy(voxels['intersected']).cuda(), None,
            aabb=[[-.5, -.5, -.5], [.5, .5, .5]], grid_size=RESOLUTION, train=False)
    control = trimesh.Trimesh(vertices=v.cpu().numpy(), faces=f.cpu().numpy(), process=False)
    control.apply_transform(transform)
    export(control, chunk / 'voxel_control.glb')
    write(chunk / 'decoding.json', dict(status='complete', **telemetry(start),
        decoder_resolution=model_resolution() , raw_faces=len(raw.faces),
        input_coords_exact=True, input_feats_exact=True, latent_denormalization=False,
        shape_latents=fingerprint(chunk / 'shape_raw.npz'),
        decoded=fingerprint(chunk / 'decoded.glb'), voxel_control_faces=len(control.faces)))
    print(f'Decoded {chunk}: {len(raw.faces):,} triangles', flush=True)


def model_resolution():
    return RESOLUTION


def child(stage, target):
    command = [sys.executable, '-m', 'experiments.gt_latent_baseline.reconstruction.run_baseline', stage, str(target)]
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               ATTN_BACKEND='flash_attn', SPARSE_ATTN_BACKEND='flash_attn',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    log_dir = Path(target) if stage != 'evaluate-scene' else OUT / target
    with (log_dir / f'{stage}.log').open('w') as log:
        p = subprocess.Popen(command, env=env, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            log.write(line)
            log.flush()
            print(line, end='', flush=True)
        status = p.wait()
    if status:
        raise RuntimeError(f'{stage} failed for {target}; see {log_dir / (stage + ".log")}')


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / 'provenance.json').exists():
        provenance()
    # Establish and persist one example before preparing/encoding the dataset.
    pilot = prepare('scene_04')
    chunk = OUT / 'scene_04' / pilot['chunks'][0]['id']
    if not (OUT / 'working_example.json').exists():
        for stage, artifact in [('encode', 'encoding.json'), ('decode', 'decoding.json'), ('evaluate-chunk', 'metrics.json')]:
            if not (chunk / artifact).exists():
                child(stage, chunk)
        metrics = read(chunk / 'metrics.json')
        # Gross sanity gate, not a claim about fine detail fidelity.
        precision = metrics['scores']['4_voxels']['precision']
        recall = metrics['scores']['4_voxels']['recall']
        if min(precision, recall) < .5:
            raise RuntimeError('Working example failed coarse geometry gate. Inspect frame and pipeline before proceeding.')
        write(OUT / 'working_example.json', dict(status='passed', scene='scene_04', chunk=chunk.name,
            description='largest furniture area chunk; includes supplied architecture',
            frame_checks=read(chunk / 'transform.json'), precision_at_4_voxels=precision,
            recall_at_4_voxels=recall, criterion='both precision and recall >= 0.5; gross pipeline sanity only',
            comparison=str(chunk / 'comparison.glb'), completed_before_dataset=True))
    for scene_id in ['scene_04', 'scene_01', 'scene_02', 'scene_03', 'scene_05']:
        plan = prepare(scene_id)
        for entry in plan['chunks']:
            chunk = OUT / scene_id / entry['id']
            for stage, artifact in [('encode', 'encoding.json'), ('decode', 'decoding.json')]:
                if not (chunk / artifact).exists():
                    child(stage, chunk)
        if not (OUT / scene_id / 'metrics.json').exists():
            child('evaluate-scene', scene_id)
    write(OUT / 'summary.json', {f'scene_{i:02d}': read(OUT / f'scene_{i:02d}' / 'metrics.json') for i in range(1, 6)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['run', 'provenance', 'prepare', 'encode', 'decode', 'evaluate-chunk', 'evaluate-scene'])
    parser.add_argument('target', nargs='?')
    args = parser.parse_args()
    if args.stage == 'run':
        run()
    elif args.stage == 'provenance':
        provenance()
    elif args.stage == 'prepare':
        prepare(args.target)
    elif args.stage == 'encode':
        encode(Path(args.target))
    elif args.stage == 'decode':
        decode(Path(args.target))
    elif args.stage == 'evaluate-chunk':
        evaluate_chunk(Path(args.target))
    else:
        evaluate_scene(args.target)


if __name__ == '__main__':
    main()
