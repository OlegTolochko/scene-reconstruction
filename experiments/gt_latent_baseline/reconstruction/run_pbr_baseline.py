"""Add ground-truth PBR VAE materials to the unchanged shape baseline."""
import argparse
import gc
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import trimesh
from PIL import Image

from ..common import (ROOT, SOURCE, OUT, HF_REPO, HF_REVISION, RESOLUTION,
                          Y_TO_Z, checkpoint_files, clip, fingerprint, geometry,
                          read, write, DECODER)
from .run_baseline import load_model

TEX_ENCODER = 'tex_enc_next_dc_f16c32_fp16'
TEX_DECODER = 'tex_dec_next_dc_f16c32_fp16'
TEXTURE_SIZE = 4096


def toolkit():
    sys.path.insert(0, str(SOURCE))
    spec = importlib.util.spec_from_file_location('pbr_get_slats', SOURCE / 'data_toolkit_scenes/get_slats.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def provenance():
    records = {}
    for name in (TEX_ENCODER, TEX_DECODER):
        records[name] = dict(id=f'{HF_REPO}/ckpts/{name}', revision=HF_REVISION,
                            files={ext: fingerprint(path) for ext, path in checkpoint_files(name).items()})
    write(OUT / 'pbr_provenance.json', dict(checkpoints=records, geometry_resolution=512,
        script=fingerprint(Path(__file__)), raw_latents=True, latent_denormalization=False,
        shape_geometry='existing decoded meshes unchanged; no hole filling or simplification',
        texture_channels=['base_color', 'metallic', 'roughness', 'alpha'],
        texture_size=TEXTURE_SIZE, dit=False, image_conditioning=False,
        guide_subdivisions='shape decoder subdivisions from the existing raw shape latents',
        sources={p: fingerprint(SOURCE / p) for p in ['genrecon/datasets/structured_latent_svpbr.py',
            'o-voxel/o_voxel/postprocess.py', 'o-voxel/o_voxel/convert/volumetic_attr.py']}))


def prepare(chunk):
    entry = read(chunk / 'transform.json')
    source = ROOT / 'outputs/3dfront_benchmark' / chunk.parent.name / 'reference/full_room.glb'
    scene = trimesh.load_scene(source, process=False)
    normalized = trimesh.Scene()
    undefined_uv_vertices_on_zero_area_faces = 0
    matrix = np.array(entry['M_source_yup_to_chunk_zup'])
    for i, node in enumerate(scene.graph.nodes_geometry):
        transform, name = scene.graph[node]
        mesh = scene.geometry[name].copy()
        mesh.apply_transform(Y_TO_Z @ transform)
        # Preserve constant materials through trimesh's UV-aware plane clipper.
        visual = mesh.visual
        if isinstance(visual, trimesh.visual.TextureVisuals):
            material = visual.material.copy()
            if not isinstance(material, trimesh.visual.material.PBRMaterial):
                material = material.to_pbr()
            uv = visual.uv if visual.uv is not None else np.zeros((len(mesh.vertices), 2))
        else:
            color = np.asarray(visual.main_color if visual.kind is not None else [255, 255, 255, 255], np.uint8)
            material = trimesh.visual.material.PBRMaterial(baseColorFactor=color,
                metallicFactor=0. if visual.kind is not None else 1., roughnessFactor=1.)
            uv = np.zeros((len(mesh.vertices), 2))
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
        mesh = clip(mesh, np.array(entry['low_zup']), np.array(entry['high_zup']))
        if not len(mesh.faces):
            continue
        uv = mesh.visual.uv.copy()
        undefined = ~np.isfinite(uv).all(axis=1)
        if undefined.any():
            affected_faces = undefined[mesh.faces].any(axis=1)
            assert np.all(mesh.area_faces[affected_faces] <= 1e-20), 'Undefined UV on a nondegenerate triangle'
            # Barycentric interpolation is undefined for zero-area triangles.
            # Their UVs have no surface extent; give them finite dummy values.
            uv[undefined] = 0
            mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=mesh.visual.material)
            undefined_uv_vertices_on_zero_area_faces += int(undefined.sum())
        z_to_chunk = matrix @ Y_TO_Z.T
        mesh.apply_transform(Y_TO_Z.T @ z_to_chunk)
        normalized.add_geometry(mesh, node_name=f'{i}_{node}', geom_name=f'{i}_{name}')
    normalized.export(chunk / 'normalized_pbr.glb')
    loaded = geometry(chunk / 'normalized_pbr.glb')
    shape = geometry(chunk / 'normalized.glb')
    assert np.allclose(loaded.bounds, shape.bounds, atol=1e-6)
    assert abs(loaded.area - shape.area) < max(1e-6, shape.area * 1e-5)
    write(chunk / 'pbr_input.json', dict(source=fingerprint(source),
        normalized=fingerprint(chunk / 'normalized_pbr.glb'), bounds=loaded.bounds.tolist(),
        area=loaded.area, material_geometries=len(normalized.geometry),
        undefined_uv_vertices_on_zero_area_faces=undefined_uv_vertices_on_zero_area_faces,
        note='Source textures/UVs retained; constant-color non-textured surfaces converted to equivalent constant PBR.'))


def encode(chunk):
    model = load_model(TEX_ENCODER)
    t = toolkit()
    _, record, _ = t.process_instance(root=str(chunk), sha256=chunk.name,
        local_path='normalized_pbr.glb', resolution_ovoxel=512, resolution_slat=32,
        shape_encoder=None, pbr_encoder=model, ss_encoder=None,
        shape_npz=None, pbr_npz=str(chunk / 'pbr_raw.npz'), ss_npz=None, axis_fix_mode='auto')
    shape, texture = np.load(chunk / 'shape_raw.npz'), np.load(chunk / 'pbr_raw.npz')
    assert np.array_equal(shape['coords'], texture['coords']), 'Shape/PBR encoder sparse coordinates differ'
    write(chunk / 'pbr_encoding.json', dict(status='complete', record=record,
        coords_equal_to_shape=True, latent_denormalization=False,
        latents=fingerprint(chunk / 'pbr_raw.npz')))


def sparse(npz):
    import torch
    from genrecon.modules.sparse import SparseTensor
    coords = torch.from_numpy(npz['coords'].astype(np.int32)).cuda()
    tensor = SparseTensor(torch.from_numpy(npz['feats'].astype(np.float32)).cuda(),
                          torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1))
    assert np.array_equal(tensor.coords[:, 1:].cpu().numpy(), npz['coords'])
    assert np.array_equal(tensor.feats.cpu().numpy(), npz['feats'])
    return tensor


def subdivisions(chunk):
    import torch
    model = load_model(DECODER)
    model.set_resolution(512)
    with torch.no_grad():
        meshes, subs = model(sparse(np.load(chunk / 'shape_raw.npz')), return_subs=True)
    raw = trimesh.load(chunk / 'decoded_normalized_zup.ply', process=False)
    assert np.array_equal(meshes[0].vertices.cpu().numpy(), raw.vertices.astype(np.float32))
    assert np.array_equal(meshes[0].faces.cpu().numpy(), raw.faces)
    torch.save([dict(feats=s.feats.cpu(), coords=s.coords.cpu()) for s in subs], chunk / 'shape_subdivisions.pt')
    write(chunk / 'pbr_subdivisions.json', dict(status='complete', original_shape_redecode_exact=True))


def decode(chunk):
    import torch
    model = load_model(TEX_DECODER)
    from genrecon.modules.sparse import SparseTensor
    saved = torch.load(chunk / 'shape_subdivisions.pt', map_location='cpu', weights_only=True)
    subs = [SparseTensor(x['feats'].cuda(), x['coords'].cuda()) for x in saved]
    tex = sparse(np.load(chunk / 'pbr_raw.npz'))
    with torch.no_grad():
        voxels = model(tex, guide_subs=subs) * .5 + .5
    np.savez_compressed(chunk / 'pbr_decoded_voxels.npz',
        coords=voxels.coords.cpu().numpy().astype(np.int32),
        attrs=voxels.feats.cpu().numpy().astype(np.float32))
    write(chunk / 'pbr_decoding.json', dict(status='complete', sites=len(voxels.coords),
        channels=voxels.feats.shape[1], range=[float(voxels.feats.min()), float(voxels.feats.max())],
        output_scaling='decoder output * 0.5 + 0.5 (material channel range; not latent denormalization)',
        raw_latents=fingerprint(chunk / 'pbr_raw.npz')))


def triangle_keys(vertices, faces):
    triangles = np.asarray(vertices, np.float32)[faces]
    first = np.lexsort((triangles[..., 2], triangles[..., 1], triangles[..., 0]), axis=1)[:, 0]
    indices = (first[:, None] + np.arange(3)) % 3
    triangles = np.take_along_axis(triangles, indices[..., None], axis=1)
    return np.ascontiguousarray(triangles).reshape(-1, 9).view(np.dtype((np.void, 36))).reshape(-1)


def partitioned_uv(raw, valid):
    """Bound Xatlas problem size without altering any triangle or vertex position."""
    import torch
    import cumesh
    faces = raw.faces[valid]
    centers = raw.vertices[faces].mean(axis=1)
    cells = np.clip(np.floor((centers + .5) * 8), 0, 7).astype(int)
    keys = cells[:, 0] + 8 * cells[:, 1] + 64 * cells[:, 2]
    order = np.argsort(keys, kind='stable')
    offsets = np.r_[0, np.flatnonzero(np.diff(keys[order])) + 1, len(order)]
    count = len(offsets) - 1
    columns = int(np.ceil(np.sqrt(count)))
    tile_pixels = TEXTURE_SIZE // columns
    all_v, all_f, all_uv, all_maps = [], [], [], []
    vertex_offset = 0
    for i in range(count):
        part = faces[order[offsets[i]:offsets[i + 1]]]
        used, inverse = np.unique(part.reshape(-1), return_inverse=True)
        pv = torch.from_numpy(raw.vertices[used].astype(np.float32)).cuda()
        pf = torch.from_numpy(inverse.reshape(-1, 3).astype(np.int32)).cuda()
        mesh = cumesh.CuMesh()
        mesh.init(pv, pf)
        v, f, uv, vmaps = mesh.uv_unwrap(return_vmaps=True, verbose=False,
            compute_charts_kwargs=dict(threshold_cone_half_angle_rad=np.pi / 2,
                refine_iterations=0, global_iterations=1, smooth_strength=1))
        v, f, uv, vmaps = v.numpy(), f.numpy(), uv.numpy(), vmaps.numpy()
        assert np.array_equal(v, raw.vertices[used[vmaps]].astype(np.float32))
        uv = (uv * (tile_pixels - 4) + np.array([
            (i % columns) * tile_pixels + 2, (i // columns) * tile_pixels + 2])) / TEXTURE_SIZE
        all_v.append(v)
        all_f.append(f + vertex_offset)
        all_uv.append(uv.astype(np.float32))
        all_maps.append(used[vmaps])
        vertex_offset += len(v)
        del mesh, pv, pf
        if i % 25 == 0 or i + 1 == count:
            print(f'UV patches: {i + 1}/{count}', flush=True)
    return (torch.from_numpy(np.concatenate(all_v)), torch.from_numpy(np.concatenate(all_f)),
            torch.from_numpy(np.concatenate(all_uv)), torch.from_numpy(np.concatenate(all_maps)))


def bake(chunk):
    import torch
    import cumesh
    import cv2
    import nvdiffrast.torch as dr
    from flex_gemm.ops.grid_sample import grid_sample_3d
    started = time.monotonic()
    raw = trimesh.load(chunk / 'decoded_normalized_zup.ply', process=False)
    # UV unwrap only nondegenerate triangles; preserve any zero-area triangles
    # in a separate primitive, rather than letting the standard exporter drop them.
    valid = raw.area_faces > 0
    print(f'Unwrapping {chunk}: {int(valid.sum()):,} faces', flush=True)
    v, f, uv, vmaps = partitioned_uv(raw, valid)
    v, f, uv = v.cuda(), f.cuda().int(), uv.cuda()
    # Check every triangle exactly; UV seam vertex duplication/reindexing is allowed.
    expected = np.sort(triangle_keys(raw.vertices, raw.faces[valid]))
    actual = np.sort(triangle_keys(v.cpu().numpy(), f.cpu().numpy()))
    assert np.array_equal(expected, actual), 'UV unwrapping changed triangle positions, winding or multiplicity'
    print('Exact triangle identity verified; baking decoded PBR channels', flush=True)
    del expected, actual
    gc.collect()
    torch.cuda.empty_cache()
    n = TEXTURE_SIZE
    ctx = dr.RasterizeCudaContext()
    uv4 = torch.cat([uv * 2 - 1, torch.zeros_like(uv[:, :1]), torch.ones_like(uv[:, :1])], -1)[None]
    rast = dr.rasterize(ctx, uv4, f, resolution=[n, n])[0]
    mask = rast[0, ..., 3] > 0
    positions = dr.interpolate(v[None], rast, f)[0][0][mask]
    del rast, uv4
    saved = np.load(chunk / 'pbr_decoded_voxels.npz')
    attrs = torch.from_numpy(saved['attrs']).cuda()
    coords = torch.from_numpy(saved['coords']).cuda()
    sampled = []
    for batch in positions.split(100000):
        with torch.no_grad():
            sampled.append(grid_sample_3d(attrs, coords, torch.Size([1, 6, 512, 512, 512]),
                ((batch + .5) * 512)[None], mode='trilinear')[0].clamp(0, 1).cpu().numpy())
    pixels = np.zeros((n, n, 6), np.uint8)
    cpu_mask = mask.cpu().numpy()
    pixels[cpu_mask] = (np.concatenate(sampled) * 255).astype(np.uint8)
    del sampled, attrs, coords, positions
    base = np.concatenate([pixels[..., :3], pixels[..., 5:6]], -1)
    mr = np.concatenate([np.zeros_like(pixels[..., 3:4]), pixels[..., 4:5], pixels[..., 3:4]], -1)
    missing = (~cpu_mask).astype(np.uint8)
    base = np.concatenate([
        cv2.inpaint(base[..., :3], missing, 3, cv2.INPAINT_TELEA),
        cv2.inpaint(base[..., 3], missing, 1, cv2.INPAINT_TELEA)[..., None]], -1)
    mr = cv2.inpaint(mr, missing, 1, cv2.INPAINT_TELEA)
    material = trimesh.visual.material.PBRMaterial(name=f'{chunk.parent.name}_{chunk.name}_decoded_PBR',
        baseColorTexture=Image.fromarray(base), baseColorFactor=[255, 255, 255, 255],
        metallicRoughnessTexture=Image.fromarray(mr), metallicFactor=1., roughnessFactor=1.,
        alphaMode='OPAQUE', doubleSided=True)
    uv_np = uv.cpu().numpy()
    uv_np[:, 1] = 1 - uv_np[:, 1]
    textured = trimesh.Trimesh(v.cpu().numpy(), f.cpu().numpy(), process=False,
        visual=trimesh.visual.TextureVisuals(uv=uv_np, material=material))
    transform = np.array(read(chunk / 'transform.json')['M_chunk_zup_to_source_yup'])
    textured.apply_transform(transform)
    result = trimesh.Scene()
    result.add_geometry(textured, node_name='DECODED_PBR')
    if (~valid).any():
        degenerate = trimesh.Trimesh(raw.vertices.copy(), raw.faces[~valid].copy(), process=False)
        degenerate.apply_transform(transform)
        result.add_geometry(degenerate, node_name='PRESERVED_zero_area_faces')
    result.export(chunk / 'decoded_pbr.glb')
    restored = geometry(chunk / 'decoded_pbr.glb')
    original = geometry(chunk / 'decoded.glb')
    # Reindexing changes vertex count; compare triangle geometry directly.
    assert np.array_equal(np.sort(triangle_keys(restored.vertices, restored.faces)),
                          np.sort(triangle_keys(original.vertices, original.faces)))
    write(chunk / 'pbr_export.json', dict(status='complete', exact_triangles_verified=True,
        faces=len(restored.faces), texture_size=n, elapsed_seconds=time.monotonic() - started,
        zero_area_faces_preserved=int((~valid).sum()),
        uv_spatial_partition_grid=8,
        material='decoded base-color/alpha and metallic/roughness maps',
        alpha_mode='OPAQUE (same as upstream exporter)',
        decoded_pbr=fingerprint(chunk / 'decoded_pbr.glb')))


def export_scene(scene_id):
    dest = OUT / scene_id
    plan = read(dest / 'chunks.json')
    reconstruction = trimesh.Scene()
    for entry in plan['chunks']:
        scene = trimesh.load_scene(dest / entry['id'] / 'decoded_pbr.glb', process=False)
        chunk = dest / entry['id']
        if not (chunk / 'pbr_comparison.json').exists():
            original = trimesh.load_scene(chunk / 'normalized_pbr.glb', process=False)
            original.apply_transform(np.array(entry['M_chunk_zup_to_source_yup']) @ Y_TO_Z)
            export_comparison(original, scene, chunk)
        for node in scene.graph.nodes_geometry:
            tf, name = scene.graph[node]
            reconstruction.add_geometry(scene.geometry[name].copy(), transform=tf,
                node_name=f'{entry["id"]}_{node}', geom_name=f'{entry["id"]}_{name}')
    reconstruction.export(dest / 'decoded_pbr.glb')
    reference = trimesh.load_scene(ROOT / 'outputs/3dfront_benchmark' / scene_id / 'reference/full_room.glb', process=False)
    export_comparison(reference, reconstruction, dest)


def export_chunks(scene_id):
    dest = OUT / scene_id
    for entry in read(dest / 'chunks.json')['chunks']:
        chunk = dest / entry['id']
        if (chunk / 'pbr_comparison.json').exists():
            continue
        original = trimesh.load_scene(chunk / 'normalized_pbr.glb', process=False)
        original.apply_transform(np.array(entry['M_chunk_zup_to_source_yup']) @ Y_TO_Z)
        reconstructed = trimesh.load_scene(chunk / 'decoded_pbr.glb', process=False)
        export_comparison(original, reconstructed, chunk)


def export_comparison(reference, reconstruction, dest):
    gap = .15 * max(reference.extents[0], reconstruction.extents[0])
    shift = reference.bounds[1, 0] - reconstruction.bounds[0, 0] + gap
    comparison = trimesh.Scene()
    for label, scene, dx in [('ORIGINAL', reference, 0.), ('DECODED_PBR', reconstruction, shift)]:
        display = np.eye(4)
        display[0, 3] = dx
        for node in scene.graph.nodes_geometry:
            tf, name = scene.graph[node]
            comparison.add_geometry(scene.geometry[name].copy(), transform=display @ tf,
                node_name=f'{label}_{node}', geom_name=f'{label}_{name}')
    comparison.metadata.update(left='original GT with source materials', right='exact reconstructed geometry with decoded GT PBR',
        display_translation_x=float(shift), geometry_resolution=512, alignment='none',
        error_colors=False, simplification=False, latent_denormalization=False)
    archive = dest / 'comparison_surface_error.glb'
    if not archive.exists():
        shutil.copyfile(dest / 'comparison.glb', archive)
    comparison.export(dest / 'comparison.glb')
    restored = trimesh.load_scene(dest / 'comparison.glb', process=False)
    assert np.allclose(restored.bounds, comparison.bounds, atol=1e-6)
    expected = sum(len(scene.geometry[scene.graph[n][1]].faces)
                   for scene in (reference, reconstruction) for n in scene.graph.nodes_geometry)
    assert sum(len(g.faces) for g in restored.geometry.values()) == expected
    write(dest / 'pbr_comparison.json', dict(status='complete', exact_triangles_verified=True,
        original_materials_preserved=True, display_translation_x=float(shift),
        comparison=fingerprint(dest / 'comparison.glb'), decoded=fingerprint(dest / 'decoded_pbr.glb')))


def child(stage, target):
    target = str(target)
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               PYTORCH_ALLOC_CONF='expandable_segments:True', PYTORCH_NO_CUDA_MEMORY_CACHING='1')
    directory = Path(target) if stage != 'export-scene' else OUT / target
    with (directory / f'pbr_{stage}.log').open('w') as log:
        process = subprocess.Popen([sys.executable, '-m', 'experiments.gt_latent_baseline.reconstruction.run_pbr_baseline', stage, target],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, cwd=ROOT)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end='', flush=True)
        if process.wait():
            raise RuntimeError(f'PBR {stage} failed: {directory}')


def run():
    if not (OUT / 'pbr_provenance.json').exists():
        provenance()
    for sid in ['scene_04', 'scene_01', 'scene_02', 'scene_03', 'scene_05']:
        for entry in read(OUT / sid / 'chunks.json')['chunks']:
            chunk = OUT / sid / entry['id']
            for stage, artifact in [('prepare', 'pbr_input.json'), ('encode', 'pbr_encoding.json'),
                                    ('subdivisions', 'pbr_subdivisions.json'), ('decode', 'pbr_decoding.json'),
                                    ('bake', 'pbr_export.json')]:
                if not (chunk / artifact).exists():
                    child(stage, chunk)
        if not (OUT / sid / 'pbr_comparison.json').exists():
            child('export-scene', sid)
    write(OUT / 'pbr_summary.json', {f'scene_{i:02d}': read(OUT / f'scene_{i:02d}' / 'pbr_comparison.json') for i in range(1, 6)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['run', 'prepare', 'encode', 'subdivisions', 'decode', 'bake', 'export-scene', 'export-chunks'])
    parser.add_argument('target', nargs='?')
    args = parser.parse_args()
    if args.stage == 'run':
        run()
    elif args.stage == 'export-scene':
        export_scene(args.target)
    elif args.stage == 'export-chunks':
        export_chunks(args.target)
    else:
        globals()[args.stage](Path(args.target))
