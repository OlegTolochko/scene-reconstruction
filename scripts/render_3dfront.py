"""Render metric furniture + shell from interior cameras (Blender 4.5+)."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree


def bvh(objects):
    vertices, faces = [], []
    for obj in objects:
        offset = len(vertices)
        vertices.extend(obj.matrix_world @ v.co for v in obj.data.vertices)
        faces.extend(tuple(offset + i for i in p.vertices) for p in obj.data.polygons)
    return BVHTree.FromPolygons(vertices, faces), np.array(vertices)


def inside(tree, position):
    """Ray parity keeps cameras out of closed furniture volumes."""
    direction = Vector((1, .371, .127)).normalized()
    for crossings in range(10000):
        hit = tree.ray_cast(position, direction)[0]
        if hit is None:
            return crossings % 2 == 1
        position = hit + direction * 1e-5
    return True


def metric_transform(scene_json, room_id, shell_objects):
    """Recover MIDI's normalization from matching original room architecture.

    Original JSON mesh xyz values are world-space metres, as read by the
    official 3D-FRONT toolbox (scripts/scene.py). Blender imports GLBs Z-up.
    """
    source = scene_json.read_bytes()
    data = json.loads(source)
    house, room_name = room_id.split('/')
    if data['uid'] != house:
        raise ValueError('Original scene JSON belongs to a different house')
    rooms = [r for r in data['scene']['room'] if r['instanceid'] == room_name]
    if len(rooms) != 1:
        raise ValueError(f'Original scene JSON must contain exactly one room {room_name}')
    refs = {c['ref'] for c in rooms[0]['children']}
    vertices, faces = [], []
    for mesh in data['mesh']:
        if mesh['uid'] not in refs:
            continue
        offset = len(vertices)
        vertices.extend(np.asarray(mesh['xyz'], dtype=float).reshape(-1, 3))
        faces.extend(np.asarray(mesh['faces'], dtype=int).reshape(-1, 3) + offset)
    if not faces:
        raise ValueError('Original room has no architectural triangles')
    original = np.asarray(vertices)[:, [0, 2, 1]] * [1, -1, 1]  # Y-up -> Z-up.
    normalized_tree, normalized = bvh(shell_objects)
    original_tree = BVHTree.FromPolygons([Vector(v) for v in original], faces, all_triangles=True)
    original_extent, normalized_extent = np.ptp(original, axis=0), np.ptp(normalized, axis=0)
    if (not np.isfinite(original).all() or not np.isfinite(normalized).all()
            or np.any(original_extent <= 0) or np.any(normalized_extent <= 0)):
        raise ValueError('Expected finite room architecture with nonzero extent on every axis')
    scale = float(np.median(original_extent / normalized_extent))
    if not np.allclose(normalized_extent * scale, original_extent, rtol=1e-4, atol=1e-4):
        raise ValueError('MIDI shell does not match original room extents under a uniform scale')
    translation = (original.min(0) + original.max(0)) / 2 - scale * (normalized.min(0) + normalized.max(0)) / 2
    # Probe vertices and triangle interiors, so a missing floor cannot pass on its edges.
    probes = list(normalized)
    for obj in shell_objects:
        obj.data.calc_loop_triangles()
        probes.extend(obj.matrix_world @ t.center for t in obj.data.loop_triangles)
    original_probes = np.concatenate([original, original[np.asarray(faces)].mean(axis=1)])
    errors = [original_tree.find_nearest(Vector(np.asarray(v) * scale + translation))[3] for v in probes]
    errors += [normalized_tree.find_nearest(Vector((v - translation) / scale))[3] * scale for v in original_probes]
    error = float(max(errors))
    if not np.isfinite(error) or error > .005:
        raise ValueError(f'MIDI shell does not match original room surfaces (max error {error:.6f} m)')
    transform = np.diag([scale, scale, scale, 1.])
    transform[:3, 3] = translation
    evidence = {'scene_json': str(scene_json.resolve()), 'sha256': hashlib.sha256(source).hexdigest(),
                'room': room_id, 'max_surface_error_m': error, 'surface_tolerance_m': .005,
                'surface_probes': 'vertices and triangle centroids in both directions',
                'normalized_to_world': transform.tolist()}
    return transform, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Downloaded room with manifest.json')
    parser.add_argument('--output', type=Path, required=True, help='Empty render directory')
    parser.add_argument('--front-json', type=Path, required=True, help='Original 3D-FRONT <house>.json in metric units')
    parser.add_argument('--views', type=int, default=128)
    parser.add_argument('--resolution', type=int, default=768)
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if min(args.views, args.resolution, args.samples) <= 0:
        parser.error('View count, resolution and samples must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('--output must be empty')
    manifest = json.loads((args.data / 'manifest.json').read_text())
    room = args.data / '3D-FRONT-TEST-SCENE' / manifest['room']
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=str(room.with_suffix('.glb').resolve()))
    furniture = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    shell_files = [room / (name + '.glb') for name in ('ceil', 'floor', 'wall', 'others')]
    shell_files = [p for p in shell_files if p.is_file()]
    if not shell_files:
        parser.error('Room has no architectural GLBs')
    for path in shell_files:
        bpy.ops.import_scene.gltf(filepath=str(path.resolve()))
    objects = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    shell_objects = [o for o in objects if o not in furniture]
    transform, scale_source = metric_transform(args.front_json, manifest['room'], shell_objects)
    for obj in bpy.context.scene.objects:
        if obj.parent is None:
            obj.matrix_world = Matrix(transform.tolist()) @ obj.matrix_world
    bpy.context.view_layer.update()
    shell, vertices = bvh(shell_objects)
    scene_bvh, _ = bvh(objects)
    obstacles = [bvh([o])[0] for o in furniture]
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    rng = np.random.default_rng(args.seed)
    positions = []
    for _ in range(args.views * 1000):
        position = Vector((*rng.uniform(low[:2], high[:2]), low[2] + 1.5))
        hits = [shell.ray_cast(position, Vector(d)) for d in
                ((0, 0, -1), (0, 0, 1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0))]
        if any(h[0] is None or h[3] < .25 for h in hits) or abs(hits[0][3] - 1.5) > .15:
            continue
        if scene_bvh.find_nearest(position)[3] < .25 or any(inside(t, position) for t in obstacles):
            continue
        positions.append(position)
        if len(positions) == args.views:
            break
    if len(positions) != args.views:
        raise ValueError('Cannot place interior cameras; check metric scale and shell completeness')
    args.output.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:
        obj.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str((args.output / 'reference.glb').resolve()), use_selection=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples, scene.cycles.seed = args.samples, args.seed
    scene.render.resolution_x = scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.film_transparent = False
    for position in positions[:8]:
        lamp = bpy.data.lights.new('interior', 'POINT')
        lamp.energy, lamp.shadow_soft_size = 20, .3
        obj = bpy.data.objects.new(lamp.name, lamp)
        scene.collection.objects.link(obj)
        obj.location = position + Vector((0, 0, .2))
    camera = bpy.data.objects.new('Camera', bpy.data.cameras.new('Camera'))
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.sensor_fit, camera.data.sensor_width, camera.data.lens = 'HORIZONTAL', 36, 18
    camera.data.clip_start = .01
    frames = []
    (args.output / 'rgb').mkdir()
    for i, position in enumerate(positions):
        yaw, pitch = rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi / 6, np.pi / 6)
        direction = Vector((np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)))
        camera.location, camera.rotation_euler = position, direction.to_track_quat('-Z', 'Y').to_euler()
        bpy.context.view_layer.update()
        name = f'{i:04d}.png'
        scene.render.filepath = str((args.output / 'rgb' / name).resolve())
        bpy.ops.render.render(write_still=True)
        frames.append({'file_path': name, 'transform_matrix': [list(r) for r in camera.matrix_world]})
    (args.output / 'transforms.json').write_text(json.dumps({
        'room': manifest['room'], 'revision': manifest['revision'], 'units': 'metres',
        'meters_per_unit': float(transform[0, 0]), 'scale_source': scale_source, 'shell_included': True,
        'shell_files': [p.name for p in shell_files],
        'w': args.resolution, 'h': args.resolution, 'fl_x': args.resolution / 2,
        'fl_y': args.resolution / 2, 'cx': args.resolution / 2, 'cy': args.resolution / 2,
        'frames': frames}, indent=2) + '\n')


if __name__ == '__main__':
    main()
