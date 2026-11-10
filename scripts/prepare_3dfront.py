"""Write a 3D-FRONT render (scripts/render_3dfront.py) as a scene in the ScanNet++ layout.

The shared ScanNet++ scripts then run on 3D-FRONT unchanged, with --data data/3dfront:
select_views_scannetpp.py, experiments/genrecon_baseline/run.py (GenRecon --mode Scannet_colmap)
and evaluate_scannetpp.py.

Writes <output>/data/<scene>/ and adds <scene> to <output>/splits/nvs_sem_val.txt:
    scans/mesh_aligned_0.05.ply             reference mesh (furniture + shell), metres, Z-up
    dslr/nerfstudio/transforms_undistorted.json
        frames: the renders available as inputs; test_frames: held-out views, picked first by
        farthest-point sampling of the camera centres
    dslr/resized_undistorted_images/        the renders
    dslr/resized_undistorted_masks/         all pixels valid
    dslr/colmap/                            support points for chunk placement, see below

ScanNet++ places chunks with COLMAP points from all DSLR frames. Here the points come from
GT depth of all non-test renders. Each point lists the renders that see it (depth test) as its
track, with reprojection error 0, so GenRecon's default point filtering applies unchanged.

Usage:
    papers/GenRecon/.venv/bin/python scripts/prepare_3dfront.py --renders RENDER_DIR --scene NAME
"""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial.transform import Rotation
import trimesh

Y_TO_Z = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
# Inverse of the remap GenRecon and evaluate_scannetpp.py apply to ScanNet++ nerfstudio poses
# (the matrix is its own inverse), so the poses land in the mesh frame after their remap.
NERFSTUDIO_TO_SCAN = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], float)
OPENGL_TO_OPENCV = np.diag([1., -1., -1., 1.])
VISIBILITY_TOLERANCE = 0.01  # metres


def farthest_views(frames, count):
    """Farthest-point sampling of camera centres, starting at the first frame."""
    centres = np.array([np.asarray(f['transform_matrix'])[:3, 3] for f in frames])
    picked = [0]
    distance = np.linalg.norm(centres - centres[0], axis=1)
    while len(picked) < count:
        picked.append(int(distance.argmax()))
        distance = np.minimum(distance, np.linalg.norm(centres - centres[picked[-1]], axis=1))
    return {frames[i]['file_path'] for i in picked}


def support_points(mesh, frames, metadata, renders, count, seed):
    """Points on the GT surface seen by the frames, with colours and the frames that see them."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)),
                        o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.uint32)))
    w, h = metadata['w'], metadata['h']
    K = np.array([[metadata['fl_x'], 0, metadata['cx']], [0, metadata['fl_y'], metadata['cy']], [0, 0, 1.]])
    poses = [np.asarray(f['transform_matrix']) @ OPENGL_TO_OPENCV for f in frames]  # OpenCV cam-to-world
    # Pixel stride for about 4x the requested points before subsampling, at any resolution.
    stride = max(1, int(np.sqrt(w * h * len(frames) / (4 * count))))
    v, u = np.mgrid[0:h:stride, 0:w:stride]
    rays = np.stack([(u + .5 - K[0, 2]) / K[0, 0], (v + .5 - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=-1)
    points, colours = [], []
    for frame, pose in zip(frames, poses):
        directions = rays @ pose[:3, :3].T
        origins = np.broadcast_to(pose[:3, 3], directions.shape)
        depth = scene.cast_rays(o3d.core.Tensor(
            np.concatenate([origins, directions], axis=-1).astype(np.float32)))['t_hit'].numpy()
        hit = np.isfinite(depth)
        points.append(directions[hit] * depth[hit, None] + pose[:3, 3])
        with Image.open(renders / 'rgb' / frame['file_path']) as image:
            colours.append(np.asarray(image.convert('RGB'))[v[hit], u[hit]])
    points, colours = np.concatenate(points), np.concatenate(colours)
    if not len(points):
        raise ValueError('The renders see no GT surface')
    picks = np.random.default_rng(seed).choice(len(points), min(count, len(points)), replace=False)
    points, colours = points[picks], colours[picks]

    # A point is seen by a frame if it projects into the image and nothing lies in front of it.
    seen = np.zeros((len(points), len(frames)), dtype=bool)
    for j, pose in enumerate(poses):
        camera = (points - pose[:3, 3]) @ pose[:3, :3]
        z = camera[:, 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            x, y = K[0, 0] * camera[:, 0] / z + K[0, 2], K[1, 1] * camera[:, 1] / z + K[1, 2]
        candidate = np.flatnonzero((z > 0) & (x >= 0) & (x < w) & (y >= 0) & (y < h))
        offset = points[candidate] - pose[:3, 3]
        distance = np.linalg.norm(offset, axis=1)
        query = np.concatenate([np.broadcast_to(pose[:3, 3], offset.shape), offset / distance[:, None]], axis=1)
        t_hit = scene.cast_rays(o3d.core.Tensor(query.astype(np.float32)))['t_hit'].numpy()
        seen[candidate[t_hit >= distance - VISIBILITY_TOLERANCE], j] = True
    return points, colours, seen, poses, K


def write_colmap(directory, frames, poses, K, metadata, points, colours, seen):
    directory.mkdir(parents=True)
    (directory / 'cameras.txt').write_text(
        f"1 PINHOLE {metadata['w']} {metadata['h']} {K[0, 0]} {K[1, 1]} {K[0, 2]} {K[1, 2]}\n")
    with (directory / 'images.txt').open('w') as handle:
        for i, (frame, pose) in enumerate(zip(frames, poses), 1):
            world_to_camera = np.linalg.inv(pose)
            q = Rotation.from_matrix(world_to_camera[:3, :3]).as_quat()  # x, y, z, w
            handle.write(f"{i} {q[3]} {q[0]} {q[1]} {q[2]} {' '.join(map(str, world_to_camera[:3, 3]))} "
                         f"1 {frame['file_path']}\n\n")  # no 2D observations are stored
    with (directory / 'points3D.txt').open('w') as handle:
        handle.write('# ID X Y Z R G B ERROR TRACK[] as (IMAGE_ID, POINT2D_IDX); '
                     'GT points, the track lists the images that see the point, POINT2D_IDX is unused (0)\n')
        for i, (xyz, rgb, row) in enumerate(zip(points, colours, seen), 1):
            track = ' '.join(f'{j + 1} 0' for j in np.flatnonzero(row))
            handle.write(f"{i} {xyz[0]} {xyz[1]} {xyz[2]} {rgb[0]} {rgb[1]} {rgb[2]} 0 {track}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--renders', type=Path, required=True, help='output of scripts/render_3dfront.py')
    parser.add_argument('--output', type=Path, default=Path('data/3dfront'), help='dataset root')
    parser.add_argument('--scene', required=True, help='scene ID (directory name)')
    parser.add_argument('--test-views', type=int, default=32)
    parser.add_argument('--points', type=int, default=500_000, help='support points for chunk placement')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not args.scene or Path(args.scene).name != args.scene or args.scene in ('.', '..'):
        parser.error('--scene must be a single directory name')
    metadata = json.loads((args.renders / 'transforms.json').read_text())
    evidence = metadata.get('scale_source')
    if (metadata.get('units') != 'metres' or not metadata.get('shell_included')
            or not isinstance(evidence, dict) or evidence.get('room') != metadata['room']
            or not 0 <= evidence.get('max_surface_error_m', float('inf')) <= .005):
        parser.error('Render with --front-json: require a verified match to original metric room geometry')
    frames = sorted(metadata['frames'], key=lambda f: f['file_path'])
    if not 0 < args.test_views < len(frames):
        parser.error(f'--test-views must leave input views among the {len(frames)} renders')
    scene = args.output / 'data' / args.scene
    if scene.exists():
        parser.error(f'{scene} already exists')

    test = farthest_views(frames, args.test_views)
    inputs = [f for f in frames if f['file_path'] not in test]
    mesh = trimesh.load_scene(args.renders / 'reference.glb', process=False).to_mesh()
    mesh.apply_transform(Y_TO_Z)  # glTF Y-up -> Blender's Z-up world, as the cameras
    points, colours, seen, poses, K = support_points(mesh, inputs, metadata, args.renders, args.points, args.seed)

    (scene / 'scans').mkdir(parents=True)
    mesh.export(scene / 'scans/mesh_aligned_0.05.ply')
    (scene / 'scans/mesh_aligned_0.05_mask.txt').write_text('')  # nothing is anonymized
    dslr = scene / 'dslr'
    (dslr / 'resized_undistorted_masks').mkdir(parents=True)
    Image.new('L', (metadata['w'], metadata['h']), 255).save(dslr / 'resized_undistorted_masks/valid.png')
    (dslr / 'resized_undistorted_images').symlink_to((args.renders / 'rgb').resolve())

    def frame(f):
        pose = NERFSTUDIO_TO_SCAN @ np.asarray(f['transform_matrix'])
        return {**f, 'transform_matrix': pose.tolist(), 'mask_path': 'valid.png', 'is_bad': False}

    (dslr / 'nerfstudio').mkdir()
    (dslr / 'nerfstudio/transforms_undistorted.json').write_text(json.dumps({
        **{k: metadata[k] for k in ('w', 'h', 'fl_x', 'fl_y', 'cx', 'cy')},
        'frames': [frame(f) for f in inputs], 'test_frames': [frame(f) for f in frames if f['file_path'] in test],
    }, indent=2) + '\n')
    write_colmap(dslr / 'colmap', inputs, poses, K, metadata, points, colours, seen)
    (scene / 'scene_manifest.json').write_text(json.dumps({
        'renders': str(args.renders.resolve()), 'room': metadata['room'], 'revision': metadata['revision'],
        'units': 'metres', 'shell_included': True, 'meters_per_unit': metadata['meters_per_unit'],
        'scale_source': evidence, 'test_views': sorted(test), 'seed': args.seed,
        'support_points': f'{len(points)} GT-depth points from the {len(inputs)} non-test renders',
        'support_points_seen_by_4_or_more': int((seen.sum(axis=1) >= 4).sum()),
    }, indent=2) + '\n')

    split = args.output / 'splits/nvs_sem_val.txt'
    split.parent.mkdir(parents=True, exist_ok=True)
    scenes = split.read_text().split() if split.exists() else []
    split.write_text('\n'.join(sorted({*scenes, args.scene})) + '\n')
    print(f'Wrote {scene}; next: uv run scripts/select_views_scannetpp.py 32 --data {args.output}')


if __name__ == '__main__':
    main()
