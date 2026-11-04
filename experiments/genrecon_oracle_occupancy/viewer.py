"""Data for viewer/index.html: the scan, the baked meshes of bake.py and the cells stage 2 got.

For each scene, writes to outputs/genrecon_oracle_occupancy/viewer/<scene>/ (everything Y-up, as GLB and
three.js expect: the scan's (x, y, z) becomes (x, z, -y)):

    scan.glb               the laser scan with its colours, simplified
    <run>.glb              the baked mesh (a link to bake.py's scene.glb), run = baseline | oracle
    <run>_distance.glb     the same geometry, coloured by its distance to the scan
    <run>_cells.bin        centres of the cells stage 2 got (float32 xyz)
    meta.json              metrics, cell size and the camera of one test view

Usage:
    uv run experiments/genrecon_oracle_occupancy/viewer.py 5f99900f09 cc5237fd77 27dd4da69e
    python -m http.server -b 127.0.0.1 8765  # from the repository root, then open
    # http://127.0.0.1:8765/experiments/genrecon_oracle_occupancy/viewer/index.html
"""

import argparse
import json
import sys

import numpy as np
import open3d as o3d
import trimesh

from occupancy import BASELINE, GT, OUT, ROOT, layout, predicted_cells

sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_scannetpp as ev  # noqa: E402

VIEWER = OUT / "viewer"
SCAN_FACES = 1_500_000
# distance colour scale: (metres, RGB)
STOPS = [(0.0, (45, 95, 230)), (0.05, (60, 200, 130)), (0.10, (245, 215, 50)), (0.20, (220, 45, 40))]
Z_UP_TO_Y_UP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)


def scan_glb(scene: str, path):
    """The scan with its colours, without the anonymized faces, simplified for the browser."""
    scan = trimesh.load(ev.DATA / scene / "scans/mesh_aligned_0.05.ply", process=False)
    faces = ev.load_scan(ev.DATA / scene).faces
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(scan.vertices), o3d.utility.Vector3iVector(faces))
    mesh.vertex_colors = o3d.utility.Vector3dVector(scan.visual.vertex_colors[:, :3] / 255.0)
    mesh = mesh.simplify_quadric_decimation(SCAN_FACES)
    colours = (np.clip(np.asarray(mesh.vertex_colors), 0, 1) * 255).astype(np.uint8)
    trimesh.Trimesh(np.asarray(mesh.vertices) @ Z_UP_TO_Y_UP.T, np.asarray(mesh.triangles), vertex_colors=colours,
                    process=False).export(path)


def distance_glb(baked, scan_scene, path):
    """The baked geometry, coloured by its distance to the scan."""
    mesh = trimesh.load(baked, force="mesh", process=False)  # chunks merged, textures dropped
    vertices = (mesh.vertices @ Z_UP_TO_Y_UP).astype(np.float32)  # back to Z-up
    distance = scan_scene.compute_distance(o3d.core.Tensor(vertices)).numpy()
    colours = np.stack([np.interp(distance, [d for d, _ in STOPS], [c[i] for _, c in STOPS]) for i in range(3)], 1)
    trimesh.Trimesh(mesh.vertices, mesh.faces, vertex_colors=colours.astype(np.uint8), process=False).export(path)


def test_camera(scene: str, scan_scene) -> dict:
    """The test view that shows most of the room (largest median scan depth), as a three.js camera."""
    views = list(ev.test_views(ev.DATA / scene))
    depths = [np.median(np.nan_to_num(ev.render_depth(scan_scene, K, c2w, v.shape[1], v.shape[0]), posinf=0))
              for K, c2w, v in views]
    K, c2w, valid = views[int(np.argmax(depths))]
    return {"position": (c2w[:3, 3] @ Z_UP_TO_Y_UP.T).tolist(),
            "target": ((c2w[:3, 3] + c2w[:3, 2]) @ Z_UP_TO_Y_UP.T).tolist(),  # 1 m ahead
            "fov": float(np.degrees(2 * np.arctan(valid.shape[0] / 2 / K[1, 1])))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenes", nargs="+")
    args = parser.parse_args()

    for scene in args.scenes:
        out = VIEWER / scene
        out.mkdir(parents=True, exist_ok=True)
        if not (out / "scan.glb").exists():
            scan_glb(scene, out / "scan.glb")
        scan_scene = ev.raycasting_scene(ev.load_scan(ev.DATA / scene))
        lay = layout(BASELINE / "32views" / scene)
        meta = {"cell": lay["cell"], "camera": test_camera(scene, scan_scene), "runs": {}}
        cells = {"baseline": predicted_cells(BASELINE / "32views" / scene, lay),
                 "oracle": np.load(GT / f"32views/{scene}.npz")["visible"].astype(int)}
        for run, run_dir in {"baseline": BASELINE, "oracle": OUT}.items():
            baked = OUT / "baked" / run / scene / "scene.glb"
            (out / f"{run}.glb").unlink(missing_ok=True)
            (out / f"{run}.glb").symlink_to(baked)
            distance_glb(baked, scan_scene, out / f"{run}_distance.glb")
            centres = lay["origin"] + (cells[run] + 0.5) * lay["cell"]
            (centres @ Z_UP_TO_Y_UP.T).astype(np.float32).tofile(out / f"{run}_cells.bin")
            scene_dir = run_dir / "32views" / scene
            meta["runs"][run] = {**json.loads((scene_dir / "metrics.json").read_text()),
                                 **json.loads((scene_dir / "fscores.json").read_text()), "cells": len(centres)}
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"wrote {out}", flush=True)

    scenes = sorted(d.name for d in VIEWER.iterdir() if (d / "meta.json").exists())
    (VIEWER / "scenes.json").write_text(json.dumps(scenes))


if __name__ == "__main__":
    main()
