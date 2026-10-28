"""Export full-room and furniture-only GLBs for manual inspection.

Preserve source glTF y-up coordinates, materials, and mesh detail. These files
are source reference geometry, not GenRecon reconstructions. Scale to original
metres is unknown.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def main(root, output):
    manifest = json.loads((root / "manifest.json").read_text())
    room = manifest["room"]
    source = root / f"3D-FRONT-TEST-SCENE/{room}.glb"
    shell_paths = [source.with_suffix('')/name for name in ('ceil.glb','wall.glb','floor.glb','others.glb')]
    shell_paths = [p for p in shell_paths if p.exists()]
    output.mkdir(parents=True, exist_ok=True)
    furniture = trimesh.load_scene(source, process=False)
    full = furniture.copy()
    exports = [("furniture_only.glb", furniture)]
    if shell_paths:
        for shell_path in shell_paths:
            shell = trimesh.load_scene(shell_path, process=False)
            for index, node in enumerate(shell.graph.nodes_geometry):
                transform, name = shell.graph[node]
                label=f'Room_context_{shell_path.stem}_{index}'
                full.add_geometry(shell.geometry[name].copy(), transform=transform,
                                  node_name=label, geom_name=label)
        exports.append(("full_room.glb", full))
    else:
        print(f'No supplementary assets supplied for {room}; exporting furniture only.')
    for name, scene in exports:
        path = output / name
        scene.metadata.update({"source_room": room, "source_revision": manifest["revision"],
                               "metric_scale_verified": False, "coordinates": "source glTF y-up"})
        scene.export(path, file_type="glb")
        restored = trimesh.load_scene(path, process=False)
        expected_faces = sum(len(g.faces) for g in scene.geometry.values())
        actual_faces = sum(len(g.faces) for g in restored.geometry.values())
        if expected_faces != actual_faces or not np.allclose(scene.bounds, restored.bounds):
            raise RuntimeError(f"Export round-trip changed geometry: {path}")
        if len(scene.graph.nodes_geometry) != len(restored.graph.nodes_geometry):
            raise RuntimeError(f"Export round-trip changed object count: {path}")
        print(f"{path}: {path.stat().st_size / 1024**2:.2f} MiB; "
              f"{len(restored.graph.nodes_geometry)} objects; {actual_faces:,} triangles")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/3dfront/pilot"))
    parser.add_argument("--output", type=Path, default=Path("outputs/3dfront_validation/inspection"))
    args = parser.parse_args()
    main(args.data, args.output)
