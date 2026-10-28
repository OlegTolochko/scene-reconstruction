"""Put the reference furniture and GenRecon output side by side in one GLB.

Both inputs must use the reference glTF y-up axes and dataset scale. The only
display adjustment is an x translation of the prediction; no scale fitting,
alignment optimization, mesh simplification, or material changes are applied.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[3]


def main(prediction_path, output, data=ROOT / 'data/3dfront/pilot'):
    manifest = json.loads((data / 'manifest.json').read_text())
    reference_path = data / f"3D-FRONT-TEST-SCENE/{manifest['room']}.glb"
    reference = trimesh.load_scene(reference_path, process=False)
    prediction = trimesh.load_scene(prediction_path, process=False)
    if not prediction.geometry:
        raise RuntimeError("Prediction contains no geometry")
    gap = .2 * max(reference.extents[0], prediction.extents[0])
    shift = reference.bounds[1, 0] - prediction.bounds[0, 0] + gap
    combined = trimesh.Scene()
    expected_faces = 0
    for label, scene, translation in (("REFERENCE", reference, 0.), ("GENRECON", prediction, shift)):
        display = np.eye(4)
        display[0, 3] = translation
        for index, node in enumerate(scene.graph.nodes_geometry):
            transform, geometry = scene.graph[node]
            mesh = scene.geometry[geometry].copy()
            expected_faces += len(mesh.faces)
            combined.add_geometry(mesh, transform=display @ transform,
                                  node_name=f"{label}_{index}_{node}",
                                  geom_name=f"{label}_{index}_{geometry}")
    combined.metadata.update({"left": "REFERENCE furniture", "right": "GENRECON reconstruction",
                              "prediction_display_x_translation": float(shift),
                              "metric_scale_verified": False})
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.export(output, file_type="glb")
    restored = trimesh.load_scene(output, process=False)
    actual_faces = sum(len(restored.geometry[restored.graph[node][1]].faces)
                       for node in restored.graph.nodes_geometry)
    if actual_faces != expected_faces or not np.allclose(restored.bounds, combined.bounds):
        raise RuntimeError("Comparison GLB round-trip changed geometry")
    print(f"{output}: reference left; reconstruction right; {actual_faces:,} triangles")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path,
                        default=ROOT / "outputs/genrecon/3dfront_pilot/scene_reference_frame.glb")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/genrecon/3dfront_pilot/comparison_side_by_side.glb")
    parser.add_argument('--data',type=Path,default=ROOT/'data/3dfront/pilot')
    args = parser.parse_args()
    main(args.prediction, args.output, args.data)
