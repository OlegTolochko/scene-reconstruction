"""Inspect one MIDI room and compare released depth with CPU mesh ray casting.

This validates dataset consistency, not reconstruction quality. Distances are
reported in dataset coordinates until an original metric scale is available.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import OpenEXR
from PIL import Image
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def valid_depth(depth):
    # The pilot EXRs use float16's maximum finite value for background.
    return np.isfinite(depth) & (depth > 0) & (depth < 65504)


def load_mesh(path):
    scene = trimesh.load_scene(path, process=False)
    return scene, scene.to_mesh()


def read_depth(path):
    with OpenEXR.File(str(path), separate_channels=True) as source:
        channels = source.channels()
        key = "V" if "V" in channels else "R" if "R" in channels else next(iter(channels))
        return np.asarray(channels[key].pixels, dtype=np.float64)


def ray_depth(mesh, c2w, fov, xs, ys, width, height):
    focal = width / (2 * np.tan(fov / 2))
    local = np.stack([(xs + .5 - width / 2) / focal,
                      -(ys + .5 - height / 2) / focal, -np.ones_like(xs)], axis=-1)
    directions = local @ c2w[:3, :3].T
    origins = np.broadcast_to(c2w[:3, 3], directions.shape).copy()
    points, rays, _ = mesh.ray.intersects_location(origins, directions, multiple_hits=False)
    distance = np.full(len(xs), np.nan)
    zdepth = np.full(len(xs), np.nan)
    distance[rays] = np.linalg.norm(points - origins[rays], axis=-1)
    zdepth[rays] = np.einsum("ij,ij->i", points - origins[rays], directions[rays]) / np.sum(directions[rays]**2, axis=-1)
    return {"ray_distance": distance, "camera_z": zdepth}


def main(root, output, grid):
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_text())
    for record in manifest["files"]:
        with (root / record["path"]).open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != record["sha256"]:
                raise RuntimeError(f"Checksum mismatch: {record['path']}")
    room = manifest["room"]
    render_dir = root / f"3D-FRONT-TEST-RENDER/{room}"
    mesh_path = root / f"3D-FRONT-TEST-SCENE/{room}.glb"
    scene, mesh = load_mesh(mesh_path)
    meta = json.loads((render_dir / "meta.json").read_text())
    rotations = {
        "as_stored": np.eye(4),
        "gltf_y_up_to_blender_z_up": np.array([[1,0,0,0], [0,0,-1,0], [0,1,0,0], [0,0,0,1]]),
    }
    candidates = {}
    for orientation, transform in rotations.items():
        candidate = mesh.copy()
        candidate.apply_transform(transform)
        candidates[orientation] = candidate
    object_info = []
    structural_mesh = None
    for path in sorted(mesh_path.with_suffix("").glob("*.glb")):
        _, obj = load_mesh(path)
        object_info.append({"file": path.name, "bounds": obj.bounds.tolist(),
                            "extents": obj.extents.tolist(), "faces": len(obj.faces),
                            "watertight": bool(obj.is_watertight)})
        if path.name == "ceil.glb":
            structural_mesh = obj.copy()
            structural_mesh.apply_transform(rotations["gltf_y_up_to_blender_z_up"])
    missing = sorted({f["name"] for camera in meta["locations"] for f in camera["frames"]
                      if not (render_dir / f["name"]).exists()})
    results = []
    views = []
    depth_stats = []
    diagnostic = None
    for camera in meta["locations"]:
        index = camera["index"]
        depth_path = render_dir / f"depth_{index}.exr"
        rgb_path = render_dir / f"render_{index}.webp"
        if not depth_path.exists() or not rgb_path.exists():
            raise RuntimeError(f"Missing sequential pair for {index}")
        depth = read_depth(depth_path)
        height, width = depth.shape
        image_size = Image.open(rgb_path).size
        if image_size != (width, height):
            raise RuntimeError(f"RGB/depth size mismatch: {index}")
        finite = valid_depth(depth)
        depth_stats.append({"index": index, "width": width, "height": height,
                            "valid_fraction": float(finite.mean()),
                            "background_65504_pixels": int((depth == 65504).sum()),
                            "minimum": float(depth[finite].min()), "maximum": float(depth[finite].max())})
        c2w = np.asarray(camera["transform_matrix"], dtype=float)
        if not np.allclose(c2w[3], [0,0,0,1]) or not np.allclose(c2w[:3,:3].T @ c2w[:3,:3], np.eye(3), atol=1e-5):
            raise RuntimeError(f"Invalid camera rotation: {index}")
        axis = np.linspace(0, width - 1, grid).round().astype(int)
        xx, yy = np.meshgrid(axis, np.linspace(0, height-1, grid).round().astype(int))
        xs, ys = xx.ravel(), yy.ravel()
        views.append((index, rgb_path, depth))
        for orientation, candidate in candidates.items():
            projections = ray_depth(candidate, c2w, meta["camera_angle_x"], xs, ys, width, height)
            if index == "0000" and orientation == "gltf_y_up_to_blender_z_up":
                diagnostic = (rgb_path, depth, xs, ys, projections["camera_z"])
            mappings = {"sequential_index": depth_path}
            named = next(f["name"] for f in camera["frames"] if f["type"] == "depth")
            if (render_dir / named).exists():
                mappings["metadata_frame_name"] = render_dir / named
            for mapping, target in mappings.items():
                reference = read_depth(target)[ys, xs]
                valid_gt = valid_depth(reference)
                for depth_mode, predicted in projections.items():
                    valid_pred = np.isfinite(predicted)
                    both = valid_gt & valid_pred
                    union = valid_gt | valid_pred
                    errors = np.abs(predicted[both] - reference[both])
                    relative = errors / reference[both]
                    results.append({"camera_index": index, "mapping": mapping, "orientation": orientation,
                        "depth_mode": depth_mode, "compared_pixels": int(both.sum()),
                        "silhouette_iou": float(both.sum() / max(1, union.sum())),
                        "median_absolute_error": float(np.median(errors)) if both.any() else None,
                        "median_relative_error": float(np.median(relative)) if both.any() else None,
                        "fraction_within_one_percent": float((relative < .01).mean()) if both.any() else None})
        print(f"Checked camera {index}", flush=True)
    summary = []
    for orientation in rotations:
        for mapping in ("sequential_index", "metadata_frame_name"):
            for mode in ("ray_distance", "camera_z"):
                rows = [r for r in results if r["orientation"] == orientation and r["mapping"] == mapping
                        and r["depth_mode"] == mode and r["compared_pixels"]]
                if rows:
                    summary.append({"orientation": orientation, "mapping": mapping, "depth_mode": mode,
                                    "views": len(rows),
                                    "mean_silhouette_iou": float(np.mean([r["silhouette_iou"] for r in rows])),
                                    "median_view_relative_error": float(np.median([r["median_relative_error"] for r in rows]))})
    summary.sort(key=lambda r: r["median_view_relative_error"])
    expected = [r for r in results if r["orientation"] == "gltf_y_up_to_blender_z_up"
                and r["mapping"] == "sequential_index" and r["depth_mode"] == "camera_z"]
    alignment_passed = len(expected) == len(views) and all(
        r["compared_pixels"] >= 20 and r["silhouette_iou"] >= .95
        and r["median_relative_error"] < .01 for r in expected)
    report = {"room": room, "revision": manifest["revision"], "retained_bytes": manifest["retained_bytes"],
              "camera_count": len(views), "mesh_bounds_as_stored": mesh.bounds.tolist(),
              "mesh_extents_as_stored": mesh.extents.tolist(), "vertices": len(mesh.vertices), "faces": len(mesh.faces),
              "geometry_names": list(scene.geometry), "object_files": object_info,
              "missing_metadata_frame_files": missing, "ray_grid": grid,
              "depth_background_value": 65504, "depth_statistics": depth_stats,
              "metric_scale_verified": False, "camera_hypotheses": summary, "per_view_checks": results}
    report["alignment_check_passed"] = alignment_passed
    report["alignment_check_rule"] = "All views: >=20 paired rays, silhouette IoU >=0.95, median relative depth error <0.01"
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    figure, axes = plt.subplots(3, 4, figsize=(12, 9))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (index, path, depth) in zip(axes.ravel(), views):
        ax.imshow(Image.open(path)); ax.set_title(f"render_{index}")
    figure.tight_layout(); figure.savefig(output / "rgb_contact_sheet.png", dpi=130); plt.close(figure)
    path, depth, xs, ys, predicted = diagnostic
    reference = depth[ys, xs]
    valid_pred = np.isfinite(predicted)
    good = valid_pred & valid_depth(reference) & (np.abs(predicted - reference) / reference < .01)
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(Image.open(path)); axes[0].set_title("Released RGB: view 0000")
    image_depth = np.where(valid_depth(depth), depth, np.nan)
    im = axes[1].imshow(image_depth, cmap="viridis"); axes[1].set_title("Released camera-axis depth")
    figure.colorbar(im, ax=axes[1], fraction=.046, label="Dataset units (metres unverified)")
    axes[2].imshow(Image.open(path))
    axes[2].scatter(xs[good], ys[good], c="lime", s=6)
    bad = (valid_pred | valid_depth(reference)) & ~good
    axes[2].scatter(xs[bad], ys[bad], c="red", s=8)
    axes[2].set_title("Mesh ray check: green <1% error; red mismatch")
    for ax in axes: ax.axis("off")
    figure.tight_layout(); figure.savefig(output / "camera_alignment.png", dpi=150); plt.close(figure)
    if structural_mesh is not None:
        figure = plt.figure(figsize=(7, 6)); ax = figure.add_subplot(projection="3d")
        vertices = structural_mesh.vertices
        ax.add_collection3d(Poly3DCollection(vertices[structural_mesh.faces], facecolor="lightblue",
                                            edgecolor="gray", linewidth=.3, alpha=.25))
        ax.auto_scale_xyz(*vertices.T); ax.set_box_aspect(np.ptp(vertices, axis=0))
        ax.set_title("Separate ceil.glb: structural geometry (dataset units)")
        figure.savefig(output / "structural_asset.png", dpi=130); plt.close(figure)
    print(json.dumps({k: v for k,v in report.items() if k not in ("per_view_checks", "object_files")}, indent=2))
    if not alignment_passed:
        raise SystemExit("Camera/depth consistency check failed; inspect report.json before using this room.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/3dfront/pilot"))
    parser.add_argument("--output", type=Path, default=Path("outputs/3dfront_validation"))
    parser.add_argument("--grid", type=int, default=16)
    args = parser.parse_args()
    main(args.data, args.output, args.grid)
