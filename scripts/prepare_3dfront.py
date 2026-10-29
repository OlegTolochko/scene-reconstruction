"""Convert rendered 3D-FRONT rooms to GenRecon's COLMAP input layout.

Supplied camera poses condition RGB inference. A 10k-point cloud backprojected
from the selected views' GT depth defines scene chunks; this is not RGB-only
reconstruction. Coordinates retain the source dataset scale.
"""

import argparse
import json
from pathlib import Path

import OpenEXR
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def read_depth(path):
    with OpenEXR.File(str(path), separate_channels=True) as source:
        channels = source.channels()
        key = "V" if "V" in channels else "R" if "R" in channels else next(iter(channels))
        return np.asarray(channels[key].pixels, dtype=np.float64)


def prepare(root, output, count=8, seed=42):
    if output.exists() and any(output.iterdir()):
        raise SystemExit("Output directory is not empty; reuse the prepared inputs or choose a new --output.")
    manifest = json.loads((root / "manifest.json").read_text())
    render_dir = root / f"3D-FRONT-TEST-RENDER/{manifest['room']}"
    metadata = json.loads((render_dir / "meta.json").read_text())
    cameras = metadata["locations"]
    if not 1 <= count <= len(cameras):
        raise ValueError(f"--views must be between 1 and {len(cameras)}")
    indices = np.linspace(0, len(cameras) - 1, count).round().astype(int)
    selected = [cameras[i] for i in indices]
    colmap = output / "iphone/colmap"
    rgb_dir = output / "iphone/rgb"
    colmap.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    flip = np.diag([1., -1., -1., 1.])
    all_points, all_colors = [], []
    camera_lines, image_lines, records = [], [], []
    for image_id, camera in enumerate(selected, 1):
        index = camera["index"]
        source = render_dir / f"render_{index}.webp"
        depth = read_depth(render_dir / f"depth_{index}.exr")
        image = Image.open(source)
        width, height = image.size
        focal = width / (2 * np.tan(metadata["camera_angle_x"] / 2))
        c2w = np.asarray(camera["transform_matrix"], dtype=float)
        w2c = flip @ np.linalg.inv(c2w)
        quaternion = Rotation.from_matrix(w2c[:3, :3]).as_quat()[[3, 0, 1, 2]]
        name = f"{index}.png"
        image.save(rgb_dir / name)  # Preserve alpha where supplied.
        camera_lines.append(f"{image_id} PINHOLE {width} {height} {focal} {focal} {width/2} {height/2}")
        pose = " ".join(map(str, [*quaternion, *w2c[:3, 3]]))
        image_lines.append(f"{image_id} {pose} {image_id} {name}\n")
        # Sample a regular pixel grid before pooling, limiting temporary memory.
        ys, xs = np.mgrid[0:height:4, 0:width:4]
        z = depth[ys, xs]
        # EXR background is encoded as float16's maximum finite value.
        keep = np.isfinite(z) & (z > 0) & (z < 65504)
        x = (xs + .5 - width/2) / focal * z
        y = -(ys + .5 - height/2) / focal * z
        local = np.stack([x[keep], y[keep], -z[keep]], axis=-1)
        all_points.append(local @ c2w[:3, :3].T + c2w[:3, 3])
        all_colors.append(np.asarray(image.convert("RGB"))[ys[keep], xs[keep]])
        # Verify COLMAP conversion reproduces source camera-to-world matrices.
        if not np.allclose(np.linalg.inv(w2c) @ flip, c2w, atol=1e-6):
            raise RuntimeError("Camera conversion failed")
        records.append({"index": index, "source_rgb": str(source), "image": name})
    points = np.concatenate(all_points)
    if not len(points):
        raise ValueError("Selected views contain no valid depth points")
    colors = np.concatenate(all_colors)
    picks = np.random.default_rng(seed).choice(len(points), min(10_000, len(points)), replace=False)
    points, colors = points[picks], colors[picks]
    (colmap / "cameras.txt").write_text("\n".join(camera_lines) + "\n")
    (colmap / "images.txt").write_text("\n".join(image_lines) + "\n")
    with (colmap / "points3D.txt").open("w") as handle:
        for point_id, (xyz, color) in enumerate(zip(points, colors), 1):
            handle.write(f"{point_id} " + " ".join(map(str, [*xyz, *color, 0.])) + "\n")
    details = {"room": manifest["room"], "dataset_revision": manifest["revision"],
               "views": records, "point_count": len(points), "point_seed": seed,
               "support_points": "Backprojected GT depth from selected input views only",
               "metric_scale_verified": False, "coordinate_frame": "Blender z-up, source dataset units",
               "model_input": "RGB plus supplied cameras; depth-derived points define chunks",
               "reference_shell_used_for_input": False}
    (output / "input_manifest.json").write_text(json.dumps(details, indent=2) + "\n")
    print(f"Prepared {len(selected)} views and {len(points)} visible support points in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/3dfront/pilot"))
    parser.add_argument("--output", type=Path, default=Path("data/3dfront/genrecon_pilot"))
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.data, args.output, args.views, args.seed)
