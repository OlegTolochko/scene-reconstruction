"""
Render one room scene loaded from a .blend file.

Run (example):
  blender -b /path/to/room.blend -P render_rooms.py -- \
    --room_json /path/to/layout.json \
    --out_folder /path/to/out_dir \
    --num_views 16 \
    --render_resolution 1024 --device CUDA --samples 32
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

# Post-render quality gate constants.
ALMOST_BLACK_MEAN_LUMA_THRESHOLD = 0.01
ALMOST_BLACK_BRIGHT_LUMA_THRESHOLD = 0.08
ALMOST_BLACK_BRIGHT_FRACTION_THRESHOLD = 0.0005
MAX_BLACKFRAME_RESAMPLE_ATTEMPTS = 20


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def init_render(engine="CYCLES", resolution=512, device="CUDA", samples=32, use_denoising=True):
    scene = bpy.context.scene
    render = scene.render
    cycles = scene.cycles

    render.engine = engine
    render.resolution_x = int(resolution)
    render.resolution_y = int(resolution)
    render.resolution_percentage = 100
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    render.film_transparent = True

    render_device = device.upper()
    cycles.device = "CPU" if render_device == "CPU" else "GPU"
    cycles.samples = int(samples)
    if hasattr(cycles, "pixel_filter_type"):
        cycles.pixel_filter_type = "BOX"
    elif hasattr(cycles, "filter_type"):
        cycles.filter_type = "BOX"
    if hasattr(cycles, "filter_width"):
        cycles.filter_width = 1
    elif hasattr(render, "filter_size"):
        render.filter_size = 1.0
    cycles.diffuse_bounces = 1
    cycles.glossy_bounces = 1
    cycles.transparent_max_bounces = 3
    cycles.transmission_bounces = 3
    cycles.use_denoising = bool(use_denoising)

    if render_device != "CPU":
        bpy.context.preferences.addons["cycles"].preferences.get_devices()
        bpy.context.preferences.addons["cycles"].preferences.compute_device_type = render_device


def init_camera() -> Tuple[bpy.types.Object, bpy.types.Object]:
    cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.data.sensor_height = cam.data.sensor_width = 32.0

    track = cam.constraints.new(type="TRACK_TO")
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"

    target = bpy.data.objects.new("CameraTarget", None)
    target.location = (0.0, 0.0, 0.0)
    bpy.context.scene.collection.objects.link(target)
    track.target = target
    return cam, target


def get_transform_matrix(obj: bpy.types.Object) -> list:
    pos, rt, _ = obj.matrix_world.decompose()
    rt = rt.to_matrix()
    matrix = []
    for ii in range(3):
        row = []
        for jj in range(3):
            row.append(rt[ii][jj])
        row.append(pos[ii])
        matrix.append(row)
    matrix.append([0, 0, 0, 1])
    return matrix


def extract_room_dimensions(room_json_path: str) -> Tuple[float, float, float]:
    with open(room_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dims: Optional[Dict[str, float]] = None
    if isinstance(data, dict):
        if isinstance(data.get("rooms"), list) and data["rooms"]:
            room0 = data["rooms"][0]
            if isinstance(room0, dict) and isinstance(room0.get("dimensions"), dict):
                dims = room0["dimensions"]
        if dims is None and isinstance(data.get("dimensions"), dict):
            dims = data["dimensions"]
        if dims is None and all(k in data for k in ("width", "length", "height")):
            dims = data

    if dims is None:
        raise ValueError(f"Could not find room dimensions in JSON: {room_json_path}")

    w = float(dims["width"])
    l = float(dims["length"])
    h = float(dims["height"])
    if w <= 0.0 or l <= 0.0 or h <= 0.0:
        raise ValueError(f"Invalid room dimensions: width={w}, length={l}, height={h}")
    return w, l, h


def _clear_all_lights() -> None:
    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)


def _safe_uniform(lo: float, hi: float, fallback: float) -> float:
    if hi <= lo:
        return float(fallback)
    return float(np.random.uniform(lo, hi))


def _split_budget_dirichlet(num_lights: int, total_budget: float) -> np.ndarray:
    weights = np.random.dirichlet(np.ones(num_lights, dtype=np.float64))
    energies = weights * float(total_budget)

    min_e = float(total_budget) * 0.05
    max_e = float(total_budget) * 0.50
    energies = np.clip(energies, min_e, max_e)
    s = float(np.sum(energies))
    if s > 1e-8:
        energies = energies * (float(total_budget) / s)
    return np.clip(energies, min_e, max_e)


def init_room_lighting(width: float, length: float, height: float, ambient_strength: float = 2.0) -> None:
    _clear_all_lights()

    if bpy.context.scene.world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    else:
        world = bpy.context.scene.world

    world.use_nodes = True
    nt = world.node_tree
    nodes = nt.nodes
    links = nt.links
    for node in list(nodes):
        nodes.remove(node)

    bg_node = nodes.new(type="ShaderNodeBackground")
    bg_node.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg_node.inputs["Strength"].default_value = float(ambient_strength)

    out_node = nodes.new(type="ShaderNodeOutputWorld")
    links.new(bg_node.outputs["Background"], out_node.inputs["Surface"])

    area = max(width * length, 1e-6)
    base_n = int(round(area / 8.0))
    num_lights = int(max(2, min(8, base_n + int(np.random.randint(-1, 2)))))

    margin_xy = 0.3
    min_spacing = float(np.clip(0.15 * min(width, length), 0.3, 1.5))
    z_min = height - 0.25
    z_max = height - 0.05
    z_fallback = max(0.2, height - 0.1)

    total_budget = float(np.random.uniform(15.0, 25.0) * area)
    energies = _split_budget_dirichlet(num_lights, total_budget)

    positions: List[Vector] = []
    for i in range(num_lights):
        placed = False
        for _ in range(100):
            x = _safe_uniform(margin_xy, width - margin_xy, width * 0.5)
            y = _safe_uniform(margin_xy, length - margin_xy, length * 0.5)
            z = _safe_uniform(z_min, z_max, z_fallback)
            p = Vector((x, y, z))
            if all((p - q).length >= min_spacing for q in positions):
                positions.append(p)
                placed = True
                break

        if not placed:
            positions.append(
                Vector(
                    (
                        _safe_uniform(margin_xy, width - margin_xy, width * 0.5),
                        _safe_uniform(margin_xy, length - margin_xy, length * 0.5),
                        _safe_uniform(z_min, z_max, z_fallback),
                    )
                )
            )

        light_data = bpy.data.lights.new(f"RoomArea_{i:02d}", type="AREA")
        light_obj = bpy.data.objects.new(f"RoomArea_{i:02d}", light_data)
        bpy.context.scene.collection.objects.link(light_obj)
        light_obj.location = positions[-1]
        light_obj.rotation_euler = (
            math.radians(float(np.random.uniform(-5.0, 5.0))),
            math.radians(float(np.random.uniform(-5.0, 5.0))),
            float(np.random.uniform(-math.pi, math.pi)),
        )

        light_data.shape = "RECTANGLE"
        light_data.size = float(np.random.uniform(0.25, 0.85))
        light_data.size_y = float(np.random.uniform(0.25, 0.85))
        light_data.energy = float(energies[i])
        light_data.color = (1.0, 1.0, 1.0)


def build_scene_bvh_entries() -> Tuple[object, List[Tuple[BVHTree, object, object]]]:
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    entries: List[Tuple[BVHTree, object, object]] = []

    for obj in scene.objects:
        if obj.type != "MESH" or obj.hide_render:
            continue

        eval_obj = obj.evaluated_get(depsgraph)
        try:
            bvh = BVHTree.FromObject(eval_obj, depsgraph, epsilon=0.0)
        except TypeError:
            bvh = BVHTree.FromObject(eval_obj, depsgraph)

        if bvh is None:
            continue

        mw = eval_obj.matrix_world.copy()
        imw = mw.inverted_safe()
        entries.append((bvh, mw, imw))

    if not entries:
        raise RuntimeError("No mesh BVH could be built from scene.")
    return depsgraph, entries


def nearest_surface_distance(point_world: Vector, bvh_entries: List[Tuple[BVHTree, object, object]]) -> float:
    min_dist = float("inf")
    for bvh, mw, imw in bvh_entries:
        p_local = imw @ point_world
        if hasattr(bvh, "find_nearest"):
            nearest = bvh.find_nearest(p_local)
        elif hasattr(bvh, "nearest"):
            nearest = bvh.nearest(p_local)
        else:
            raise RuntimeError("BVHTree has neither find_nearest nor nearest method.")

        if nearest is None or nearest[0] is None:
            continue

        co_world = mw @ nearest[0]
        dist = (co_world - point_world).length
        if dist < min_dist:
            min_dist = dist
    return min_dist


def _scene_ray_cast(scene, depsgraph, origin: Vector, direction: Vector, distance: float):
    try:
        return scene.ray_cast(depsgraph, origin, direction, distance=distance)
    except TypeError:
        return scene.ray_cast(depsgraph, origin, direction, distance)


def has_line_of_sight(scene, depsgraph, origin: Vector, target: Vector) -> bool:
    vec = target - origin
    dist = vec.length
    if dist <= 1e-6:
        return False

    d = vec.normalized()
    ray_origin = origin + d * 0.03
    max_dist = max(0.01, dist - 0.06)
    hit, *_ = _scene_ray_cast(scene, depsgraph, ray_origin, d, max_dist)
    return not bool(hit)


def _sample_direction_in_cone(forward: Vector, half_angle_rad: float) -> Vector:
    w = forward.normalized()
    tmp = Vector((0.0, 0.0, 1.0)) if abs(w.z) < 0.99 else Vector((1.0, 0.0, 0.0))
    u = w.cross(tmp)
    if u.length < 1e-9:
        tmp = Vector((0.0, 1.0, 0.0))
        u = w.cross(tmp)
    u.normalize()
    v = w.cross(u).normalized()

    cos_min = math.cos(half_angle_rad)
    cos_theta = float(np.random.uniform(cos_min, 1.0))
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    phi = float(np.random.uniform(0.0, 2.0 * math.pi))

    x = sin_theta * math.cos(phi)
    y = sin_theta * math.sin(phi)
    z = cos_theta
    return (u * x + v * y + w * z).normalized()


def is_directly_blocked(
    scene,
    depsgraph,
    origin: Vector,
    forward: Vector,
    num_rays: int = 21,
    cone_deg: float = 25.0,
    max_dist: float = 2.5,
    near_thresh: float = 0.4,
    blocked_ratio: float = 0.6,
) -> bool:
    near_hits = 0
    half_angle = math.radians(cone_deg)
    for _ in range(num_rays):
        d = _sample_direction_in_cone(forward, half_angle)
        hit, loc, *_ = _scene_ray_cast(scene, depsgraph, origin + d * 0.03, d, max_dist)
        if hit and (loc - origin).length < near_thresh:
            near_hits += 1
    return (near_hits / float(num_rays)) >= blocked_ratio


def _sample_look_at_target(width: float, length: float, height: float) -> Vector:
    margin = 0.3
    if float(np.random.rand()) < 0.60:
        cx = width * 0.5
        cy = length * 0.5
        tx = float(np.random.normal(cx, max(0.1, width * 0.15)))
        ty = float(np.random.normal(cy, max(0.1, length * 0.15)))
        tz = float(np.random.normal(min(1.2, max(0.2, height * 0.45)), 0.25))
    else:
        tx = _safe_uniform(margin, width - margin, width * 0.5)
        ty = _safe_uniform(margin, length - margin, length * 0.5)
        tz = _safe_uniform(0.2, max(0.25, height - 0.2), min(1.2, max(0.2, height * 0.5)))

    tx = _clamp(tx, margin, max(margin, width - margin))
    ty = _clamp(ty, margin, max(margin, length - margin))
    tz = _clamp(tz, 0.2, max(0.25, height - 0.2))
    return Vector((tx, ty, tz))


def sample_camera_pose(
    width: float,
    length: float,
    height: float,
    scene,
    depsgraph,
    bvh_entries: List[Tuple[BVHTree, object, object]],
    max_attempts: int = 500,
) -> Tuple[Vector, Vector]:
    margin_xy = 0.3
    x_lo, x_hi = margin_xy, width - margin_xy
    y_lo, y_hi = margin_xy, length - margin_xy

    z_lo = 0.5
    z_hi = 2.2

    for _ in range(max_attempts):
        cam_pos = Vector(
            (
                _safe_uniform(x_lo, x_hi, width * 0.5),
                _safe_uniform(y_lo, y_hi, length * 0.5),
                _safe_uniform(z_lo, z_hi, max(0.2, min(1.2, height * 0.5))),
            )
        )

        cam_radius = float(np.random.uniform(0.15, 0.2))
        d_near = nearest_surface_distance(cam_pos, bvh_entries)
        if not np.isfinite(d_near) or d_near < cam_radius:
            continue

        for _ in range(40):
            target = _sample_look_at_target(width, length, height)
            view_vec = target - cam_pos
            if view_vec.length < 0.8:
                continue
            if not has_line_of_sight(scene, depsgraph, cam_pos, target):
                continue
            if is_directly_blocked(scene, depsgraph, cam_pos, view_vec.normalized()):
                continue
            return cam_pos, target

    raise RuntimeError("Failed to sample a valid camera pose after many attempts.")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def is_render_almost_black(
    mean_luma_threshold: float = ALMOST_BLACK_MEAN_LUMA_THRESHOLD,
    bright_luma_threshold: float = ALMOST_BLACK_BRIGHT_LUMA_THRESHOLD,
    bright_fraction_threshold: float = ALMOST_BLACK_BRIGHT_FRACTION_THRESHOLD,
) -> bool:
    render_result = bpy.data.images.get("Render Result")
    if render_result is None or render_result.size[0] <= 0 or render_result.size[1] <= 0:
        return False

    pixels = np.asarray(render_result.pixels[:], dtype=np.float32)
    if pixels.size < 4:
        return False

    rgba = pixels.reshape(-1, 4)
    rgb = rgba[:, :3]
    luma = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]

    mean_luma = float(np.mean(luma))
    bright_fraction = float(np.mean(luma > bright_luma_threshold))
    return (mean_luma < mean_luma_threshold) and (bright_fraction < bright_fraction_threshold)


def main(args) -> None:
    mesh_count = sum(1 for obj in bpy.context.scene.objects if obj.type == "MESH" and not obj.hide_render)
    if mesh_count == 0:
        raise RuntimeError("Loaded scene has no renderable mesh objects. Provide a valid room .blend.")
    if args.num_views <= 0:
        raise ValueError(f"--num_views must be >= 1, got: {args.num_views}")

    width, length, height = extract_room_dimensions(args.room_json)

    init_render(
        engine=args.engine,
        resolution=args.render_resolution,
        device=args.device,
        samples=args.samples,
        use_denoising=not args.no_denoising,
    )

    cam, cam_target = init_camera()
    depsgraph, bvh_entries = build_scene_bvh_entries()

    ensure_dir(args.out_folder)

    # Sample room-level constants once.
    init_room_lighting(width, length, height, ambient_strength=2.0)
    lens_mm = float(np.random.uniform(20.0, 35.0))
    cam.data.lens = lens_mm
    camera_angle_x = float(2.0 * math.atan((cam.data.sensor_width * 0.5) / cam.data.lens))

    to_export = {
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "scale": 1.0,
        "offset": [0.0, 0.0, 0.0],
        "frames": [],
    }

    for i in range(args.num_views):
        img_name = f"{i:03d}.png"
        out_file = os.path.join(args.out_folder, img_name)
        accepted = False

        for attempt in range(MAX_BLACKFRAME_RESAMPLE_ATTEMPTS):
            cam_pos, target_pos = sample_camera_pose(width, length, height, bpy.context.scene, depsgraph, bvh_entries)
            cam.location = cam_pos
            cam_target.location = target_pos
            bpy.context.scene.render.filepath = out_file

            bpy.ops.render.render(write_still=True)
            bpy.context.view_layer.update()

            if is_render_almost_black():
                print(
                    f"[WARN] {img_name} attempt {attempt + 1}/{MAX_BLACKFRAME_RESAMPLE_ATTEMPTS} rejected: almost black"
                )
                continue

            to_export["frames"].append(
                {
                    "file_path": img_name,
                    "camera_angle_x": camera_angle_x,
                    "transform_matrix": get_transform_matrix(cam),
                }
            )
            accepted = True
            break

        if not accepted:
            raise RuntimeError(
                f"Failed to render non-black frame for {img_name} after "
                f"{MAX_BLACKFRAME_RESAMPLE_ATTEMPTS} attempts."
            )

    with open(os.path.join(args.out_folder, "transforms.json"), "w", encoding="utf-8") as f:
        json.dump(to_export, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render room-level images from room BLEND + room JSON.")
    parser.add_argument("--room_json", type=str, required=True, help="Path to corresponding room JSON")
    parser.add_argument("--out_folder", type=str, required=True, help="Rendered output folder")
    parser.add_argument("--num_views", type=int, required=True, help="Number of views to render")
    parser.add_argument("--render_resolution", type=int, default=1024, help="Resolution of rendered image")
    parser.add_argument("--engine", type=str, default="CYCLES", help="Render engine")
    parser.add_argument("--device", type=str, default="CUDA", choices=["CPU", "CUDA", "OPTIX", "HIP"], help="Device")
    parser.add_argument("--samples", type=int, default=32, help="Cycles samples")
    parser.add_argument("--no_denoising", action="store_true", default=False, help="Disable denoising")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed")

    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parser.parse_args(argv)

    if args.seed is not None:
        np.random.seed(int(args.seed))

    main(args)
