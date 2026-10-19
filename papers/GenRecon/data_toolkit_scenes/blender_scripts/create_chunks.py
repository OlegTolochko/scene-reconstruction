"""
Batch random crops from a preprocessed room .blend and export each crop to .glb.

Run (example):
  blender -b -P create_chunks.py -- \
    /path/to/rooms_preprocessed/<scene_id>.blend \
    /path/to/crops \
    50 \
    '[2.7,3.0]' \
    '{"width":4.0,"length":5.0,"height":2.7}' \
    --seed 123

Args (after --):
  1) blend_path        (input .blend)
  2) out_dir           (output folder)
  3) num_crops         (int)
  4) crop_size_range   (json array: [low, high] — sampled uniformly per crop)
  5) scene_dims_json   (json string: {"width":..,"length":..,"height":..})

Optional flags:
  --seed <int>         (reproducible sampling; default: random)
  --eps <float>        (sampling bound expansion; default: 1.0)
  --start_idx <int>    (resume from this crop index; default: 0)

What it does per crop:
  - reloads the .blend (clean baseline)
  - samples random crop size c from [low, high]
  - derives zcenter = c / 2
  - samples random Z rotation angle in [0, 360)
  - samples random crop cube center (x,y) in the ORIGINAL (unrotated) room space:
      x ∈ [0.5*c - eps, W - 0.5*c + eps]
      y ∈ [0.5*c - eps, L - 0.5*c + eps]
    then rotates (x,y) into the rotated world frame via the rotation transform
    and z = zcenter
  - boolean INTERSECT each mesh with the cube
  - deletes empty meshes and the cutter cube
  - normalizes cropped content into unit cube using crop geometry:
      scale = 1/crop_size, center = (cx, cy, cz)
  - exports GLB: <out_dir>/<scene_id>_<crop_idx>.glb
"""

import json
import math
import random
import sys
from pathlib import Path

import bpy
import mathutils


# -------------------------
# Normalization
# -------------------------
def normalize_to_crop_cube(cx: float, cy: float, cz: float, crop_size: float):
    """
    Normalize the scene so that the crop cube maps to [-0.5, 0.5]^3.
    scale = 1/crop_size, center = (cx, cy, cz) in rotated world space.
    Returns (scale, center_vector).
    """
    scale = 1.0 / crop_size
    center = mathutils.Vector((cx, cy, cz))

    scene = bpy.context.scene
    root = bpy.data.objects.new("NORMALIZE_ROOT", None)
    scene.collection.objects.link(root)
    root.empty_display_size = 0.25

    objs = [o for o in scene.objects if not o.hide_render and o != root]
    for o in objs:
        if o.parent is None:
            o.parent = root
            o.matrix_parent_inverse = root.matrix_world.inverted()

    root.location = (-center) * scale
    root.scale = (scale, scale, scale)

    print(f"[Crops] Normalized: crop_center=({cx:.4f},{cy:.4f},{cz:.4f}), scale={scale:.6f}")
    return scale, center


# -------------------------
# Crop
# -------------------------
def create_crop_cube(name="__CROP_CUBE__", edge=3.0, location=(0.0, 0.0, 1.5), z_extend=0.05):
    """
    Create the crop cube cutter. The XY extent is exactly `edge`; the Z extent is
    `edge + 2*z_extend` (symmetric), so the top and bottom faces are never coplanar
    with the room ceiling/floor, avoiding silent boolean failures on coplanar faces.
    The crop center and normalization are unaffected — z_extend only applies to the cutter.
    """
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    cube = bpy.context.active_object
    cube.name = name
    cube.scale = (edge, edge, edge + 2.0 * z_extend)
    bpy.ops.object.transform_apply(scale=True)
    cube.hide_render = True
    cube.display_type = "WIRE"
    return cube


def _mesh_world_bbox_size(obj):
    if obj.type != "MESH" or obj.data is None or len(obj.data.vertices) == 0:
        return (0.0, 0.0, 0.0)
    corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    min_v = mathutils.Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
    max_v = mathutils.Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
    size = max_v - min_v
    return (float(size.x), float(size.y), float(size.z))


def _is_suspicious_boolean_result(pre_size, post_size, cube_edge):
    if cube_edge <= 0:
        return False
    pre_max = max(pre_size)
    post_max = max(post_size)
    if pre_max <= 1e-8:
        return False

    near_full_cube = all(d >= (0.90 * cube_edge) for d in post_size)
    blown_up = (post_max / pre_max) >= 4.0
    return near_full_cube and blown_up


def apply_boolean_intersect_api(obj, cutter, solver="EXACT", cube_edge=3.0, retry_on_suspicious=True):
    """
    Cropping via boolean intersection. Check for suspicious objects.
    """
    if obj.type != "MESH" or obj == cutter or obj.data is None:
        return False
    if len(obj.data.vertices) == 0:
        return False

    # Fix multi-user meshes
    if obj.data.users > 1:
        obj.data = obj.data.copy()

    # Solidify before boolean: flat/thin meshes (e.g. floors, ceilings) are non-manifold
    # and cause silent boolean failures. A tiny thickness makes them manifold without
    # any visible effect.
    solidify = obj.modifiers.new(name="__SOLIDIFY__", type="SOLIDIFY")
    solidify.thickness = 0.001
    solidify.offset = 0.0  # extend equally in both directions to stay centered

    mod = obj.modifiers.new(name="__CROP_INTERSECT__", type="BOOLEAN")
    mod.operation = "INTERSECT"
    mod.object = cutter
    mod.solver = solver
    if hasattr(mod, "use_hole_tolerant"):
        mod.use_hole_tolerant = True

    pre_size = _mesh_world_bbox_size(obj)

    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)

    new_mesh = bpy.data.meshes.new_from_object(obj_eval, preserve_all_data_layers=True, depsgraph=depsgraph)

    old_mesh = obj.data
    obj.data = new_mesh
    obj.modifiers.remove(mod)
    obj.modifiers.remove(solidify)
    post_size = _mesh_world_bbox_size(obj)

    if _is_suspicious_boolean_result(pre_size, post_size, cube_edge):
        # Retry once with FAST solver before dropping the object.
        obj.data = old_mesh
        if new_mesh.users == 0:
            bpy.data.meshes.remove(new_mesh)
        # Blender 5.0 renamed "FAST" to "MANIFOLD"; fall back to whichever alternative exists.
        fallback_solver = (
            "MANIFOLD"
            if "MANIFOLD" in bpy.types.BooleanModifier.bl_rna.properties["solver"].enum_items.keys()
            else "FAST"
        )
        if retry_on_suspicious and solver not in (fallback_solver, "FAST", "MANIFOLD"):
            print(f"[Crops] Suspicious boolean result on {obj.name} with solver={solver}; retrying {fallback_solver}.")
            return apply_boolean_intersect_api(
                obj,
                cutter,
                solver=fallback_solver,
                cube_edge=cube_edge,
                retry_on_suspicious=False,
            )
        print(f"[Crops] Dropping suspicious boolean artifact object: {obj.name}")
        bpy.data.objects.remove(obj, do_unlink=True)
        if old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)
        return False

    if old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)

    return True


def delete_empty_mesh_objects():
    scene = bpy.context.scene
    empties = []
    for obj in scene.objects:
        if obj.type == "MESH" and obj.data and len(obj.data.vertices) == 0:
            empties.append(obj)
    for obj in empties:
        bpy.data.objects.remove(obj, do_unlink=True)
    return len(empties)


def crop_scene_to_cube(edge=3.0, location=(0.0, 0.0, 1.5), solver="EXACT", delete_cutter=True):
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    cutter = create_crop_cube(edge=edge, location=location)

    scene = bpy.context.scene
    cropped = 0
    for obj in list(scene.objects):
        if obj.type != "MESH" or obj == cutter:
            continue
        if apply_boolean_intersect_api(obj, cutter, solver=solver, cube_edge=edge):
            cropped += 1

    removed_empty = delete_empty_mesh_objects()

    if delete_cutter:
        bpy.data.objects.remove(cutter, do_unlink=True)

    print(f"[Crops] Crop @ {location} edge={edge}: cropped {cropped}, removed_empty {removed_empty}")


# -------------------------
# Rotation helpers
# -------------------------
def _z_rotation_transform_and_dims(angle_deg: float, width: float, length: float):
    """
    Arbitrary Z rotation. Translation keeps all room corners in positive space.
    Returns (transform_matrix, effective_width, effective_length).
    """
    theta = math.radians(angle_deg)
    rot = mathutils.Matrix.Rotation(theta, 4, "Z")
    corners = [(0.0, 0.0), (width, 0.0), (width, length), (0.0, length)]
    rotated = [rot @ mathutils.Vector((x, y, 0.0, 1.0)) for x, y in corners]
    min_x = min(v.x for v in rotated)
    min_y = min(v.y for v in rotated)
    max_x = max(v.x for v in rotated)
    max_y = max(v.y for v in rotated)
    trans = mathutils.Matrix.Translation((-min_x, -min_y, 0.0))
    return trans @ rot, (max_x - min_x), (max_y - min_y)


def _apply_transform_to_scene_roots(transform: mathutils.Matrix):
    """
    Apply transform to root objects only, preserving parent-child hierarchies.
    """
    for obj in bpy.context.scene.objects:
        if obj.parent is None:
            obj.matrix_world = transform @ obj.matrix_world


def _sampling_bounds(width: float, length: float, crop_size: float, eps: float):
    half_c = 0.5 * crop_size
    xmin, xmax = (half_c - eps), (width - half_c + eps)
    ymin, ymax = (half_c - eps), (length - half_c + eps)
    if xmin > xmax or ymin > ymax:
        return None
    return xmin, xmax, ymin, ymax


def _mat4_to_list(m: mathutils.Matrix):
    return [[float(m[r][c]) for c in range(4)] for r in range(4)]


def _normalization_matrix(center: mathutils.Vector, scale: float):
    """
    Build rotated-world -> normalized-chunk transform explicitly:
      p_chunk = scale * (p_rot - center)
    """
    sx = float(scale)
    cx, cy, cz = float(center.x), float(center.y), float(center.z)
    return mathutils.Matrix(
        (
            (sx, 0.0, 0.0, -sx * cx),
            (0.0, sx, 0.0, -sx * cy),
            (0.0, 0.0, sx, -sx * cz),
            (0.0, 0.0, 0.0, 1.0),
        )
    )


# -------------------------
# Export
# -------------------------
def export_glb(filepath: Path):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Ensure object mode
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    # Export GLB (binary glTF). Blender 5 uses export_vertex_color, older builds used export_colors.
    kwargs = {
        "filepath": str(filepath),
        "export_format": "GLB",
        "export_apply": True,
        "export_yup": True,
        "export_texcoords": True,
        "export_normals": True,
        "export_materials": "EXPORT",
        "use_selection": False,
    }
    gltf_props = bpy.ops.export_scene.gltf.get_rna_type().properties.keys()
    if "export_vertex_color" in gltf_props:
        kwargs["export_vertex_color"] = "MATERIAL"
    elif "export_colors" in gltf_props:
        kwargs["export_colors"] = True

    bpy.ops.export_scene.gltf(**kwargs)
    print(f"[Crops] Exported: {filepath}")


# -------------------------
# Args / main
# -------------------------
def _args_after_double_dash(argv):
    if "--" not in argv:
        return []
    return argv[argv.index("--") + 1 :]


def _pop_flag(args, flag, has_value=True):
    if flag not in args:
        return None
    i = args.index(flag)
    if has_value:
        if i + 1 >= len(args):
            raise SystemExit(f"Missing value for {flag}")
        val = args[i + 1]
        del args[i : i + 2]
        return val
    else:
        del args[i : i + 1]
        return True


def main():
    args = _args_after_double_dash(sys.argv)
    if len(args) < 5:
        raise SystemExit(
            "Usage:\n"
            "  blender -b -P create_chunks.py -- <blend_path> <out_dir> <num_crops> <crop_size_range_json> <scene_dims_json> [--seed N] [--eps E] [--start_idx N]\n"
            "Example:\n"
            '  blender -b -P create_chunks.py -- room.blend ./crops 10 \'[2.7,3.0]\' \'{"width":4,"length":5,"height":2.7}\' --seed 123 --eps 1.0'
        )

    seed_str = _pop_flag(args, "--seed", has_value=True)
    eps_str = _pop_flag(args, "--eps", has_value=True)
    start_idx_str = _pop_flag(args, "--start_idx", has_value=True)

    blend_path = Path(args[0]).expanduser().resolve()
    out_dir = Path(args[1]).expanduser().resolve()
    num_crops = int(args[2])
    crop_size_range = json.loads(args[3])
    dims = json.loads(args[4])

    crop_size_min = float(crop_size_range[0])
    crop_size_max = float(crop_size_range[1])

    W = float(dims["width"])
    L = float(dims["length"])
    # height provided but not used in XY sampling; kept for completeness
    _H = float(dims.get("height", 0.0))

    eps = float(eps_str) if eps_str is not None else 1.0
    start_idx = int(start_idx_str) if start_idx_str is not None else 0

    if not blend_path.is_file():
        raise SystemExit(f"Input .blend not found: {blend_path}")

    scene_id = blend_path.stem  # .../<scene_id>.blend
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sanity check: sampling is always done in original room space, so only W, L matter.
    if _sampling_bounds(W, L, crop_size_max, eps) is None:
        print(
            f"[Crops] WARNING: crop_size_max={crop_size_max} is too large for room dims W={W}, L={L}. "
            f"Some or all crops will be skipped."
        )

    if seed_str is None:
        seed_val = random.randrange(1 << 30)
    else:
        seed_val = int(seed_str)

    # For deterministic resumption: shift the seed by start_idx so resumed crops
    # don't repeat the same random sequence as the already-produced ones.
    random.seed(seed_val + start_idx)

    print("[Crops] blend_path:", blend_path)
    print("[Crops] out_dir:", out_dir)
    print("[Crops] scene_id:", scene_id)
    print("[Crops] num_crops:", num_crops)
    print("[Crops] crop_size_range:", [crop_size_min, crop_size_max])
    print("[Crops] eps:", eps)
    print("[Crops] dims:", dims)
    print("[Crops] seed:", seed_val, "(effective seed:", seed_val + start_idx, ")")
    print("[Crops] start_idx:", start_idx)

    # Load existing metadata if resuming
    scene_meta_path = out_dir / f"{scene_id}.json"
    if scene_meta_path.is_file() and start_idx > 0:
        with scene_meta_path.open("r", encoding="utf-8") as f:
            existing_meta = json.load(f)
        chunks_meta = existing_meta.get("chunks", [])
        print(f"[Crops] Resuming: loaded {len(chunks_meta)} existing chunks from {scene_meta_path}")
    else:
        chunks_meta = []

    for i in range(start_idx, num_crops):
        # Reload baseline for each crop (robust, still faster than launching Blender per crop)
        bpy.ops.wm.open_mainfile(filepath=str(blend_path))

        # Sample crop_size and derive zcenter
        crop_size = random.uniform(crop_size_min, crop_size_max)
        zcenter = crop_size / 2.0

        # Sample angle freely — bounds are computed in original room space, so they don't
        # depend on the rotation angle.
        angle = random.uniform(0.0, 360.0)
        transform, _, _ = _z_rotation_transform_and_dims(angle, W, L)

        # Sample crop center in original (unrotated) room space.
        bounds = _sampling_bounds(W, L, crop_size, eps)
        if bounds is None:
            print(f"[Crops] crop_idx={i}: room too small for crop_size={crop_size:.4f}; skipping.")
            continue

        xmin, xmax, ymin, ymax = bounds
        cx_orig = random.uniform(xmin, xmax)
        cy_orig = random.uniform(ymin, ymax)

        # Rotate the sampled center into the rotated world frame.
        p_orig = mathutils.Vector((cx_orig, cy_orig, 0.0, 1.0))
        p_rot = transform @ p_orig
        cx, cy = p_rot.x, p_rot.y
        cz = zcenter

        _apply_transform_to_scene_roots(transform)

        print(
            "[Crops] crop_idx=%d angle=%.2f crop_size=%.4f zcenter=%.4f "
            "orig_center=(%.4f, %.4f) rot_center=(%.4f, %.4f, %.4f)"
            % (i, angle, crop_size, zcenter, cx_orig, cy_orig, cx, cy, cz)
        )

        crop_scene_to_cube(edge=crop_size, location=(cx, cy, cz), solver="EXACT", delete_cutter=True)

        normalization_scale, norm_center = normalize_to_crop_cube(cx, cy, cz, crop_size)

        m_norm = _normalization_matrix(norm_center, normalization_scale)
        m_scene = transform.copy()  # original -> rotated
        m_original_to_chunk = m_norm @ m_scene
        m_chunk_to_original = m_original_to_chunk.inverted()

        out_path = out_dir / f"{scene_id}_{i:04d}.glb"
        export_glb(out_path)

        chunks_meta.append(
            {
                "index": int(i),
                "rotation_deg_z": float(angle),
                "crop_center": [float(cx), float(cy), float(cz)],
                "crop_size": float(crop_size),
                "normalization_scale": float(normalization_scale),
                "M_chunk_to_original": _mat4_to_list(m_chunk_to_original),
                "M_original_to_chunk": _mat4_to_list(m_original_to_chunk),
            }
        )

        # Optional: purge orphans to reduce memory growth across iterations
        try:
            bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
        except Exception:
            pass

    scene_meta = {
        "scene_id": scene_id,
        "chunks": chunks_meta,
    }
    with scene_meta_path.open("w", encoding="utf-8") as f:
        json.dump(scene_meta, f, indent=2)
    print("[Crops] Metadata:", scene_meta_path)

    print("[Crops] Done.")


if __name__ == "__main__":
    main()
