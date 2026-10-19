"""
Room Builder

Run from command line (example):
  blender -b -P build_rooms.py -- /path/to/raw_rooms/<room_id> /path/to/processed_rooms/<room_id>.blend

Input folder must contain:
  - layout_<id>.json   (exact name can vary; script finds the first layout_*.json)
  - materials/         (textures + door pkl UVs)
  - objects/           (meshes .ply + textures)

Output:
  - <output>.blend

Notes:
  - custom PLY parser for vertex/texcoord/face indices
  - door planes with UVs from PKL
  - simple Principled BSDF material with ImageTexture(Base Color)
  - deletes hide_render objects
  - POT+square enforcement via RESCALE (stretch) into new image datablocks
  - tries to set viewport shading to Material Preview (no-op in background)
"""

import json
import math
import os
import pickle
import struct
import sys
from pathlib import Path

import bpy

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
SET_MATERIAL_PREVIEW = True

# Door UVs: if door appears upside-down, set to True
DOOR_FLIP_V = False
FRAME_FLIP_V = False

# Door placement:
DOOR_INWARD_SIGN = -1.0  # set to +1.0 to flip side
DOOR_EPSILON = 0.005
EXTRA_OFFSET = 0.001

# PBR defaults
DEFAULT_WALL_ROUGHNESS = 0.95
DEFAULT_FLOOR_ROUGHNESS = 0.90
DEFAULT_CEIL_ROUGHNESS = 0.95


# -------------------------------------------------------------------
# Logging helper
# -------------------------------------------------------------------
def log(*args):
    print("[RoomBuilder]", *args)


# -------------------------------------------------------------------
# Material asset resolution (strict JSON-driven names)
# -------------------------------------------------------------------
def surface_texture_path(materials_dir: Path, material_name: str):
    if not material_name:
        return None
    return materials_dir / f"{material_name}.png"


def door_asset_paths(materials_dir: Path, door_material: str):
    return (
        materials_dir / f"{door_material}_texture.png",
        materials_dir / f"{door_material}_tex_coords.pkl",
        materials_dir / f"{door_material}_frame_texture.png",
        materials_dir / f"{door_material}_frame_tex_coords.pkl",
    )


# -------------------------------------------------------------------
# Basic scene helpers
# -------------------------------------------------------------------
def clear_all_objects_and_orphans():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Optional cleanup of orphan datablocks
    for mat in list(bpy.data.materials):
        if mat.users == 0:
            bpy.data.materials.remove(mat)
    for img in list(bpy.data.images):
        if img.users == 0:
            bpy.data.images.remove(img)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def set_material_preview_if_possible():
    if not bpy.context.screen:
        # Background / no UI
        return
    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    space.shading.type = "MATERIAL"


def place_object(obj, pos_xyz, rot_xyz_deg):
    rx = math.radians(rot_xyz_deg.get("x", 0.0))
    ry = math.radians(rot_xyz_deg.get("y", 0.0))
    rz = math.radians(rot_xyz_deg.get("z", 0.0))
    obj.location = (pos_xyz["x"], pos_xyz["y"], pos_xyz["z"])
    obj.rotation_euler = (rx, ry, rz)


# -------------------------------------------------------------------
# PLY parser (same approach as your original)
# -------------------------------------------------------------------
def parse_ply(path: Path):
    """
    Minimal PLY parser for files that contain:
      - element vertex: properties x y z (float)
      - element texcoord: properties u v (float)
      - element face: two list properties:
          list <count_type> <index_type> vertex_indices
          list <count_type> <index_type> texcoord_indices
    Supports:
      - format ascii 1.0
      - format binary_little_endian 1.0
    """
    path = Path(path)
    with path.open("rb") as f:
        # ---- Read header ----
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError("Unexpected EOF while reading PLY header")
            line_str = line.decode("ascii", errors="strict").rstrip("\r\n")
            header_lines.append(line_str)
            if line_str == "end_header":
                break

        fmt = None
        elements = []
        current = None

        for ln in header_lines:
            parts = ln.split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                current = {"name": parts[1], "count": int(parts[2]), "props": []}
                elements.append(current)
            elif parts[0] == "property" and current is not None:
                if parts[1] == "list":
                    current["props"].append(
                        {"kind": "list", "count_type": parts[2], "item_type": parts[3], "name": parts[4]}
                    )
                else:
                    current["props"].append({"kind": "scalar", "type": parts[1], "name": parts[2]})

        if fmt not in ("ascii", "binary_little_endian"):
            raise RuntimeError(f"Unsupported PLY format: {fmt}. Need ascii or binary_little_endian.")

        def scalar_unpack(tname):
            m = {
                "char": ("b", 1),
                "uchar": ("B", 1),
                "int8": ("b", 1),
                "uint8": ("B", 1),
                "short": ("h", 2),
                "ushort": ("H", 2),
                "int16": ("h", 2),
                "uint16": ("H", 2),
                "int": ("i", 4),
                "uint": ("I", 4),
                "int32": ("i", 4),
                "uint32": ("I", 4),
                "float": ("f", 4),
                "float32": ("f", 4),
                "double": ("d", 8),
                "float64": ("d", 8),
            }
            if tname not in m:
                raise RuntimeError(f"Unsupported scalar type in PLY: {tname}")
            return m[tname]

        vertices = []
        texcoords = []
        faces_v = []
        faces_uv = []

        if fmt == "ascii":
            with path.open("r", encoding="ascii", errors="strict") as tf:
                # skip header lines
                for _ in header_lines:
                    tf.readline()

                for el in elements:
                    name, count, props = el["name"], el["count"], el["props"]
                    for _ in range(count):
                        line = tf.readline()
                        if not line:
                            raise RuntimeError("Unexpected EOF while reading ASCII PLY body")
                        parts = line.strip().split()
                        idx = 0

                        if name == "vertex":
                            scalars = []
                            for p in props:
                                if p["kind"] != "scalar":
                                    raise RuntimeError("Unexpected list property in vertex element (ASCII)")
                                scalars.append(float(parts[idx]))
                                idx += 1
                            vertices.append((scalars[0], scalars[1], scalars[2]))

                        elif name == "texcoord":
                            scalars = []
                            for p in props:
                                if p["kind"] != "scalar":
                                    raise RuntimeError("Unexpected list property in texcoord element (ASCII)")
                                scalars.append(float(parts[idx]))
                                idx += 1
                            texcoords.append((scalars[0], scalars[1]))

                        elif name == "face":
                            lists = []
                            for p in props:
                                if p["kind"] != "list":
                                    # ignore scalars if any
                                    _ = parts[idx]
                                    idx += 1
                                    continue
                                n = int(parts[idx])
                                idx += 1
                                arr = [int(parts[idx + i]) for i in range(n)]
                                idx += n
                                lists.append(arr)

                            if len(lists) < 2:
                                raise RuntimeError(
                                    "Face element must contain 2 list properties (vertex_indices and texcoord_indices)."
                                )

                            faces_v.append(lists[0])
                            faces_uv.append(lists[1])
        else:
            # binary_little_endian: continue reading from current file position
            for el in elements:
                name, count, props = el["name"], el["count"], el["props"]
                for _ in range(count):
                    if name in ("vertex", "texcoord"):
                        scalars = []
                        for p in props:
                            if p["kind"] != "scalar":
                                raise RuntimeError(f"Unexpected list property in {name} element (binary).")
                            fmt_char, sz = scalar_unpack(p["type"])
                            buf = f.read(sz)
                            if len(buf) != sz:
                                raise RuntimeError("Unexpected EOF while reading binary PLY scalars")
                            val = struct.unpack("<" + fmt_char, buf)[0]
                            scalars.append(val)

                        if name == "vertex":
                            vertices.append((float(scalars[0]), float(scalars[1]), float(scalars[2])))
                        else:
                            texcoords.append((float(scalars[0]), float(scalars[1])))

                    elif name == "face":
                        lists = []
                        for p in props:
                            if p["kind"] == "list":
                                c_fmt, c_sz = scalar_unpack(p["count_type"])
                                i_fmt, i_sz = scalar_unpack(p["item_type"])

                                c_buf = f.read(c_sz)
                                if len(c_buf) != c_sz:
                                    raise RuntimeError("Unexpected EOF reading face list count")
                                n = struct.unpack("<" + c_fmt, c_buf)[0]

                                arr = []
                                for _i in range(int(n)):
                                    b = f.read(i_sz)
                                    if len(b) != i_sz:
                                        raise RuntimeError("Unexpected EOF reading face list item")
                                    arr.append(struct.unpack("<" + i_fmt, b)[0])
                                lists.append(arr)
                            else:
                                fmt_char, sz = scalar_unpack(p["type"])
                                _ = f.read(sz)

                        if len(lists) < 2:
                            raise RuntimeError(
                                "Face element must contain 2 list properties (vertex_indices and texcoord_indices)."
                            )

                        faces_v.append([int(x) for x in lists[0]])
                        faces_uv.append([int(x) for x in lists[1]])
                    else:
                        raise RuntimeError(f"Unsupported element in binary PLY: {name}")

    if not vertices or not faces_v:
        raise RuntimeError("Parsed PLY but got no vertices or faces. Check file contents.")
    if len(faces_v) != len(faces_uv):
        raise RuntimeError("Face count mismatch between vertex indices and texcoord indices lists.")
    if not texcoords:
        raise RuntimeError("No texcoords found in PLY. Cannot build UV map.")

    return vertices, faces_v, texcoords, faces_uv


def build_mesh_with_uv(vertices, faces_v, texcoords, faces_uv, name="PLY_UV_Mesh"):
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    mesh.from_pydata(vertices, [], faces_v)
    mesh.update()

    uv_layer = mesh.uv_layers.new(name="UVMap")

    for poly in mesh.polygons:
        uv_idx_list = faces_uv[poly.index]
        if len(uv_idx_list) != poly.loop_total:
            raise RuntimeError(
                f"Face {poly.index} loop count {poly.loop_total} != texcoord index count {len(uv_idx_list)}"
            )
        for corner_i, loop_index in enumerate(poly.loop_indices):
            uv_index = uv_idx_list[corner_i]
            u, v = texcoords[uv_index]
            uv_layer.data[loop_index].uv = (u, v)

    return obj


# -------------------------------------------------------------------
# Materials
# -------------------------------------------------------------------
def load_image(path: Path, colorspace="sRGB"):
    path = Path(path)
    if not path.is_file():
        return None

    abspath = str(path.resolve())

    # Try reuse
    for img in bpy.data.images:
        try:
            if os.path.abspath(bpy.path.abspath(img.filepath)) == abspath:
                try:
                    img.colorspace_settings.name = colorspace
                except Exception:
                    pass
                return img
        except Exception:
            pass

    img = bpy.data.images.load(abspath, check_existing=True)
    try:
        img.colorspace_settings.name = colorspace
    except Exception:
        pass
    return img


def assign_texture_pbr_material(obj, tex_path: Path, metallic=0.0, roughness=0.5, mat_name="TexturePBR"):
    tex_path = Path(tex_path)

    mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (500, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (250, 0)

    tex = nodes.new("ShaderNodeTexImage")
    tex.location = (0, 150)

    tc = nodes.new("ShaderNodeTexCoord")
    tc.location = (-450, 150)
    links.new(tc.outputs["UV"], tex.inputs["Vector"])

    img = load_image(tex_path, "sRGB")
    if img:
        tex.image = img
        links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        log("WARNING: missing texture", str(tex_path))

    bsdf.inputs["Metallic"].default_value = float(metallic)
    bsdf.inputs["Roughness"].default_value = float(roughness)

    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


# -------------------------------------------------------------------
# Room shell
# -------------------------------------------------------------------
def make_wall(start, end, height=2.7, thickness=0.1, name="Wall"):
    sx, sy, _sz = start
    ex, ey, _ez = end
    dx, dy = ex - sx, ey - sy
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1e-8:
        return None

    cx, cy = (sx + ex) * 0.5, (sy + ey) * 0.5
    cz = height * 0.5
    angle = math.atan2(dy, dx)

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(cx, cy, cz))
    obj = bpy.context.object
    obj.name = name
    obj.rotation_euler = (0.0, 0.0, angle)
    obj.scale = (length, thickness, height)
    bpy.ops.object.transform_apply(scale=True)
    return obj


def make_floor_ceiling(width, length, height, name_prefix="Room"):
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(width / 2, length / 2, 0.0))
    floor = bpy.context.object
    floor.name = f"{name_prefix}_Floor"
    floor.scale = (width, length, 1.0)
    bpy.ops.object.transform_apply(scale=True)

    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(width / 2, length / 2, height))
    ceil = bpy.context.object
    ceil.name = f"{name_prefix}_Ceiling"
    ceil.scale = (width, length, 1.0)
    ceil.rotation_euler = (math.pi, 0.0, 0.0)
    bpy.ops.object.transform_apply(scale=True)

    return floor, ceil


def unwrap_basic(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    try:
        bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.02)
    except Exception as e:
        log("unwrap failed:", obj.name, e)
    bpy.ops.object.mode_set(mode="OBJECT")


# -------------------------------------------------------------------
# Door creation from PKL UVs + textures
# -------------------------------------------------------------------
def load_uv_pkl(pkl_path: Path):
    with Path(pkl_path).open("rb") as f:
        return pickle.load(f)


def _as_uv_list(pkl_data):
    def to_uv_rows(x):
        if hasattr(x, "shape") and len(x.shape) == 2 and x.shape[1] == 2:
            return [(float(x[i, 0]), float(x[i, 1])) for i in range(x.shape[0])]
        if isinstance(x, (list, tuple)) and len(x) > 0:
            if isinstance(x[0], (list, tuple)) and len(x[0]) == 2:
                return [(float(u), float(v)) for (u, v) in x]
        return None

    def order_quad_from_points(points):
        us = [p[0] for p in points]
        vs = [p[1] for p in points]
        umin, umax = min(us), max(us)
        vmin, vmax = min(vs), max(vs)
        # Match quad vertex order in build_door_plane: BL, BR, TR, TL.
        return [(umin, vmin), (umax, vmin), (umax, vmax), (umin, vmax)]

    def unique_points(points, eps=1e-6):
        uniq = []
        for u, v in points:
            keep = True
            for uu, vv in uniq:
                if abs(u - uu) <= eps and abs(v - vv) <= eps:
                    keep = False
                    break
            if keep:
                uniq.append((u, v))
        return uniq

    if isinstance(pkl_data, dict):
        for k in ("uv", "uvs", "texcoord", "texcoords", "coords"):
            if k in pkl_data:
                pkl_data = pkl_data[k]
                break
        else:
            # Dataset door/frame UV PKLs use {'vts': Nx2, 'fts': Mx3}.
            vts = to_uv_rows(pkl_data.get("vts")) if "vts" in pkl_data else None
            fts = pkl_data.get("fts")
            if vts:
                # Prefer indexed UVs via fts when present.
                if fts is not None:
                    tri_idx = []
                    if hasattr(fts, "shape") and len(fts.shape) == 2:
                        tri_idx = [[int(fts[i, j]) for j in range(fts.shape[1])] for i in range(fts.shape[0])]
                    elif isinstance(fts, (list, tuple)):
                        for tri in fts:
                            if isinstance(tri, (list, tuple)):
                                tri_idx.append([int(x) for x in tri])

                    if tri_idx:
                        flat = []
                        for tri in tri_idx:
                            for idx in tri:
                                if 0 <= idx < len(vts):
                                    flat.append(vts[idx])
                        if len(flat) in (4, 6):
                            return flat
                        if len(flat) > 6:
                            uq = unique_points(flat)
                            if len(uq) == 4:
                                return order_quad_from_points(uq)
                            return order_quad_from_points(flat)

                if len(vts) in (4, 6):
                    return vts
                if len(vts) > 6:
                    uq = unique_points(vts)
                    if len(uq) == 4:
                        return order_quad_from_points(uq)
                    return order_quad_from_points(vts)

    uv_rows = to_uv_rows(pkl_data)
    if uv_rows is not None:
        return uv_rows

    raise RuntimeError(f"Unsupported UV PKL format: type={type(pkl_data)} sample={str(pkl_data)[:200]}")


def build_door_plane(name, width, height):
    """
    Quad plane centered at origin in local XZ, normal +Y
    """
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    hw = width * 0.5
    hh = height * 0.5

    verts = [(-hw, 0.0, -hh), (hw, 0.0, -hh), (hw, 0.0, hh), (-hw, 0.0, hh)]
    faces = [(0, 1, 2, 3)]
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    mesh.uv_layers.new(name="UVMap")
    return obj


def apply_uvs_to_quad(obj, uv_list, flip_v=False):
    me = obj.data
    if not me.uv_layers:
        me.uv_layers.new(name="UVMap")
    uv_layer = me.uv_layers.active

    def conv(u, v):
        return (u, 1.0 - v) if flip_v else (u, v)

    if len(me.polygons) != 1 or me.polygons[0].loop_total != 4:
        raise RuntimeError("apply_uvs_to_quad expects a single quad face mesh.")

    if len(uv_list) == 4:
        poly = me.polygons[0]
        for corner_i, li in enumerate(poly.loop_indices):
            u, v = uv_list[corner_i]
            uv_layer.data[li].uv = conv(u, v)
        return

    if len(uv_list) == 6:
        # triangulate quad (operator-based like your original)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
        bpy.ops.object.mode_set(mode="OBJECT")

        if len(me.polygons) != 2 or any(p.loop_total != 3 for p in me.polygons):
            raise RuntimeError("Triangulation did not produce 2 triangles as expected.")

        k = 0
        for poly in me.polygons:
            for li in poly.loop_indices:
                u, v = uv_list[k]
                uv_layer.data[li].uv = conv(u, v)
                k += 1
        return

    raise RuntimeError(f"UV list must have length 4 or 6 for a quad door, got {len(uv_list)}")


def place_door_on_wall(
    door_obj,
    wall_start,
    wall_end,
    position_on_wall,
    door_height,
    wall_thickness=0.1,
    z_bottom=0.0,
    inward_sign=+1.0,
    epsilon=0.02,
    extra_offset=0.0,
):
    sx, sy = wall_start["x"], wall_start["y"]
    ex, ey = wall_end["x"], wall_end["y"]
    dx, dy = ex - sx, ey - sy

    px = sx + dx * position_on_wall
    py = sy + dy * position_on_wall

    angle = math.atan2(dy, dx)

    # orient door so its width runs along the wall direction
    door_obj.rotation_euler = (0.0, 0.0, angle)

    # center in height
    door_obj.location = (px, py, z_bottom + door_height * 0.5)

    # wall normal
    nx, ny = -math.sin(angle), math.cos(angle)

    # push out of the wall volume: half thickness + epsilon
    push = (wall_thickness * 0.5) + epsilon + extra_offset

    door_obj.location.x += inward_sign * nx * push
    door_obj.location.y += inward_sign * ny * push


# -------------------------------------------------------------------
# Object mesh cache by source_id (cleaner: cache meshes, not hidden objects)
# -------------------------------------------------------------------
class MeshCache:
    def __init__(self, objects_dir: Path):
        self.objects_dir = Path(objects_dir)
        self._mesh_by_sid = {}  # sid -> bpy.types.Mesh

    def get_or_build(self, sid: str):
        if sid not in self._mesh_by_sid:
            ply_path = self.objects_dir / f"{sid}.ply"
            if not ply_path.is_file():
                return None, f"Missing mesh: {ply_path}"

            verts, faces_v, uvs, faces_uv = parse_ply(ply_path)
            tmp_obj = build_mesh_with_uv(verts, faces_v, uvs, faces_uv, name=f"{sid}_meshbuild_tmp")
            mesh = tmp_obj.data
            # Remove the temporary object; keep mesh datablock
            bpy.data.objects.remove(tmp_obj, do_unlink=True)

            mesh.name = f"{sid}_MESH"
            self._mesh_by_sid[sid] = mesh

        mesh = self._mesh_by_sid[sid]

        obj = bpy.data.objects.new(f"{sid}_inst", mesh)
        bpy.context.collection.objects.link(obj)
        return obj, None


# -------------------------------------------------------------------
# Ensure correct texture image sizes (same behavior: RESCALE to square POT)
# -------------------------------------------------------------------
def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def needs_fix(w: int, h: int) -> bool:
    return (w != h) or (not is_power_of_two(w))


def rescale_image_to_square_pot(src_img: bpy.types.Image, new_img: bpy.types.Image):
    # Ensure source pixels are loaded before copy/scale.
    if not src_img.has_data:
        src_img.pixels[:]  # triggers load in some cases

    # Make a temporary copy so we don't alter the original datablock
    tmp = src_img.copy()
    tmp.name = src_img.name + "_TMP_RESCALE"

    if not tmp.has_data:
        tmp.pixels[:]  # triggers load

    target_w, target_h = new_img.size

    # Scale temp to target size (stretches if needed)
    tmp.scale(target_w, target_h)

    # Copy scaled pixels to destination
    new_img.pixels = list(tmp.pixels)
    new_img.update()

    # Clean up temp
    bpy.data.images.remove(tmp)


def fix_image_to_square_pot(img: bpy.types.Image):
    w, h = img.size
    if not needs_fix(w, h):
        return None

    new_size = next_power_of_two(max(w, h))

    base_name = img.name.rsplit(".", 1)[0]
    new_name = f"{base_name}_POT_{new_size}_RESCALED"

    new_img = bpy.data.images.new(
        name=new_name,
        width=new_size,
        height=new_size,
        alpha=True,
        float_buffer=False,
    )

    try:
        new_img.colorspace_settings.name = img.colorspace_settings.name
    except Exception:
        pass

    rescale_image_to_square_pot(img, new_img)

    # Blender 5 can reload generated images as black unless packed.
    # Packing persists the pixel buffer inside the .blend.
    try:
        new_img.pack()
    except Exception:
        pass

    return new_img


def enforce_square_pot_textures(scene: bpy.types.Scene):
    fixed_cache = {}  # img.name -> new_img
    touched_nodes = 0
    fixed_images = 0

    mats = set()
    for obj in scene.objects:
        if obj.type in {"MESH", "CURVE", "SURFACE", "META", "FONT"}:
            for slot in obj.material_slots:
                if slot.material:
                    mats.add(slot.material)

    for mat in mats:
        if not mat.use_nodes or not mat.node_tree:
            continue

        nt = mat.node_tree
        for node in nt.nodes:
            if node.type != "TEX_IMAGE":
                continue

            img = node.image
            if img is None:
                continue

            w, h = img.size
            if not needs_fix(w, h):
                continue

            if img.name not in fixed_cache:
                new_img = fix_image_to_square_pot(img)
                if new_img is None:
                    continue
                fixed_cache[img.name] = new_img
                fixed_images += 1
            else:
                new_img = fixed_cache[img.name]

            node.image = new_img
            touched_nodes += 1

    log(f"[Texture POT] Fixed images: {fixed_images}, updated texture nodes: {touched_nodes}")
    return fixed_cache


# -------------------------------------------------------------------
# Pipeline steps
# -------------------------------------------------------------------
def delete_hide_render_objects(scene: bpy.types.Scene):
    to_delete = [obj for obj in scene.objects if obj.hide_render]
    for obj in to_delete:
        bpy.data.objects.remove(obj, do_unlink=True)
    log(f"Deleted {len(to_delete)} objects with hide_render=True (camera icon off).")


def find_layout_json(input_dir: Path) -> Path:
    input_dir = Path(input_dir)
    candidates = sorted(input_dir.glob("layout_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No layout_*.json found in: {input_dir}")
    return candidates[0]


def load_layout(layout_path: Path) -> dict:
    with Path(layout_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def build_room_shell(room: dict, room_id: str, materials_dir: Path):
    W = room["dimensions"]["width"]
    L = room["dimensions"]["length"]
    H = room["dimensions"]["height"]

    floor, ceil = make_floor_ceiling(W, L, H, name_prefix=room_id)
    unwrap_basic(floor)
    unwrap_basic(ceil)

    floor_mat_name = room.get("floor_material")
    default_wall_material = None
    if room.get("walls"):
        default_wall_material = room["walls"][0].get("material")

    # Dataset often has no dedicated ceiling material. In that case use wall material.
    ceil_mat_name = room.get("ceiling_material") or default_wall_material

    floor_tex = surface_texture_path(materials_dir, floor_mat_name)
    ceil_tex = surface_texture_path(materials_dir, ceil_mat_name)

    if floor_tex and floor_tex.is_file():
        assign_texture_pbr_material(
            floor, floor_tex, metallic=0.0, roughness=DEFAULT_FLOOR_ROUGHNESS, mat_name=f"{room_id}_floor_mat"
        )
    else:
        log("Floor texture missing (ok):", str(floor_tex), "material=", floor_mat_name)

    if ceil_tex and ceil_tex.is_file():
        assign_texture_pbr_material(
            ceil, ceil_tex, metallic=0.0, roughness=DEFAULT_CEIL_ROUGHNESS, mat_name=f"{room_id}_ceil_mat"
        )
    else:
        log("Ceiling texture missing (ok):", str(ceil_tex), "material=", ceil_mat_name)

    # Walls
    for w in room.get("walls", []):
        sp = w["start_point"]
        ep = w["end_point"]
        wall_mat_name = w.get("material") or default_wall_material
        wall_tex = surface_texture_path(materials_dir, wall_mat_name)

        wall_obj = make_wall(
            start=(sp["x"], sp["y"], sp.get("z", 0.0)),
            end=(ep["x"], ep["y"], ep.get("z", 0.0)),
            height=w.get("height", H),
            thickness=w.get("thickness", 0.1),
            name=w.get("id", "Wall"),
        )
        if wall_obj and wall_tex and wall_tex.is_file():
            unwrap_basic(wall_obj)
            assign_texture_pbr_material(
                wall_obj, wall_tex, metallic=0.0, roughness=DEFAULT_WALL_ROUGHNESS, mat_name=f"wall_mat_{wall_mat_name}"
            )
        elif wall_obj:
            log("Wall texture missing (ok):", str(wall_tex), "wall=", w.get("id"), "material=", wall_mat_name)

    return W, L, H


def build_doors(room: dict, wall_geom_by_id: dict, materials_dir: Path):
    for d in room.get("doors", []):
        door_mat = d.get("door_material", "Door_9")
        wall_id = d["wall_id"]
        wall_json = wall_geom_by_id.get(wall_id)

        if not wall_json:
            log("[Door] Missing wall geometry for wall_id:", wall_id)
            continue

        door_tex, door_uvpkl, frame_tex, frame_uvpkl = door_asset_paths(materials_dir, door_mat)

        if not door_tex.is_file() or not door_uvpkl.is_file():
            log("[Door] Missing door texture/uv:", str(door_tex), str(door_uvpkl))
            continue

        # Door leaf
        door_obj = build_door_plane(d.get("id", "Door"), d["width"], d["height"])
        try:
            uv_data = load_uv_pkl(door_uvpkl)
            uv_list = _as_uv_list(uv_data)
            apply_uvs_to_quad(door_obj, uv_list, flip_v=DOOR_FLIP_V)
        except Exception as e:
            log("[Door] UV apply failed:", str(door_uvpkl), "err:", e)

        assign_texture_pbr_material(door_obj, door_tex, metallic=0.0, roughness=0.6, mat_name=f"mat_{door_mat}")

        place_door_on_wall(
            door_obj,
            wall_json["start_point"],
            wall_json["end_point"],
            d.get("position_on_wall", 0.5),
            d["height"],
            wall_thickness=wall_json.get("thickness", 0.1),
            z_bottom=0.0,
            inward_sign=DOOR_INWARD_SIGN,
            epsilon=DOOR_EPSILON,
            extra_offset=EXTRA_OFFSET,
        )

        # Optional frame
        if frame_tex.is_file() and frame_uvpkl.is_file():
            frame_obj = build_door_plane(f"{d.get('id','Door')}_frame", d["width"], d["height"])
            try:
                uv_data = load_uv_pkl(frame_uvpkl)
                uv_list = _as_uv_list(uv_data)
                apply_uvs_to_quad(frame_obj, uv_list, flip_v=FRAME_FLIP_V)
            except Exception as e:
                log("[DoorFrame] UV apply failed:", str(frame_uvpkl), "err:", e)

            assign_texture_pbr_material(
                frame_obj, frame_tex, metallic=0.0, roughness=0.7, mat_name=f"mat_{door_mat}_frame"
            )

            # slight extra offset to avoid z-fighting
            place_door_on_wall(
                frame_obj,
                wall_json["start_point"],
                wall_json["end_point"],
                d.get("position_on_wall", 0.5),
                d["height"],
                wall_thickness=wall_json.get("thickness", 0.1),
                z_bottom=0.0,
                inward_sign=DOOR_INWARD_SIGN,
                epsilon=DOOR_EPSILON,
                extra_offset=0.0,
            )
        else:
            log("[Door] Frame not found (ok):", str(frame_tex), str(frame_uvpkl))


def build_objects(room: dict, objects_dir: Path):
    cache = MeshCache(objects_dir)

    for obj_entry in room.get("objects", []):
        sid = obj_entry.get("source_id")
        if not sid:
            log("Skipping without source_id:", obj_entry.get("id"))
            continue

        obj, err = cache.get_or_build(sid)
        if err:
            log(err, "for", obj_entry.get("id"))
            continue

        obj.name = obj_entry.get("id", f"{sid}_inst")

        # Placement
        place_object(obj, obj_entry["position"], obj_entry["rotation"])

        # Texture + scalar PBR
        tex_path = objects_dir / f"{sid}_texture.png"
        pbr = obj_entry.get("pbr_parameters", {}) or {}
        metallic = float(pbr.get("metallic", 0.0))
        roughness = float(pbr.get("roughness", 0.5))

        if tex_path.is_file():
            assign_texture_pbr_material(obj, tex_path, metallic=metallic, roughness=roughness, mat_name=f"mat_{sid}")
        else:
            log("Missing texture:", str(tex_path), "for", obj_entry.get("id"))


def save_blend(output_blend_path: Path):
    output_blend_path = Path(output_blend_path)
    if output_blend_path.suffix != ".blend":
        output_blend_path = output_blend_path / "room.blend"
    output_blend_path.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.save_as_mainfile(filepath=str(output_blend_path))
    log("Saved:", str(output_blend_path))
    return output_blend_path


# -------------------------------------------------------------------
# Main entry
# -------------------------------------------------------------------
def run(input_dir: Path, output_blend_path: Path):
    input_dir = Path(input_dir)
    output_blend_path = Path(output_blend_path)

    layout_path = find_layout_json(input_dir)
    materials_dir = input_dir / "materials"
    objects_dir = input_dir / "objects"

    if not materials_dir.is_dir():
        raise FileNotFoundError(f"Missing materials dir: {materials_dir}")
    if not objects_dir.is_dir():
        raise FileNotFoundError(f"Missing objects dir: {objects_dir}")

    log("Input:", str(input_dir))
    log("Layout:", str(layout_path))
    log("Materials:", str(materials_dir))
    log("Objects:", str(objects_dir))
    log("Output:", str(output_blend_path))

    clear_all_objects_and_orphans()

    layout = load_layout(layout_path)
    room = layout["rooms"][0]
    room_id = room["id"]

    wall_geom_by_id = {w["id"]: w for w in room.get("walls", [])}

    build_room_shell(room, room_id, materials_dir)
    build_doors(room, wall_geom_by_id, materials_dir)
    build_objects(room, objects_dir)

    if SET_MATERIAL_PREVIEW:
        set_material_preview_if_possible()

    # Delete non-visible objects
    delete_hide_render_objects(bpy.context.scene)

    # Enforce square + power-of-two (rescaled) textures
    enforce_square_pot_textures(bpy.context.scene)

    # Save
    save_blend(output_blend_path)

    log("Done.")


def _parse_args_after_double_dash(argv):
    """
    Blender passes script args after '--'
    """
    if "--" not in argv:
        return []
    idx = argv.index("--")
    return argv[idx + 1 :]


if __name__ == "__main__":
    args = _parse_args_after_double_dash(sys.argv)
    if len(args) < 2:
        raise SystemExit(
            "Usage:\n"
            "  blender -b -P build_rooms.py -- <input_folder> <output_blend_path>\n"
            "Example:\n"
            "  blender -b -P build_rooms.py -- /data/cf6c1dfd /data/out/cf6c1dfd.blend"
        )

    in_dir = args[0]
    out_blend_path = args[1]
    run(in_dir, out_blend_path)
