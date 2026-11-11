"""Blender script: writes a room's renderable geometry as one triangle mesh in Blender's world (= room) frame.

The objects are those render_rooms.py renders and ray-casts against: meshes with hide_render off. Their
evaluated meshes (modifiers applied) are triangulated and transformed by their world matrices, the frame of
transforms.json's camera matrices. true_depth.py ray-casts this mesh for the true depth of every render.

With the room's layout JSON, the triangles are also labelled for thin_walls.py. build_rooms.py makes each wall
a box of the layout's thickness (0.1 m) centred on the wall line, which is 2-3 cells thick in the 64^3
occupancy, while a scan (ScanNet++) only has the side facing the room. `wall` marks the triangles of the
walls (objects named by a wall id of the layout), `keep` every triangle except those of a wall's outer face,
end caps and top: of a wall, only the face whose normal points into the room is kept.

    blender -b ROOM.blend --python-exit-code 1 -P export_room_mesh.py -- OUT.npz [LAYOUT.json]
OUT.npz: vertices [V, 3] float32, faces [F, 3] int32, and with the layout wall [F] and keep [F] (bool).
"""

import json
import sys

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
out = argv[0]
walls = {}  # wall id -> unit normal pointing into the room (x, y)
if len(argv) > 1:
    for room in json.load(open(argv[1]))["rooms"]:
        pts = np.array([[w[k]["x"], w[k]["y"]] for w in room["walls"] for k in ("start_point", "end_point")])
        centre = pts.mean(0)
        for w in room["walls"]:
            s = np.array([w["start_point"]["x"], w["start_point"]["y"]])
            e = np.array([w["end_point"]["x"], w["end_point"]["y"]])
            d = (e - s) / max(np.linalg.norm(e - s), 1e-9)
            n = np.array([-d[1], d[0]])
            walls[w["id"]] = n if np.dot(n, centre - (s + e) / 2) > 0 else -n

depsgraph = bpy.context.evaluated_depsgraph_get()
vertices, faces, is_wall, keep, offset, found = [], [], [], [], 0, set()
for obj in bpy.context.scene.objects:
    if obj.type != "MESH" or obj.hide_render:
        continue
    ev = obj.evaluated_get(depsgraph)
    mesh = ev.to_mesh()
    mesh.calc_loop_triangles()
    nv, nt = len(mesh.vertices), len(mesh.loop_triangles)
    if nv and nt:
        co = np.empty(nv * 3, dtype=np.float64)
        mesh.vertices.foreach_get("co", co)
        M = np.array(ev.matrix_world)
        co = co.reshape(nv, 3) @ M[:3, :3].T + M[:3, 3]
        tri = np.empty(nt * 3, dtype=np.int64)
        mesh.loop_triangles.foreach_get("vertices", tri)
        tri = tri.reshape(nt, 3)
        wall_id = next((w for w in walls if obj.name == w or obj.name.startswith(w + ".")), None)
        k = np.ones(nt, dtype=bool)
        if wall_id is not None:
            found.add(wall_id)
            a, b, c = co[tri[:, 0]], co[tri[:, 1]], co[tri[:, 2]]
            normal = np.cross(b - a, c - a)
            normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
            k = normal[:, :2] @ walls[wall_id] > 0.9  # the face looking into the room
        vertices.append(co)
        faces.append(tri + offset)
        is_wall.append(np.full(nt, wall_id is not None))
        keep.append(k)
        offset += nv
    ev.to_mesh_clear()

if not faces:
    raise RuntimeError("no renderable mesh in the scene")
missing = set(walls) - found
if missing:
    raise RuntimeError(f"walls of the layout not found in the scene: {sorted(missing)}")
v, f = np.concatenate(vertices).astype(np.float32), np.concatenate(faces).astype(np.int32)
extra = {"wall": np.concatenate(is_wall), "keep": np.concatenate(keep)} if walls else {}
np.savez_compressed(out, vertices=v, faces=f, **extra)
msg = f", {len(walls)} walls, {int((~extra['keep']).sum())} wall triangles dropped for thin_walls.py" if walls else ""
print(f"wrote {out}: {len(v)} vertices, {len(f)} triangles{msg}")
