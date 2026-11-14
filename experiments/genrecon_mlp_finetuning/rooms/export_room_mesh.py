"""Blender script: writes a room's renderable geometry as one triangle mesh in Blender's world (= room) frame.

The objects are those render_rooms.py renders and ray-casts against: meshes with hide_render off. Their
evaluated meshes (modifiers applied) are triangulated and transformed by their world matrices, the frame of
transforms.json's camera matrices. true_depth.py ray-casts this mesh for the true depth of every render.

    blender -b ROOM.blend --python-exit-code 1 -P export_room_mesh.py -- OUT.npz
OUT.npz: vertices [V, 3] float32, faces [F, 3] int32.
"""

import sys

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
out = argv[0]

depsgraph = bpy.context.evaluated_depsgraph_get()
vertices, faces, offset = [], [], 0
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
        vertices.append(co)
        faces.append(tri + offset)
        offset += nv
    ev.to_mesh_clear()

if not faces:
    raise RuntimeError("no renderable mesh in the scene")
v, f = np.concatenate(vertices).astype(np.float32), np.concatenate(faces).astype(np.int32)
np.savez_compressed(out, vertices=v, faces=f)
print(f"wrote {out}: {len(v)} vertices, {len(f)} triangles")
