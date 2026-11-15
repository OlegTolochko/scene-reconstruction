"""Blender script: writes a room as a textured GLB for the W&B viewer, in the room's frame and Y-up, as mesh.py's
GLBs. Everything above CUT metres is cut away, so that the viewer looks into the room, and textures with a side
of more than SIDE pixels are shrunk to it. The .blend is not changed.

    blender -b ROOM.blend --python-exit-code 1 -P export_room_glb.py -- OUT.glb CUT SIDE
"""

import sys

import bmesh
import bpy
from mathutils import Vector

out, cut, side = sys.argv[sys.argv.index("--") + 1:]
cut, side = float(cut), int(side)

bpy.context.view_layer.update()  # matrix_world of every object
for obj in list(bpy.context.scene.objects):
    if obj.type != "MESH":
        continue
    M = obj.matrix_world
    heights = [(M @ v.co).z for v in obj.data.vertices]
    if not heights or max(heights) <= cut:
        continue
    if min(heights) >= cut:  # the ceiling
        bpy.data.objects.remove(obj, do_unlink=True)
        continue
    if obj.data.users > 1:  # an object that is placed several times shares its mesh
        obj.data = obj.data.copy()
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.bisect_plane(  # the plane z = cut, in the object's coordinates
        bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], dist=1e-5, plane_co=M.inverted() @ Vector((0, 0, cut)),
        plane_no=(M.to_3x3().transposed() @ Vector((0, 0, 1))).normalized(), clear_outer=True)
    bm.to_mesh(obj.data)
    bm.free()

for img in bpy.data.images:
    if not img.users:
        continue
    w, h = img.size
    if max(w, h) > side:
        img.scale(max(1, w * side // max(w, h)), max(1, h * side // max(w, h)))
        img.pack()  # the exporter then writes these pixels and not the image's file

# The settings of GenRecon's toolkit (create_chunks.py), which wrote the chunks' GLBs with this Blender.
kwargs = {"filepath": out, "export_format": "GLB", "export_apply": True, "export_yup": True, "export_texcoords": True,
          "export_normals": True, "export_materials": "EXPORT", "use_selection": False}
if "export_image_format" in bpy.ops.export_scene.gltf.get_rna_type().properties.keys():
    kwargs["export_image_format"] = "JPEG"  # smaller than PNG; textures with transparency stay PNG
bpy.ops.export_scene.gltf(**kwargs)
print(f"wrote {out}")
