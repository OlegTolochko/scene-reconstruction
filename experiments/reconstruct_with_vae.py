"""Reconstruct a reference mesh through GenRecon's pretrained shape/PBR VAEs."""

import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch
import trimesh
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "papers/GenRecon"))
from data_toolkit_scenes.get_slats import process_instance
from genrecon import models
from genrecon.modules.sparse import SparseTensor

HF_REPO = "microsoft/TRELLIS.2-4B"
HF_REVISION = "af44b45f2e35a493886929c6d786e563ec68364d"
Y_TO_Z = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
PBR_LAYOUT = {"base_color": slice(0, 3), "metallic": slice(3, 4),
              "roughness": slice(4, 5), "alpha": slice(5, 6)}


def load_model(name):
    """Use GenRecon's model classes with pinned weights and strict loading."""
    files = {ext: hf_hub_download(HF_REPO, f"ckpts/{name}.{ext}", revision=HF_REVISION)
             for ext in ("json", "safetensors")}
    config = json.loads(Path(files["json"]).read_text())
    model = getattr(models, config["name"])(**config["args"])
    model.load_state_dict(load_file(files["safetensors"]), strict=True)
    return model.eval().cuda()


def clip(mesh, low, high):
    """Cut triangles at cube boundaries without adding cap surfaces."""
    for axis in range(3):
        for sign, bound in ((1, low[axis]), (-1, high[axis])):
            normal, origin = np.zeros(3), np.zeros(3)
            normal[axis], origin[axis] = sign, bound
            mesh = mesh.slice_plane(origin, normal, cap=False)
            if not len(mesh.faces):
                return mesh
    return mesh


def prepare_chunks(scene, output, edge, source_to_z, textures):
    """Write normalized GLBs; record each inverse transform to source space."""
    parts = []
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        if not isinstance(scene.geometry[name], trimesh.Trimesh):
            continue
        mesh = scene.geometry[name].copy()
        mesh.apply_transform(source_to_z @ transform)
        if not len(mesh.faces):
            continue
        if textures:
            visual = mesh.visual
            if isinstance(visual, trimesh.visual.TextureVisuals):
                material = visual.material.copy()
                if not isinstance(material, trimesh.visual.material.PBRMaterial):
                    material = material.to_pbr()
                uv = visual.uv if visual.uv is not None else np.zeros((len(mesh.vertices), 2))
            else:
                colors = visual.vertex_colors if visual.kind is not None else np.array([[255] * 4])
                if not np.all(colors == colors[0]):
                    raise ValueError("--textures requires PBR materials or constant vertex colors")
                material = trimesh.visual.material.PBRMaterial(
                    baseColorFactor=colors[0], metallicFactor=0., roughnessFactor=1.)
                uv = np.zeros((len(mesh.vertices), 2))
            mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
        else:
            mesh = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
        if not np.isfinite(mesh.vertices).all():
            raise ValueError("Input mesh contains nonfinite vertices")
        parts.append(mesh)
    if not parts:
        raise ValueError("Input contains no triangle meshes")
    bounds = np.array([np.min([p.bounds[0] for p in parts], axis=0),
                       np.max([p.bounds[1] for p in parts], axis=0)])
    counts = np.maximum(1, np.ceil((bounds[1] - bounds[0]) / edge).astype(int))
    grid_min = bounds.mean(axis=0) - counts * edge / 2
    records = []
    for index in itertools.product(*(range(int(n)) for n in counts)):
        low = grid_min + np.array(index) * edge
        center = low + edge / 2
        normalized = trimesh.Scene()
        for part in parts:
            mesh = clip(part.copy(), low, low + edge)
            # A triangle on an internal plane belongs to the lower-index cube.
            for axis in range(3):
                if index[axis] > 0 and len(mesh.faces):
                    mesh.update_faces(~np.all(mesh.triangles[:, :, axis] == low[axis], axis=1))
            if not len(mesh.faces) or mesh.area <= 1e-12:
                continue
            if textures:
                invalid = ~np.isfinite(mesh.visual.uv).all(axis=1)
                if invalid.any():
                    if np.any(mesh.area_faces[invalid[mesh.faces].any(axis=1)] > 1e-20):
                        raise ValueError("Clipping produced invalid UVs on a nondegenerate triangle")
                    mesh.visual.uv[invalid] = 0
            mesh.vertices = (mesh.vertices - center) / edge
            # GLB is Y-up; get_slats applies Y_TO_Z once on loading.
            mesh.apply_transform(Y_TO_Z.T)
            normalized.add_geometry(mesh)
        if not normalized.geometry:
            continue
        directory = output / f"chunk_{len(records):03d}"
        directory.mkdir()
        normalized.export(directory / "normalized.glb")
        chunk_to_z = np.diag([edge, edge, edge, 1.])
        chunk_to_z[:3, 3] = center
        records.append({"id": directory.name, "grid_index": list(index),
                        "chunk_to_source": (np.linalg.inv(source_to_z) @ chunk_to_z).tolist()})
    if not records:
        raise ValueError("Input has no nonzero-area triangles")
    return records


def rgb_voxels(vertices, source, query, chunk_to_source, resolution):
    """Interpolate source vertex RGB at the shape grid's dual-vertex positions."""
    import open3d as o3d
    local = (vertices.coords[:, 1:].float() + vertices.feats) / resolution - .5
    points = trimesh.transform_points(local.cpu().numpy(), chunk_to_source)
    colors = []
    for batch in np.array_split(points, max(1, int(np.ceil(len(points) / 25000)))):
        closest = query.compute_closest_points(o3d.core.Tensor(batch.astype(np.float32)), nthreads=4)
        triangles = source.faces[closest["primitive_ids"].numpy()]
        uv = closest["primitive_uvs"].numpy()
        weights = np.column_stack([1 - uv.sum(axis=1), uv])
        rgb = source.visual.vertex_colors[triangles, :3].astype(np.float32) / 255
        colors.append(np.einsum("ni,nic->nc", weights, rgb).clip(0, 1))
    attributes = vertices.feats.new_ones((len(points), 6))
    attributes[:, :3] = torch.from_numpy(np.concatenate(colors)).to(attributes)
    attributes[:, 3] = 0  # Metallic=0; roughness=1 and alpha=1 remain above.
    return SparseTensor(attributes * 2 - 1, vertices.coords)


def encode(chunks, args, rgb_source=None):
    query = None
    if rgb_source is not None:
        import open3d as o3d
        query = o3d.t.geometry.RaycastingScene(nthreads=4)
        query.add_triangles(o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor(np.asarray(rgb_source.vertices, dtype=np.float32)),
            o3d.core.Tensor(np.asarray(rgb_source.faces, dtype=np.uint32))))
    shape = load_model("shape_enc_next_dc_f16c32_fp16")
    texture = load_model("tex_enc_next_dc_f16c32_fp16") if args.textures else None
    for record in chunks:
        directory = args.output / record["id"]
        print(f"Encoding {directory.name}", flush=True)
        shape_encoder = shape
        if rgb_source is not None:
            def shape_encoder(vertices, intersected):
                z_shape = shape(vertices, intersected)
                attributes = rgb_voxels(vertices, rgb_source, query,
                                        np.array(record["chunk_to_source"]), args.resolution)
                z_texture = texture(attributes)
                np.savez_compressed(directory / "texture.npz",
                                    coords=z_texture.coords[:, 1:].cpu().numpy().astype(np.uint8),
                                    feats=z_texture.feats.cpu().numpy().astype(np.float32))
                return z_shape
        process_instance(
            root=str(directory), sha256=directory.name, local_path="normalized.glb",
            resolution_ovoxel=args.resolution, resolution_slat=args.resolution // 16,
            shape_encoder=shape_encoder, pbr_encoder=texture if rgb_source is None else None, ss_encoder=None,
            shape_npz=str(directory / "shape.npz"),
            pbr_npz=str(directory / "texture.npz") if args.textures and rgb_source is None else None,
            ss_npz=None, axis_fix_mode="auto")


def load_latent(path):
    with np.load(path) as saved:
        coords = torch.from_numpy(saved["coords"].astype(np.int32)).cuda()
        features = torch.from_numpy(saved["feats"].copy()).cuda()
    return SparseTensor(features, torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1))


def decode(chunks, args, source_to_z):
    shape = load_model("shape_dec_next_dc_f16c32_fp16")
    shape.set_resolution(args.resolution)
    texture = load_model("tex_dec_next_dc_f16c32_fp16") if args.textures else None
    for record in chunks:
        directory = args.output / record["id"]
        print(f"Decoding {directory.name}", flush=True)
        latent = load_latent(directory / "shape.npz")
        if args.textures:
            meshes, subdivisions = shape(latent, return_subs=True)
        else:
            meshes = shape(latent)
        mesh = meshes[0]
        raw = trimesh.Trimesh(mesh.vertices.cpu().numpy(), mesh.faces.cpu().numpy(), process=False)
        if not len(raw.faces) or not np.isfinite(raw.vertices).all():
            raise RuntimeError(f"Invalid decoded mesh: {directory.name}")
        raw.export(directory / "decoded.ply")  # Raw normalized Z-up geometry.
        if args.textures:
            tex_latent = load_latent(directory / "texture.npz")
            if not torch.equal(latent.coords, tex_latent.coords):
                raise ValueError("Shape and texture latent coordinates differ")
            voxels = texture(tex_latent, guide_subs=subdivisions) * .5 + .5
            np.savez_compressed(directory / "texture_decoded.npz",
                                coords=voxels.coords[:, 1:].cpu().numpy(),
                                attrs=voxels.feats.cpu().numpy())
            del tex_latent, voxels, subdivisions
        del latent, mesh, meshes
        torch.cuda.empty_cache()
    export_reconstruction(chunks, args, source_to_z)


def export_reconstruction(chunks, args, source_to_z):
    """Preserve decoded triangles and display the appearance VAE's RGB."""
    from scipy.spatial import cKDTree
    reconstruction = trimesh.Scene()
    for record in chunks:
        directory = args.output / record["id"]
        mesh = trimesh.load(directory / "decoded.ply", force="mesh", process=False)
        if args.textures:
            with np.load(directory / "texture_decoded.npz") as saved:
                centers = (saved["coords"] + .5) / args.resolution - .5
                _, nearest = cKDTree(centers).query(mesh.vertices, workers=4)
                rgb = saved["attrs"][nearest, :3].clip(0, 1)
                mesh.visual.vertex_colors = np.rint(rgb * 255).astype(np.uint8)
        chunk_to_gltf = Y_TO_Z.T @ source_to_z @ np.array(record["chunk_to_source"])
        reconstruction.add_geometry(mesh.apply_transform(chunk_to_gltf), node_name=directory.name)
    reconstruction.export(args.output / "reconstructed.glb")


def export_textures(chunks, args, source_to_z):
    """Use the upstream exporter for a preview, after saving raw geometry."""
    from o_voxel.postprocess import to_glb
    preview = trimesh.Scene()
    for record in chunks:
        directory = args.output / record["id"]
        print(f"Exporting textured preview {directory.name}", flush=True)
        mesh = trimesh.load(directory / "decoded.ply", force="mesh", process=False)
        with np.load(directory / "texture_decoded.npz") as saved:
            textured = to_glb(
                vertices=torch.tensor(mesh.vertices, dtype=torch.float32, device="cuda"),
                faces=torch.tensor(mesh.faces, dtype=torch.int32, device="cuda"),
                attr_volume=torch.from_numpy(saved["attrs"].copy()).cuda(),
                coords=torch.from_numpy(saved["coords"].copy()).cuda(),
                attr_layout=PBR_LAYOUT, aabb=[[-.5] * 3, [.5] * 3], grid_size=args.resolution,
                decimation_target=250000, texture_size=args.texture_size, remesh=False)
        # to_glb has already converted local Z-up coordinates to Y-up.
        chunk_to_gltf = Y_TO_Z.T @ source_to_z @ np.array(record["chunk_to_source"])
        textured.apply_transform(chunk_to_gltf @ Y_TO_Z)
        preview.add_geometry(textured, node_name=directory.name)
        torch.cuda.empty_cache()
    preview.export(args.output / "textured_preview.glb")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Reference mesh or scene")
    parser.add_argument("--output", type=Path, required=True, help="Empty output directory")
    parser.add_argument("--chunk-size", type=float, required=True, help="Cube edge in source dataset units")
    parser.add_argument("--resolution", type=int, choices=(512, 1024), default=512)
    parser.add_argument("--up-axis", choices=("y", "z"), help="Default: Y for GLB/glTF, Z otherwise")
    parser.add_argument("--textures", action="store_true", help="Also reconstruct PBR appearance or vertex RGB")
    parser.add_argument("--texture-size", type=int, default=2048)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Input mesh not found: {args.input.resolve()}. Download it or correct --input.")
    if not np.isfinite(args.chunk_size) or args.chunk_size <= 0:
        parser.error("--chunk-size must be finite and positive")
    if args.texture_size <= 0:
        parser.error("--texture-size must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be empty; choose a new directory for another run")
    if not torch.cuda.is_available():
        parser.error("The GenRecon CUDA environment is required")
    scene = trimesh.load_scene(args.input, process=False)
    vertex_rgb = args.textures and any(
        isinstance(mesh, trimesh.Trimesh) and len(mesh.vertices) > 0 and mesh.visual.kind == "vertex"
        and np.any(mesh.visual.vertex_colors != mesh.visual.vertex_colors[0])
        for mesh in scene.geometry.values())
    if vertex_rgb and any(mesh.visual.kind == "texture" for mesh in scene.geometry.values()):
        parser.error("Use a scene with either vertex RGB or PBR textures; mixed appearance sources are unsupported")
    rgb_source = scene.to_mesh() if vertex_rgb else None
    up_axis = args.up_axis or ("y" if args.input.suffix.lower() in (".glb", ".gltf") else "z")
    source_to_z = Y_TO_Z if up_axis == "y" else np.eye(4)
    args.output.mkdir(parents=True, exist_ok=True)
    chunks = prepare_chunks(scene, args.output, args.chunk_size, source_to_z, args.textures and not vertex_rgb)
    (args.output / "run.json").write_text(json.dumps({
        "input": str(args.input.resolve()), "source_up_axis": up_axis,
        "checkpoint_repository": HF_REPO, "checkpoint_revision": HF_REVISION,
        "chunk_size": args.chunk_size, "resolution": args.resolution,
        "textures": args.textures, "texture_size": args.texture_size,
        "appearance_source": "vertex_rgb" if vertex_rgb else "pbr_materials" if args.textures else None,
        "vertex_rgb_input_defaults": {"metallic": 0, "roughness": 1, "alpha": 1} if vertex_rgb else None,
        "vertex_rgb_sampling": "closest-surface barycentric RGB at quantized shape dual vertices" if vertex_rgb else None,
        "output_rgb": "nearest decoded appearance voxel at each mesh vertex" if args.textures else None,
        "posterior": "mean", "latent_normalization": False,
        "output_frame": "GLB is Y-up; decoded.ply is normalized chunk Z-up",
        "chunking": "independent nonoverlapping cubes; open triangle clipping",
        "textured_preview": "upstream cleanup and simplification; use reconstructed.glb for raw geometry",
        "chunks": chunks,
    }, indent=2) + "\n")
    torch.manual_seed(42)
    with torch.no_grad():
        encode(chunks, args, rgb_source)
        torch.cuda.empty_cache()
        decode(chunks, args, source_to_z)
        torch.cuda.empty_cache()
        if args.textures:
            export_textures(chunks, args, source_to_z)
    print(f"Saved {args.output / 'reconstructed.glb'}")


if __name__ == "__main__":
    main()
