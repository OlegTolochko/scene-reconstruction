import argparse
import gc
import math
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import torch
import trimesh
import trimesh.transformations as tra
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import o_voxel

import genrecon.models as models
import genrecon.modules.sparse as sp

DEFAULT_SHAPE_ENCODER = "microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16"
DEFAULT_PBR_ENCODER = "microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16"
DEFAULT_SS_ENCODER = "microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16"
AABB_UNIT = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
PBR_ATTRS = ["base_color", "metallic", "roughness", "alpha"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Directly encode shape/pbr/ss latents from normalized GLB assets.")
    parser.add_argument("--root", type=str, required=True, help="Dataset root containing metadata.csv")
    parser.add_argument("--resolution_ovoxel", type=int, default=1024, help="O-Voxel conversion resolution")
    parser.add_argument(
        "--resolution_slat",
        type=int,
        default=64,
        help="Shape SLAT grid resolution. Also used as sparse-structure grid resolution when SS encoding is enabled.",
    )
    parser.add_argument("--not_encode_shape", action="store_true", help="Do not voxelize or encode shape latents")
    parser.add_argument("--not_encode_tex", action="store_true", help="Do not voxelize or encode PBR/texture latents")
    parser.add_argument("--not_encode_ss", action="store_true", help="Do not encode sparse-structure latent")
    parser.add_argument("--rank", type=int, default=0, help="Rank index for data sharding")
    parser.add_argument("--world_size", type=int, default=1, help="World size for data sharding")
    parser.add_argument("--force", action="store_true", help="Recompute outputs even if target npz already exists")
    parser.add_argument(
        "--axis_fix_mode",
        type=str,
        default="auto",
        choices=["auto", "none", "x90", "x-90"],
        help=(
            "Axis correction mode before voxelization. "
            "'auto' applies +90deg around X to glTF/GLB only; "
            "'none' disables correction; "
            "'x90'/'x-90' force a global correction for all assets."
        ),
    )
    parser.add_argument(
        "--instances", type=str, default=None, help="Comma-separated IDs or path to newline-separated IDs"
    )
    parser.add_argument(
        "--do_all",
        action="store_true",
        help=(
            "Encode shape+PBR latents at both resolution 1024 and 512, "
            "and SS latents at resolution 64 (once, from the 1024 shape latents). "
            "Incompatible with --not_encode_shape / --not_encode_tex / --not_encode_ss."
        ),
    )
    return parser.parse_args()


def _axis_fix_for_asset(asset_path: str) -> np.ndarray:
    """
    Blender importer effectively works in Blender's Z-up world.
    For glTF/glb sources, applying +90deg about X maps Y-up -> Z-up.
    Keep identity for other formats unless you verify they need conversion.
    """
    ext = os.path.splitext(asset_path)[1].lower()
    if ext in {".glb", ".gltf"}:
        return tra.rotation_matrix(math.pi / 2.0, [1.0, 0.0, 0.0])
    return np.eye(4, dtype=np.float64)


def _axis_fix_matrix(asset_path: str, axis_fix_mode: str) -> np.ndarray:
    if axis_fix_mode == "none":
        return np.eye(4, dtype=np.float64)
    if axis_fix_mode == "x90":
        return tra.rotation_matrix(math.pi / 2.0, [1.0, 0.0, 0.0])
    if axis_fix_mode == "x-90":
        return tra.rotation_matrix(-math.pi / 2.0, [1.0, 0.0, 0.0])
    return _axis_fix_for_asset(asset_path)


def _load_scene_blender_like_fast(
    asset_path: str,
    *,
    flip_uv_v: bool = True,
    axis_fix_mode: str = "auto",
    keep_shape_mesh: bool = True,
    keep_pbr_scene: bool = True,
) -> tuple[Optional[trimesh.Trimesh], Optional[trimesh.Scene]]:
    """
    Fast trimesh loader that mirrors Blender-style world-space extraction:
    - no trimesh auto-processing
    - apply scene graph node transform (like obj.matrix_world)
    - optional glTF axis fix
    - optional UV V flip to mimic blender_dump_to_volumetric_attr path
    """
    loaded = trimesh.load(asset_path, force="scene", process=False, maintain_order=True)
    scene_in = loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)

    axis_fix = _axis_fix_matrix(asset_path, axis_fix_mode)

    shape_parts = [] if keep_shape_mesh else None
    pbr_scene = trimesh.Scene() if keep_pbr_scene else None
    num_mesh_geometries = 0

    for node_name in scene_in.graph.nodes_geometry:
        node_tf, geom_name = scene_in.graph[node_name]
        world_tf = axis_fix @ node_tf

        g = scene_in.geometry[geom_name]
        if not isinstance(g, trimesh.Trimesh):
            continue
        num_mesh_geometries += 1
        g = g.copy()
        g.apply_transform(world_tf)

        # Reflection transform changes handedness; flip winding to keep orientation consistent
        if np.linalg.det(world_tf[:3, :3]) < 0:
            g.faces = g.faces[:, [0, 2, 1]]

        # Keep geometry-only copy for shape path
        if keep_shape_mesh:
            shape_parts.append(
                trimesh.Trimesh(
                    vertices=g.vertices.copy(),
                    faces=g.faces.copy(),
                    process=False,
                )
            )

        # Keep per-geometry materials/UVs for PBR path
        if (
            keep_pbr_scene
            and flip_uv_v
            and isinstance(g.visual, trimesh.visual.TextureVisuals)
            and g.visual.uv is not None
        ):
            uv = g.visual.uv.copy()
            uv[:, 1] = 1.0 - uv[:, 1]
            g.visual = trimesh.visual.TextureVisuals(uv=uv, material=g.visual.material)

        if keep_pbr_scene:
            pbr_scene.add_geometry(g, node_name=node_name)

    if num_mesh_geometries == 0:
        raise ValueError("No mesh geometry after scene flattening")

    shape_mesh = trimesh.util.concatenate(shape_parts) if keep_shape_mesh else None
    return shape_mesh, pbr_scene


def latent_name(model_path: str, resolution: int) -> str:
    return f"{model_path.rstrip('/').split('/')[-1]}_{resolution}"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_npz(path: str, data: dict) -> None:
    tmp_path = f"{path}.tmp.npz"
    np.savez_compressed(tmp_path, **data)
    os.replace(tmp_path, path)


def parse_instances_arg(instances_arg: str) -> set[str]:
    if os.path.exists(instances_arg):
        with open(instances_arg, "r") as f:
            return set(line.strip() for line in f if line.strip())
    return set(x.strip() for x in instances_arg.split(",") if x.strip())


def load_metadata(root: str, instances_arg: Optional[str]) -> pd.DataFrame:
    metadata_path = os.path.join(root, "metadata.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.csv not found at {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    required = {"sha256", "local_path"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"metadata.csv is missing required columns: {sorted(missing)}")

    metadata = metadata[metadata["local_path"].notna()].copy()
    metadata["sha256"] = metadata["sha256"].astype(str)
    metadata["local_path"] = metadata["local_path"].astype(str)

    if instances_arg is not None:
        selected = parse_instances_arg(instances_arg)
        metadata = metadata[metadata["sha256"].isin(selected)]

    metadata = metadata.sort_values("sha256").reset_index(drop=True)
    return metadata


def shard_metadata(metadata: pd.DataFrame, rank: int, world_size: int) -> pd.DataFrame:
    if world_size <= 0:
        raise ValueError("world_size must be >= 1")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size - 1}], got {rank}")
    start = len(metadata) * rank // world_size
    end = len(metadata) * (rank + 1) // world_size
    return metadata.iloc[start:end].reset_index(drop=True)


def is_within_unit_cube(vertices: torch.Tensor, eps: float = 1e-3) -> bool:
    v_min = vertices.min(dim=0).values
    v_max = vertices.max(dim=0).values
    return bool(torch.all(v_min >= -0.5 - eps) and torch.all(v_max <= 0.5 + eps))


def read_existing_records(
    sha256: str,
    shape_npz: Optional[str],
    pbr_npz: Optional[str],
    ss_npz: Optional[str],
) -> tuple[Optional[dict], Optional[dict], Optional[dict]]:
    shape_record = None
    if shape_npz is not None:
        shape_tokens = int(np.load(shape_npz)["coords"].shape[0])
        shape_record = {"sha256": sha256, "shape_latent_encoded": True, "shape_latent_tokens": shape_tokens}
    pbr_record = None
    if pbr_npz is not None:
        pbr_tokens = int(np.load(pbr_npz)["coords"].shape[0])
        pbr_record = {"sha256": sha256, "pbr_latent_encoded": True, "pbr_latent_tokens": pbr_tokens}
    ss_record = None
    if ss_npz is not None:
        _ = np.load(ss_npz)["z"]
        ss_record = {"sha256": sha256, "ss_latent_encoded": True}
    return shape_record, pbr_record, ss_record


def process_instance(
    root: str,
    sha256: str,
    local_path: str,
    resolution_ovoxel: int,
    resolution_slat: int,
    shape_encoder,
    pbr_encoder,
    ss_encoder,
    shape_npz: Optional[str],
    pbr_npz: Optional[str],
    ss_npz: Optional[str],
    axis_fix_mode: str,
) -> tuple[Optional[dict], Optional[dict], Optional[dict]]:
    asset_path = local_path if os.path.isabs(local_path) else os.path.join(root, local_path)
    if not os.path.exists(asset_path):
        raise FileNotFoundError(f"Asset not found: {asset_path}")

    encode_shape = shape_npz is not None
    shape_mesh, pbr_scene = _load_scene_blender_like_fast(
        asset_path,
        flip_uv_v=False,
        axis_fix_mode=axis_fix_mode,
        keep_shape_mesh=encode_shape,
        keep_pbr_scene=pbr_npz is not None,
    )

    shape_record = None
    shape_coords = None

    if encode_shape:
        if shape_encoder is None:
            raise ValueError("shape_encoder is required when shape encoding is enabled")
        if shape_mesh is None or len(shape_mesh.vertices) == 0 or len(shape_mesh.faces) == 0:
            raise ValueError("Empty mesh geometry")

        vertices = torch.from_numpy(shape_mesh.vertices).float()
        faces = torch.from_numpy(shape_mesh.faces).long()

        if not is_within_unit_cube(vertices):
            print(f"[Warn] {sha256}: mesh vertices are outside [-0.5, 0.5]. Proceeding without normalization.")

        voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
            vertices,
            faces,
            grid_size=resolution_ovoxel,
            aabb=AABB_UNIT,
            face_weight=1.0,
            boundary_weight=0.2,
            regularization_weight=1e-2,
            timing=False,
        )
        order = torch.argsort(o_voxel.serialize.encode_seq(voxel_indices))
        voxel_indices = voxel_indices[order]
        dual_vertices = dual_vertices[order]
        intersected = intersected[order]

        dual_vertices = dual_vertices * resolution_ovoxel - voxel_indices
        dual_vertices = torch.clamp(dual_vertices, 0.0, 1.0)
        dual_vertices = (dual_vertices * 255).to(torch.uint8)
        intersected = (intersected[:, 0:1] + 2 * intersected[:, 1:2] + 4 * intersected[:, 2:3]).to(torch.uint8)

        vertices_sparse = sp.SparseTensor(
            (dual_vertices.float() / 255.0),
            torch.cat([torch.zeros_like(voxel_indices[:, 0:1]), voxel_indices], dim=-1),
        )
        intersected_sparse = vertices_sparse.replace(
            torch.cat([intersected % 2, intersected // 2 % 2, intersected // 4 % 2], dim=-1).bool()
        )

        with torch.no_grad():
            z_shape = shape_encoder(vertices_sparse.cuda(), intersected_sparse.cuda())

        shape_feats = z_shape.feats.cpu().numpy().astype(np.float32)
        shape_coords = z_shape.coords[:, 1:].cpu().numpy().astype(np.uint8)
        save_npz(shape_npz, {"feats": shape_feats, "coords": shape_coords})
        shape_record = {
            "sha256": sha256,
            "shape_latent_encoded": True,
            "shape_latent_tokens": int(shape_coords.shape[0]),
        }

    pbr_record = None

    if pbr_npz is not None:
        if pbr_encoder is None:
            raise ValueError("pbr_encoder is required when PBR encoding is enabled")
        if pbr_scene is None:
            raise ValueError("PBR scene is required when PBR encoding is enabled")
        pbr_coords, pbr_attr = o_voxel.convert.textured_mesh_to_volumetric_attr(
            pbr_scene,
            grid_size=resolution_ovoxel,
            aabb=AABB_UNIT,
            mip_level_offset=0.0,
            verbose=False,
            timing=False,
        )

        for key in PBR_ATTRS:
            if key not in pbr_attr:
                raise KeyError(f"Missing PBR voxel attribute: {key}")
        pbr_order = torch.argsort(o_voxel.serialize.encode_seq(pbr_coords))
        pbr_coords = pbr_coords[pbr_order]
        pbr_attr = {k: v[pbr_order] for k, v in pbr_attr.items()}
        pbr_feats = torch.cat([pbr_attr[k] for k in PBR_ATTRS], dim=-1) / 255.0 * 2.0 - 1.0
        pbr_sparse = sp.SparseTensor(
            pbr_feats.float(),
            torch.cat([torch.zeros_like(pbr_coords[:, 0:1]), pbr_coords], dim=-1),
        )

        with torch.no_grad():
            z_pbr = pbr_encoder(pbr_sparse.cuda())

        pbr_feats = z_pbr.feats.cpu().numpy().astype(np.float32)
        pbr_coords = z_pbr.coords[:, 1:].cpu().numpy().astype(np.uint8)
        save_npz(pbr_npz, {"feats": pbr_feats, "coords": pbr_coords})
        pbr_record = {"sha256": sha256, "pbr_latent_encoded": True, "pbr_latent_tokens": int(pbr_coords.shape[0])}

    ss_record = None
    if ss_npz is not None:
        if shape_coords is None:
            raise ValueError("Shape latents are required to encode sparse-structure latents")
        shape_coords_t = torch.from_numpy(shape_coords).long()
        if shape_coords_t.numel() > 0 and int(shape_coords_t.max()) >= resolution_slat:
            raise ValueError(
                f"Shape latent coords exceed resolution_slat={resolution_slat}. "
                "Use a larger resolution_slat or a compatible shape latent scale."
            )
        ss = torch.zeros(
            (1, resolution_slat, resolution_slat, resolution_slat),
            dtype=torch.float32,
            device="cuda",
        )
        if shape_coords_t.numel() > 0:
            ss[:, shape_coords_t[:, 0], shape_coords_t[:, 1], shape_coords_t[:, 2]] = 1.0

        with torch.no_grad():
            z_ss = ss_encoder(ss.unsqueeze(0), sample_posterior=False)
        save_npz(ss_npz, {"z": z_ss[0].cpu().numpy().astype(np.float32)})
        ss_record = {"sha256": sha256, "ss_latent_encoded": True}

    gc.collect()

    return shape_record, pbr_record, ss_record


def main() -> None:
    args = parse_args()
    if args.do_all and (args.not_encode_shape or args.not_encode_tex or args.not_encode_ss):
        raise ValueError("--do_all is incompatible with --not_encode_shape / --not_encode_tex / --not_encode_ss.")
    if args.not_encode_shape and not args.not_encode_ss:
        raise ValueError("--not_encode_shape requires --not_encode_ss because SS latents depend on shape latents.")
    if args.not_encode_shape and args.not_encode_tex and args.not_encode_ss:
        raise ValueError("Nothing to encode. Disable at most two of shape, PBR, and SS latents.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    root = os.path.abspath(args.root)
    metadata = load_metadata(root, args.instances)
    metadata = shard_metadata(metadata, args.rank, args.world_size)

    ovoxel_resolutions = [1024, 512] if args.do_all else [args.resolution_ovoxel]
    ss_name = latent_name(DEFAULT_SS_ENCODER, args.resolution_slat)
    ss_dir = os.path.join(root, "ss_latents", ss_name)

    # Per-resolution dirs and record lists for shape and PBR.
    shape_dirs: dict = {}
    pbr_dirs: dict = {}
    shape_records_by_res: dict = {}
    pbr_records_by_res: dict = {}
    for res in ovoxel_resolutions:
        shape_dirs[res] = os.path.join(root, "shape_latents", latent_name(DEFAULT_SHAPE_ENCODER, res))
        pbr_dirs[res] = os.path.join(root, "pbr_latents", latent_name(DEFAULT_PBR_ENCODER, res))
        shape_records_by_res[res] = []
        pbr_records_by_res[res] = []
        if not args.not_encode_shape:
            ensure_dir(shape_dirs[res])
            ensure_dir(os.path.join(shape_dirs[res], "new_records"))
        if not args.not_encode_tex:
            ensure_dir(pbr_dirs[res])
            ensure_dir(os.path.join(pbr_dirs[res], "new_records"))
    if not args.not_encode_ss:
        ensure_dir(ss_dir)
        ensure_dir(os.path.join(ss_dir, "new_records"))

    shape_encoder = None
    if not args.not_encode_shape:
        print(f"Loading shape encoder: {DEFAULT_SHAPE_ENCODER}")
        shape_encoder = models.from_pretrained(DEFAULT_SHAPE_ENCODER).eval().cuda()
    pbr_encoder = None
    if not args.not_encode_tex:
        print(f"Loading pbr encoder: {DEFAULT_PBR_ENCODER}")
        pbr_encoder = models.from_pretrained(DEFAULT_PBR_ENCODER).eval().cuda()
    ss_encoder = None
    if not args.not_encode_ss:
        print(f"Loading ss encoder: {DEFAULT_SS_ENCODER}")
        ss_encoder = models.from_pretrained(DEFAULT_SS_ENCODER).eval().cuda()

    ss_records = []
    error_records = []
    skipped = 0

    for row in tqdm(metadata.itertuples(index=False), total=len(metadata), desc="Encoding latents"):
        sha256 = str(row.sha256)
        local_path = str(row.local_path)

        # Process each o-voxel resolution. SS is only computed for the first (highest) resolution.
        asset_error = None
        for res in ovoxel_resolutions:
            is_primary = res == ovoxel_resolutions[0]
            shape_npz = os.path.join(shape_dirs[res], f"{sha256}.npz") if not args.not_encode_shape else None
            pbr_npz = os.path.join(pbr_dirs[res], f"{sha256}.npz") if not args.not_encode_tex else None
            ss_npz = os.path.join(ss_dir, f"{sha256}.npz") if (not args.not_encode_ss and is_primary) else None

            done = True
            if shape_npz is not None:
                done = done and os.path.exists(shape_npz)
            if pbr_npz is not None:
                done = done and os.path.exists(pbr_npz)
            if ss_npz is not None:
                done = done and os.path.exists(ss_npz)

            if done and not args.force:
                try:
                    shape_rec, pbr_rec, ss_rec = read_existing_records(sha256, shape_npz, pbr_npz, ss_npz)
                    if shape_rec is not None:
                        shape_records_by_res[res].append(shape_rec)
                    if pbr_rec is not None:
                        pbr_records_by_res[res].append(pbr_rec)
                    if ss_rec is not None:
                        ss_records.append(ss_rec)
                    if is_primary:
                        skipped += 1
                except Exception as e:
                    error_records.append({"sha256": sha256, "error": f"res={res} failed to read existing outputs: {e}"})
                continue

            try:
                shape_rec, pbr_rec, ss_rec = process_instance(
                    root=root,
                    sha256=sha256,
                    local_path=local_path,
                    resolution_ovoxel=res,
                    resolution_slat=args.resolution_slat,
                    shape_encoder=shape_encoder,
                    pbr_encoder=pbr_encoder,
                    ss_encoder=ss_encoder,
                    shape_npz=shape_npz,
                    pbr_npz=pbr_npz,
                    ss_npz=ss_npz,
                    axis_fix_mode=args.axis_fix_mode,
                )
                if shape_rec is not None:
                    shape_records_by_res[res].append(shape_rec)
                if pbr_rec is not None:
                    pbr_records_by_res[res].append(pbr_rec)
                if ss_rec is not None:
                    ss_records.append(ss_rec)
            except Exception as e:
                asset_error = f"res={res}: {e}"
                error_records.append({"sha256": sha256, "error": asset_error})
                torch.cuda.empty_cache()
                gc.collect()
                break  # Don't attempt lower-resolution encoding if the primary failed.

    for res in ovoxel_resolutions:
        if not args.not_encode_shape:
            pd.DataFrame.from_records(shape_records_by_res[res]).to_csv(
                os.path.join(shape_dirs[res], "new_records", f"part_{args.rank}.csv"),
                index=False,
            )
        if not args.not_encode_tex:
            pd.DataFrame.from_records(pbr_records_by_res[res]).to_csv(
                os.path.join(pbr_dirs[res], "new_records", f"part_{args.rank}.csv"),
                index=False,
            )
    if not args.not_encode_ss:
        pd.DataFrame.from_records(ss_records).to_csv(
            os.path.join(ss_dir, "new_records", f"part_{args.rank}.csv"),
            index=False,
        )
    if error_records:
        pd.DataFrame.from_records(error_records).to_csv(
            os.path.join(root, f"get_slats_errors_rank{args.rank}.csv"),
            index=False,
        )

    primary_res = ovoxel_resolutions[0]
    encoded = len(shape_records_by_res[primary_res]) if not args.not_encode_shape else len(ss_records)
    print(
        f"Done. rank={args.rank}, total={len(metadata)}, "
        f"encoded={encoded - skipped}, skipped={skipped}, errors={len(error_records)}"
    )


if __name__ == "__main__":
    main()
