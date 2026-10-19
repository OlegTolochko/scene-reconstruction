"""
input:
    - sparse structure gen checkpoint
    - shape slat and texture slat checkpoint (either both at resolution 512 or both at resolution 1024)
    - mode: Sage_gt or Scannet_gt
    - path: e.g. <PATH_TO_SAGE_TEST_SET>/renders_room/0a1be5f3 for Sage_gt and <PATH_TO_SCANNETPP_V2_OFFICIAL>/data/2a1b555966 for Scannet_gt
    - output_path
does:
    - gets chunks: inference/get_chunks.py
    - gets images for chunks: inference/get_images.py
    - sets up pipeline (either 512 or 1024 depending on input checkpoints):  genrecon/pipelines/full_scene_images_to_3d.py
    - runs pipeline
    - saves output mesh + coords as .ply
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from genrecon.pipelines.full_scene_images_to_3d import FullSceneImagesTo3DPipeline
from inference.get_chunks import (
    IphoneChunker,
    SageGtChunker,
    ScannetChunker,
    ScannetGtChunker,
    ScannetIphoneChunker,
)
from inference.get_images import (
    IphoneImageSelecter,
    SageImageSelecter,
    ScannetImageSelecter,
    ScannetIphoneImageSelecter,
)
from inference.transform_to_original import (
    save_coords_to_original,
    save_mesh_to_original,
)


def _next_halving_friendly(n: int) -> int:
    """Smallest m >= n such that halving m down to <= 32 never lands on an odd > 32.

    Required by ``cumesh.remeshing.remesh_narrow_band_dc``, which asserts
    ``base_resolution % 2 == 0`` while the resolution is > 32.
    """

    def ok(x: int) -> bool:
        while x > 32:
            if x % 2 != 0:
                return False
            x //= 2
        return True

    while not ok(n):
        n += 1
    return n


def _seed_everything(seed: int) -> None:
    """Seed all global RNGs so the run is reproducible from --seed alone."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


MODES = {
    "Sage_gt": (SageGtChunker, SageImageSelecter, lambda p: p / "transforms.json"),
    "Scannet_gt": (
        ScannetGtChunker,
        ScannetImageSelecter,
        lambda p: p / "dslr" / "nerfstudio" / "transforms_undistorted.json",
    ),
    "Scannet_colmap": (
        ScannetChunker,
        ScannetImageSelecter,
        lambda p: p / "dslr" / "nerfstudio" / "transforms_undistorted.json",
    ),
    "Scannet_iphone": (
        ScannetIphoneChunker,
        ScannetIphoneImageSelecter,
        lambda p: p / "iphone" / "colmap" / "cameras.txt",
    ),
    "Iphone": (
        IphoneChunker,
        IphoneImageSelecter,
        lambda p: p / "colmap" / "cameras.txt",
    ),
}


def _save_plys(
    out_path: Path,
    scene_mesh,
    coords_list,
    coords_resolution: int,
    chunk_indices,
    *,
    mesh_transform: torch.Tensor,
    coords_transform,
    label: str,
) -> None:
    """Save the merged mesh + per-chunk coords as .ply under ``out_path``.

    ``mesh_transform`` lifts the mesh; ``coords_transform(chunk_idx)`` returns
    the per-chunk transform for its coords (identity for the chunk-local save,
    ``m_c2o[chunk_idx]`` for the world save).
    """
    save_mesh_to_original(out_path / "mesh.ply", scene_mesh, mesh_transform)
    print(label)
    for coords, chunk_idx in zip(coords_list, chunk_indices):
        save_coords_to_original(
            out_path / f"coords_{chunk_idx:03d}.ply", coords, coords_resolution, coords_transform(chunk_idx)
        )


def _save_to_glb_inputs(
    out_path: Path,
    scene_mesh,
    *,
    vertices: torch.Tensor,
    voxel_size: float,
    origin,
    label: str,
) -> None:
    """Build the chunked-GLB conversion inputs and save them to ``to_glb_inputs.pt``.

    The grid is padded up to satisfy the remesh halving constraint (see
    ``_next_halving_friendly``). ``vertices``/``voxel_size``/``origin`` must
    already be in the target frame (chunk-local or world); ``label`` is a prefix
    for the log line (e.g. ``"chunk-local "`` or ``""``).
    """
    gx, gy, gz = (int(v) for v in scene_mesh.voxel_shape[2:])
    gx_p, gy_p, gz_p = (_next_halving_friendly(v) for v in (gx, gy, gz))
    if (gx_p, gy_p, gz_p) != (gx, gy, gz):
        print(f"[chunked] padding grid ({gx}, {gy}, {gz}) → ({gx_p}, {gy_p}, {gz_p}) for remesh halving constraint")
    aabb = [
        list(origin),
        [origin[i] + (gx_p, gy_p, gz_p)[i] * voxel_size for i in range(3)],
    ]
    to_glb_inputs = {
        "vertices": vertices.detach().cpu(),
        "faces": scene_mesh.faces.detach().cpu(),
        "attr_volume": scene_mesh.attrs.detach().cpu(),
        "coords": scene_mesh.coords.detach().cpu(),
        "attr_layout": scene_mesh.layout,
        "aabb": aabb,
        "voxel_size": voxel_size,
    }
    torch.save(to_glb_inputs, out_path / "to_glb_inputs.pt")
    print(f"[chunked] saved {label}to_glb_inputs to {out_path / 'to_glb_inputs.pt'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=list(MODES))
    parser.add_argument("--path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--ss_ckpt", default=None)
    parser.add_argument("--shape_ckpt", default=None)
    parser.add_argument("--tex_ckpt", default=None)
    parser.add_argument("--pipeline", choices=["512"], default="512")
    parser.add_argument(
        "--pipeline_config",
        default=None,
        help="Path to a pipeline config .json (sampler configs etc). "
        "Defaults to ImagesTo3DPipeline.DEFAULT_PIPELINE_CONFIG_FILE "
        "(configs/pipelines/original.json).",
    )
    parser.add_argument("--num_imgs_per_scene", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save_imgs",
        action="store_true",
        help="Save the selected scene images to <output_path>/scene/ and the "
        "per-chunk 2D cond views to <output_path>/chunk_<idx>/cond2d.png.",
    )
    parser.add_argument(
        "--boundary_sensitive_slat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable boundary-sensitive overlap aggregation for shape + tex SLat "
        "(on by default; disable with --no-boundary_sensitive_slat).",
    )
    parser.add_argument(
        "--boundary_width_slat",
        type=int,
        default=1,
        help="Boundary width (in latent voxels) for SLat aggregation.",
    )
    parser.add_argument(
        "--min_overlap_factor",
        type=int,
        default=4,
        help="Sets BaseChunker.min_overlap = min_overlap_factor * atomic_distance "
        "(atomic_distance = 1/16). Integer; default 4 -> 1/4 chunk overlap.",
    )
    parser.add_argument(
        "--occ_threshold",
        type=float,
        default=-1.0,
        help="Threshold applied to SS occupancy logits when extracting coords "
        "(default -1.0, more permissive than sigmoid > 0.5). Higher = more "
        "confident / sparser; lower = more permissive / denser.",
    )
    parser.add_argument(
        "--chunk_size_factor",
        type=float,
        default=1.11,
        help="Override chunk_size = delta_z * chunk_size_factor (default 1.11). "
        "Applied to all multiplicative-chunk modes. Not supported for Sage_gt "
        "(additive chunk_size); passing it explicitly there is an error.",
    )
    parser.add_argument(
        "--colmap_subdir",
        default="colmap",
        help="Subdirectory under --path holding cameras.txt/images.txt/points3D.txt "
        "for mode=Iphone (e.g. colmap_mast3r, colmap_hloc). Ignored for other modes.",
    )
    parser.add_argument(
        "--min_points_per_chunk",
        type=int,
        default=None,
        help="Override the chunker's min-points-per-chunk threshold. ScannetMixin "
        "defaults to 500, which is too aggressive for sparse-point datasets like "
        "vfront_gtpoints_10k (only ~10k pts/scene). Lower to e.g. 30 for those.",
    )
    parser.add_argument(
        "--skip_point_cleaning",
        action="store_true",
        help="Skip the statistical+radius outlier filters in _get_clean_points. "
        "Those filters are tuned for dense COLMAP clouds (millions of pts) and "
        "reject ~all points on sparse GT clouds (e.g. vfront_gtpoints_10k).",
    )
    parser.add_argument(
        "--stat_std_ratio",
        type=float,
        default=None,
        help="Iphone mode only. Override std_ratio for open3d statistical outlier "
        "removal in _get_clean_points (default 2.0). Larger = less aggressive.",
    )
    parser.add_argument(
        "--radius_nb_points",
        type=int,
        default=None,
        help="Iphone mode only. Override nb_points for open3d radius outlier "
        "removal in _get_clean_points (default 10). Smaller = less aggressive.",
    )
    parser.add_argument(
        "--radius_m",
        type=float,
        default=None,
        help="Iphone mode only. Override radius (meters) for open3d radius outlier "
        "removal in _get_clean_points (default 0.1). Larger = less aggressive.",
    )
    parser.add_argument(
        "--validation_crop_idx",
        type=int,
        default=None,
        help="If set, bypass the chunker and read the crop transform from "
        "<path>/../../crops/<scene_id>.json[chunks][idx]. Mesh + to_glb_inputs are "
        "saved in chunk-local coords (matching the validation GT GLB frame). "
        "Sage_gt mode only.",
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="iPhone modes only: revert to the legacy single-square crop around "
        "the principal point. Default (flag omitted) crops non-square frames "
        "into two square views along the longer axis, doubling the scene-wide "
        "input pool. Ignored for non-iPhone modes.",
    )
    parser.add_argument(
        "--proj_batch_voxels",
        type=int,
        default=None,
        help="Override FullSceneImagesTo3DPipeline.proj_batch_voxels (default "
        "8192). Streams the voxel dim through the global projection+aggregator "
        "in chunks of this size; smaller = lower peak VRAM, more iterations. "
        "Pure performance knob, no numerical effect. Drop to 2048/1024 when "
        "running with very many views (e.g. all-frames iPhone captures).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    missing = [
        n
        for n, v in [("--ss_ckpt", args.ss_ckpt), ("--shape_ckpt", args.shape_ckpt), ("--tex_ckpt", args.tex_ckpt)]
        if v is None
    ]
    if missing:
        parser.error(f"{', '.join(missing)} required.")

    if args.validation_crop_idx is not None and args.mode != "Sage_gt":
        parser.error("--validation_crop_idx is only supported for --mode Sage_gt.")

    _seed_everything(args.seed)

    scene_path = Path(args.path)
    out_path = Path(args.output_path)
    out_path.mkdir(parents=True, exist_ok=True)

    with (out_path / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    pipeline = FullSceneImagesTo3DPipeline.from_finetuned(
        stage_models={
            "sparse_structure_flow_model": args.ss_ckpt,
            f"shape_slat_flow_model_{args.pipeline}": args.shape_ckpt,
            f"tex_slat_flow_model_{args.pipeline}": args.tex_ckpt,
        },
        pipeline_config_file=args.pipeline_config,
    )
    if args.proj_batch_voxels is not None:
        pipeline.proj_batch_voxels = args.proj_batch_voxels
    pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    chunker_cls, selecter_cls, transforms_json = MODES[args.mode]
    if args.mode == "Iphone":
        transforms_json = lambda p, sub=args.colmap_subdir: p / sub / "cameras.txt"
    chunker_kwargs: dict = {"min_overlap_factor": args.min_overlap_factor}
    if args.mode == "Sage_gt":
        # Sage_gt uses additive chunk_size; the multiplicative factor doesn't apply.
        # Silently skip the default, but still error if the user passed it explicitly.
        if any(a == "--chunk_size_factor" or a.startswith("--chunk_size_factor=") for a in sys.argv[1:]):
            parser.error("--chunk_size_factor is not supported for mode=Sage_gt (additive chunk_size).")
    elif args.chunk_size_factor is not None:
        chunker_kwargs["chunk_size_factor"] = args.chunk_size_factor
    if args.mode == "Iphone":
        chunker_kwargs["colmap_subdir"] = args.colmap_subdir
        if args.stat_std_ratio is not None:
            chunker_kwargs["stat_std_ratio"] = args.stat_std_ratio
        if args.radius_nb_points is not None:
            chunker_kwargs["radius_nb_points"] = args.radius_nb_points
        if args.radius_m is not None:
            chunker_kwargs["radius_m"] = args.radius_m
    elif any(v is not None for v in (args.stat_std_ratio, args.radius_nb_points, args.radius_m)):
        parser.error("--stat_std_ratio/--radius_nb_points/--radius_m are Iphone-mode only.")
    if args.min_points_per_chunk is not None:
        chunker_kwargs["min_points_per_chunk"] = args.min_points_per_chunk
    if args.skip_point_cleaning:
        chunker_kwargs["skip_point_cleaning"] = True

    if args.validation_crop_idx is not None:
        scene_id = scene_path.name
        crops_json = scene_path.parent.parent / "crops" / f"{scene_id}.json"
        with crops_json.open("r", encoding="utf-8") as f:
            crops_data = json.load(f)
        crop = crops_data["chunks"][args.validation_crop_idx]
        m_o2c = [torch.tensor(crop["M_original_to_chunk"], dtype=torch.float32)]
        m_c2o = [torch.tensor(crop["M_chunk_to_original"], dtype=torch.float32)]
        rel_t = [torch.zeros(3, dtype=torch.float32)]
        with (out_path / "crop_transform.json").open("w", encoding="utf-8") as f:
            json.dump(crop, f, indent=2)
        print(f"[validation_crop] using crops/{scene_id}.json[chunks][{args.validation_crop_idx}]")
    else:
        _, m_o2c, m_c2o, rel_t = chunker_cls(**chunker_kwargs).get_chunks(scene_path, out_path)
    selecter_kwargs: dict = {}
    if args.mode in ("Scannet_iphone", "Iphone"):
        selecter_kwargs["center_crop"] = args.center_crop
    sel = selecter_cls(**selecter_kwargs).get_images(
        m_o2c,
        transforms_json(scene_path),
        args.num_imgs_per_scene,
        out_path,
        seed=args.seed,
    )
    rel_t_kept = [rel_t[i] for i in sel.chunk_indices]

    if args.save_imgs:
        scene_dir = out_path / "scene"
        scene_dir.mkdir(parents=True, exist_ok=True)
        for view_idx, img in enumerate(sel.scene_images_1024):
            arr = (img.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(arr).save(scene_dir / f"view_{view_idx:03d}.png")
        for chunk_idx, img in zip(sel.chunk_indices, sel.cond2d_images_1024):
            chunk_dir = out_path / f"chunk_{chunk_idx:03d}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            arr = (img.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(arr).save(chunk_dir / "cond2d.png")

    ss_sampler_params: dict = {}

    slat_sampler_params: dict = {}
    if args.boundary_sensitive_slat:
        slat_sampler_params = {"boundary_sensitive": True, "boundary_width": args.boundary_width_slat}
    scene_mesh, coords_list = pipeline.run(
        sel,
        rel_t_kept,
        seed=args.seed,
        pipeline_type=args.pipeline,
        sparse_structure_sampler_params=ss_sampler_params,
        shape_slat_sampler_params=slat_sampler_params,
        tex_slat_sampler_params=slat_sampler_params,
        occ_threshold=args.occ_threshold,
    )

    coords_resolution = pipeline.models[f"shape_slat_flow_model_{args.pipeline}"].resolution

    if args.validation_crop_idx is not None:
        # Chunk-local save: the GT GLB at crops/<id>_<idx>.glb lives in this exact frame,
        # and m_c2o[0] from the validation JSON is generally rotated, so the axis-aligned
        # world-lift shortcut below would be wrong. Lift later via crop_transform.json.
        identity = torch.eye(4, dtype=torch.float32)
        _save_plys(
            out_path,
            scene_mesh,
            coords_list,
            coords_resolution,
            sel.chunk_indices,
            mesh_transform=identity,
            coords_transform=lambda i: identity,
            label="\n PLY saved (chunk-local)!",
        )
        _save_to_glb_inputs(
            out_path,
            scene_mesh,
            vertices=scene_mesh.vertices,
            voxel_size=scene_mesh.voxel_size,
            origin=[scene_mesh.origin[i].item() for i in range(3)],
            label="chunk-local ",
        )

        # Chunk-local "world": chunk 0 sits at the origin and the cube has unit
        # side length, so chunked_to_glb.py can consume this directly. The
        # generated scene.glb will be in chunk-local coords (matching gt.glb).
        # Lift to true world later via M_chunk_to_original if needed.
        chunk_inputs = {
            "chunk_centers_world": torch.zeros(len(sel.chunk_indices), 3, dtype=torch.float32),
            "chunk_size_world": 1.0,
            "chunk_indices": list(sel.chunk_indices),
            "M_chunk_to_original": m_c2o[0].detach().cpu(),
        }
        torch.save(chunk_inputs, out_path / "chunk_inputs.pt")
        print(f"[chunked] saved chunk metadata to {out_path / 'chunk_inputs.pt'}")
    else:
        # Joint frame = chunker's chunk-0 local frame, so m_c2o[0] lifts it to world.
        chunk_size = m_c2o[0][0, 0].item()
        chunk_center0 = m_c2o[0][:3, 3].to(scene_mesh.vertices.device, dtype=scene_mesh.vertices.dtype)
        _save_plys(
            out_path,
            scene_mesh,
            coords_list,
            coords_resolution,
            sel.chunk_indices,
            mesh_transform=m_c2o[0],
            coords_transform=lambda i: m_c2o[i],
            label="\n PLY saved!",
        )

        # ── Save inputs for chunked GLB conversion (testing harness) ──
        _save_to_glb_inputs(
            out_path,
            scene_mesh,
            vertices=scene_mesh.vertices * chunk_size + chunk_center0,
            voxel_size=scene_mesh.voxel_size * chunk_size,
            origin=[scene_mesh.origin[i].item() * chunk_size + chunk_center0[i].item() for i in range(3)],
            label="",
        )

        chunk_inputs = {
            "chunk_centers_world": torch.stack([m_c2o[i][:3, 3] for i in sel.chunk_indices]).detach().cpu(),
            "chunk_size_world": chunk_size,
            "chunk_indices": list(sel.chunk_indices),
        }
        torch.save(chunk_inputs, out_path / "chunk_inputs.pt")
        print(f"[chunked] saved chunk metadata to {out_path / 'chunk_inputs.pt'}")


if __name__ == "__main__":
    main()
