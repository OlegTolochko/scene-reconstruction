from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import trimesh

from genrecon.representations import MeshWithVoxel
from genrecon.utils.mesh_utils import write_pbr_ply


def _apply_transform(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """Apply a 4x4 homogeneous transform to points of shape (N, 3)."""
    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
    points_h = torch.cat([points, ones], dim=1)
    return (transform.to(points) @ points_h.T).T[:, :3]


def save_mesh_to_original(
    out_path: str | Path,
    mesh: MeshWithVoxel,
    chunk_to_original: torch.Tensor,
) -> None:
    """map vertices to world frame, write PBR PLY."""
    vertices = _apply_transform(
        mesh.vertices.detach().cpu().float(),
        chunk_to_original.detach().cpu().float(),
    ).numpy()
    faces = mesh.faces.detach().cpu().numpy().astype(np.int32, copy=False)

    attrs = mesh.query_vertex_attrs().detach().cpu().numpy()
    layout = mesh.layout
    to_u8 = lambda a: np.clip(a * 255.0, 0.0, 255.0).astype(np.uint8)

    write_pbr_ply(
        str(out_path),
        vertices.astype(np.float32, copy=False),
        faces,
        to_u8(attrs[:, layout["base_color"]]),
        to_u8(attrs[:, layout["metallic"]].reshape(-1)),
        to_u8(attrs[:, layout["roughness"]].reshape(-1)),
        to_u8(attrs[:, layout["alpha"]].reshape(-1)),
    )


def save_coords_to_original(
    out_path: str | Path,
    coords: torch.Tensor,
    resolution: int,
    chunk_to_original: torch.Tensor,
) -> None:
    """Map voxel indices at `resolution` to world XYZ centres and export as a PLY point cloud."""
    chunk_points = (coords[:, 1:].float().cpu() + 0.5) / float(resolution) - 0.5
    world_points = _apply_transform(chunk_points, chunk_to_original.detach().cpu().float()).numpy()
    trimesh.PointCloud(world_points).export(str(out_path))
