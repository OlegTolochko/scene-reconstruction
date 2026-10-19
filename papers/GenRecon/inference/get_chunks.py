from __future__ import annotations

import json
import math
from pathlib import Path
from typing import *

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh


class BaseChunker:
    def __init__(self, min_overlap_factor: int = 4) -> None:
        self.atomic_distance = 1 / 16
        self.min_overlap = min_overlap_factor * self.atomic_distance
        self.min_padding = 3.5 / 16

    def _axis_centers(self, dim: float, chunk_size: float) -> list[float]:
        """Return the minimum number of chunk centres that cover [0, dim] on one axis.

        Centres are placed symmetrically around the midpoint. Spacing is constrained to
        multiples of atomic_distance * chunk_size and must leave at least
        min_overlap * chunk_size overlap between adjacent chunks.
        """

        atomic_stride = self.atomic_distance * chunk_size
        max_stride = chunk_size * (1.0 - self.min_overlap)
        midpoint = 0.5 * dim

        if dim <= chunk_size:
            return [midpoint]

        for n in range(2, 10_000):
            min_required_stride = (dim - chunk_size) / (n - 1)
            k = max(1, math.ceil(min_required_stride / atomic_stride - 1e-9))
            stride = k * atomic_stride
            if stride > max_stride + 1e-9:
                continue
            start = midpoint - 0.5 * (n - 1) * stride
            return [start + i * stride for i in range(n)]

        raise ValueError(f"Cannot place chunks for dim={dim}, chunk_size={chunk_size}")

    def _define_centers(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        z_floor: float,
        chunk_size: float,
        out_path: Optional[str | Path] = None,
    ) -> list[list[float]]:
        """Compute 3-D chunk centres covering the floor plan and save a top-down visualisation.

        Solves x and y axes independently via _axis_centers and combines them in y-major order.
        The z centre is fixed at z_floor + 0.5 * chunk_size.
        Writes chunk_layout.png to out_path.
        Returns a list of [x, y, z] centres.
        """
        # x_max = 2
        # y_max = 3
        padding = chunk_size * self.min_padding
        x_min_padded = x_min - padding
        x_max_padded = x_max + padding
        y_min_padded = y_min - padding
        y_max_padded = y_max + padding

        width = x_max_padded - x_min_padded
        length = y_max_padded - y_min_padded
        z_center = z_floor + 0.5 * chunk_size

        xs = [x + x_min_padded for x in self._axis_centers(width, chunk_size)]
        ys = [y + y_min_padded for y in self._axis_centers(length, chunk_size)]
        centers = [[x, y, z_center] for y in ys for x in xs]

        if out_path is not None:
            self._visualize_chunks(centers, chunk_size, x_min, x_max, y_min, y_max, out_path)

        return centers

    def _visualize_chunks(
        self,
        centers: list[list[float]],
        chunk_size: float,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        out_path: str | Path,
    ) -> None:
        """Save a top-down PNG showing the room outline and colour-coded chunk rectangles."""
        width = x_max - x_min
        length = y_max - y_min

        palette = [
            "#d6604d",
            "#4191c7",
            "#70ad47",
            "#eab308",
            "#9b59b6",
            "#16a34a",
            "#f574b9",
            "#f4721c",
            "#3b82f6",
            "#be185d",
            "#14b8a6",
            "#7f56d9",
        ]
        half = 0.5 * chunk_size
        fig, ax = plt.subplots(figsize=(8, 8))
        for i, (cx, cy, _) in enumerate(centers):
            color = palette[i % len(palette)]
            ax.add_patch(
                mpatches.Rectangle(
                    (cx - half, cy - half),
                    chunk_size,
                    chunk_size,
                    linewidth=2,
                    edgecolor=color,
                    facecolor=color,
                    alpha=0.3,
                )
            )
            ax.text(cx, cy, str(i), ha="center", va="center", fontsize=10, fontweight="bold")
        ax.add_patch(
            mpatches.Rectangle(
                (x_min, y_min),
                width,
                length,
                linewidth=3,
                edgecolor="black",
                facecolor="none",
            )
        )
        margin = max(1.0, half)
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_aspect("equal")
        ax.set_title(f"{len(centers)} chunks  (chunk_size={chunk_size:.2f} m)")
        fig.tight_layout()
        fig.savefig(Path(out_path) / "chunk_layout.png", dpi=100)
        plt.close(fig)

    def _get_transforms(
        self,
        chunk_centers: list[list[float]],
        chunk_size: float,
        out_path: str | Path,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Compute normalisation transforms for each chunk and save them as chunk_transforms.json.

        For each chunk with centre (cx, cy, cz) and scale s = 1/chunk_size:
          M_original_to_chunk  maps world coords to the normalised [-0.5, 0.5]³ cube.
          M_chunk_to_original  is the inverse (scale-translate only, so trivially inverted).
          relative_translation is the centre offset from chunk 0, scaled by 1/chunk_size.

        Returns (one entry per chunk):
          m_original_to_chunks   list of float32 tensors (4, 4)
          m_chunk_to_originals   list of float32 tensors (4, 4)
          relative_translations  list of float32 tensors (3,)
        """
        s = 1.0 / chunk_size
        ref = chunk_centers[0]

        m_o2c_list: list[torch.Tensor] = []
        m_c2o_list: list[torch.Tensor] = []
        rel_t_list: list[torch.Tensor] = []
        chunks_json = []

        for i, center in enumerate(chunk_centers):
            cx, cy, cz = center
            m_o2c = [
                [s, 0, 0, -s * cx],
                [0, s, 0, -s * cy],
                [0, 0, s, -s * cz],
                [0, 0, 0, 1],
            ]
            m_c2o = [
                [chunk_size, 0, 0, cx],
                [0, chunk_size, 0, cy],
                [0, 0, chunk_size, cz],
                [0, 0, 0, 1],
            ]
            rel_t = [(c - r) / chunk_size for c, r in zip(center, ref)]

            m_o2c_list.append(torch.tensor(m_o2c, dtype=torch.float32))
            m_c2o_list.append(torch.tensor(m_c2o, dtype=torch.float32))
            rel_t_list.append(torch.tensor(rel_t, dtype=torch.float32))
            chunks_json.append(
                {
                    "index": i,
                    "crop_center": list(center),
                    "M_original_to_chunk": m_o2c,
                    "M_chunk_to_original": m_c2o,
                    "relative_translation": rel_t,
                }
            )

        with (Path(out_path) / "chunk_transforms.json").open("w", encoding="utf-8") as f:
            json.dump({"chunks": chunks_json}, f, indent=2)

        return m_o2c_list, m_c2o_list, rel_t_list


class SageGtMixin:
    def _get_floor_ceiling_height(self) -> tuple[float, float, float]:
        """Return the fixed floor height, ceiling height, and their difference (all in metres)."""
        z_floor = 0
        z_ceiling = 2.7
        delta_z = z_ceiling - z_floor
        return z_floor, z_ceiling, delta_z

    def _get_floorplan(self, path: str | Path) -> tuple[float, float, float, float]:
        """Read the room layout JSON and return (x_min, x_max, y_min, y_max) in metres.

        Derives the JSON path from the input render directory:
        renders_room/<scene_id>  →  rooms_raw/<scene_id>/layout_<scene_id>.json
        x_min and y_min are always 0; x_max and y_max come from dimensions.width/length.
        """
        path = Path(path)
        scene_id = path.name
        layout_path = path.parent.parent / "rooms_raw" / scene_id / f"layout_{scene_id}.json"
        with layout_path.open("r", encoding="utf-8") as f:
            layout = json.load(f)
        rooms = layout.get("rooms") or []
        if not rooms:
            raise ValueError(f"No rooms in layout: {layout_path}")
        dims = rooms[0]["dimensions"]
        x_min, y_min = 0.0, 0.0
        x_max = float(dims["width"])
        y_max = float(dims["length"])
        return x_min, x_max, y_min, y_max

    def get_chunks(
        self,
        path: str | Path,
        out_path: str | Path,
    ) -> tuple[list[list[float]], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Run the full chunking pipeline for one scene and return the results.

        Reads the room layout, computes chunk centres, saves a visualisation and a transform
        JSON to out_path, and returns per-chunk lists
        (chunk_centers, m_original_to_chunks, m_chunk_to_originals, relative_translations).
        """
        z_floor, z_ceiling, delta_z = self._get_floor_ceiling_height()
        chunk_size = delta_z + 0.15
        x_min, x_max, y_min, y_max = self._get_floorplan(path)
        chunk_centers = self._define_centers(x_min, x_max, y_min, y_max, z_floor, chunk_size, out_path)
        m_original_to_chunks, m_chunk_to_originals, relative_translations = self._get_transforms(
            chunk_centers, chunk_size, out_path
        )
        return chunk_centers, m_original_to_chunks, m_chunk_to_originals, relative_translations


class ScannetGtMixin:
    def _get_room_size(self, path: str | Path) -> tuple[float, float, float, float, float, float, float]:
        """Load the aligned GT mesh and return the scene's axis-aligned bounding box extents.

        Input: path to scene (e.g. <PATH_TO_SCANNETPP_V2_OFFICIAL>/data/0f3474b837)
        Mesh:  <path>/scans/mesh_aligned_0.05.ply
        Returns (x_min, x_max, y_min, y_max, z_floor, z_ceiling, delta_z) in metres.
        """
        ply_path = Path(path) / "scans" / "mesh_aligned_0.05.ply"
        mesh = trimesh.load(str(ply_path), force="mesh")
        (x_min, y_min, z_floor), (x_max, y_max, z_ceiling) = mesh.bounds
        delta_z = z_ceiling - z_floor
        return float(x_min), float(x_max), float(y_min), float(y_max), float(z_floor), float(z_ceiling), float(delta_z)

    def get_chunks(
        self,
        path: str | Path,
        out_path: str | Path,
    ) -> tuple[list[list[float]], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Run the full chunking pipeline for one scene and return the results.

        Reads the room layout, computes chunk centres, saves a visualisation and a transform
        JSON to out_path, and returns per-chunk lists
        (chunk_centers, m_original_to_chunks, m_chunk_to_originals, relative_translations).
        """
        x_min, x_max, y_min, y_max, z_floor, z_ceiling, delta_z = self._get_room_size(path)
        chunk_size = delta_z * self.chunk_size_factor
        chunk_centers = self._define_centers(x_min, x_max, y_min, y_max, z_floor, chunk_size, out_path)
        m_original_to_chunks, m_chunk_to_originals, relative_translations = self._get_transforms(
            chunk_centers, chunk_size, out_path
        )
        return chunk_centers, m_original_to_chunks, m_chunk_to_originals, relative_translations


class ScannetMixin:
    # Sub-directory under the scene root that holds the COLMAP txt files.
    # Override in a subclass (e.g. ScannetIphoneMixin) to consume iPhone data.
    colmap_subdir: str = "dslr"
    # Defaults for COLMAP point filtering. iPhone exports strip track info
    # (track_len always 0), so ScannetIphoneMixin disables that filter.
    default_max_reproj_error: float = 2.0
    default_min_track_len: int = 4

    def _points3d_path(self, path: str | Path) -> Path:
        """Path to the COLMAP points3D.txt for this mode. Subclasses with a
        different layout (e.g. MASt3R-SfM under <scene>/colmap_mast3r/)
        override this."""
        return Path(path) / self.colmap_subdir / "colmap" / "points3D.txt"

    def _get_clean_points(
        self,
        path: str | Path,
        out_path: str | Path,
        max_reproj_error: float | None = None,
        min_track_len: int | None = None,
        stat_nb_neighbors: int = 20,
        stat_std_ratio: float = 2.0,
        radius_nb_points: int = 10,
        radius_m: float = 0.1,
    ) -> np.ndarray:
        """Load the scene's COLMAP sparse point cloud and strip outliers.

        Reads <path>/<colmap_subdir>/colmap/points3D.txt, drops points whose reprojection error
        exceeds max_reproj_error or whose track length is below min_track_len (window
        reflections typically fail both), then applies statistical + radius outlier
        removal. Saves the cleaned cloud as <out_path>/clean_points.ply and returns
        a (N, 3) float array.

        points3D.txt row format: ID X Y Z R G B ERROR TRACK[(IMG_ID, PT2D_IDX), ...]
        """
        import open3d as o3d

        if max_reproj_error is None:
            max_reproj_error = self.default_max_reproj_error
        if min_track_len is None:
            min_track_len = self.default_min_track_len

        pts_path = self._points3d_path(path)
        xyz: list[list[float]] = []
        with pts_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                error = float(parts[7])
                track_len = (len(parts) - 8) // 2
                if error > max_reproj_error or track_len < min_track_len:
                    continue
                xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
        points = np.asarray(xyz, dtype=np.float64)
        print(f"\n Initial number of points: {len(points)}")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # Outlier filters are tuned for dense ScanNet clouds (millions of pts).
        # On sparse GT clouds (e.g. vfront_gtpoints_10k with 10k pts/scene) the
        # radius filter rejects ~all points because the per-0.1m-sphere density
        # is far below the 10-neighbour threshold. Bypass when explicitly told.
        if not getattr(self, "skip_point_cleaning", False):
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=stat_nb_neighbors, std_ratio=stat_std_ratio)
            pcd, _ = pcd.remove_radius_outlier(nb_points=radius_nb_points, radius=radius_m)

        clean = np.asarray(pcd.points)
        print(f"Number points after cleaning: {len(clean)}")
        trimesh.PointCloud(clean).export(str(Path(out_path) / "clean_points.ply"))
        return clean

    def _get_room_size(
        self,
        points: np.ndarray,
        xy_low_pct: float = 0.5,
        xy_high_pct: float = 99.5,
        z_low_pct: float = 0.2,
        z_high_pct: float = 99.97,
    ) -> tuple[float, float, float, float, float, float, float]:
        """Return axis-aligned extents (from percentiles) plus delta_z.

        Percentiles make the bbox robust to residual outliers (e.g. reflections leaking
        through _get_clean_points) — a single stray point no longer inflates the bounds.
        """
        x_min = float(np.percentile(points[:, 0], xy_low_pct))
        x_max = float(np.percentile(points[:, 0], xy_high_pct))
        y_min = float(np.percentile(points[:, 1], xy_low_pct))
        y_max = float(np.percentile(points[:, 1], xy_high_pct))
        z_floor = float(np.percentile(points[:, 2], z_low_pct))
        z_ceiling = float(np.percentile(points[:, 2], z_high_pct))
        delta_z = z_ceiling - z_floor
        return x_min, x_max, y_min, y_max, z_floor, z_ceiling, delta_z

    def _remove_empty_chunks(
        self,
        chunk_centers: list[list[float]],
        chunk_size: float,
        points: np.ndarray,
        out_path: str | Path,
        n: int = 500,
    ) -> list[list[float]]:
        """Drop chunks containing fewer than n points and save a top-down visualisation.

        The PNG overlays the point cloud, kept chunks (coloured, indexed) and dropped chunks
        (dashed grey) so the filter's effect is easy to inspect.
        """
        half = 0.5 * chunk_size
        pts = np.asarray(points)
        kept: list[list[float]] = []
        removed: list[list[float]] = []
        for center in chunk_centers:
            cx, cy, cz = center
            mask = (
                (pts[:, 0] >= cx - half)
                & (pts[:, 0] <= cx + half)
                & (pts[:, 1] >= cy - half)
                & (pts[:, 1] <= cy + half)
                & (pts[:, 2] >= cz - half)
                & (pts[:, 2] <= cz + half)
            )
            if int(mask.sum()) >= n:
                kept.append(center)
            else:
                removed.append(center)

        palette = [
            "#d6604d",
            "#4191c7",
            "#70ad47",
            "#eab308",
            "#9b59b6",
            "#16a34a",
            "#f574b9",
            "#f4721c",
            "#3b82f6",
            "#be185d",
            "#14b8a6",
            "#7f56d9",
        ]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(pts[:, 0], pts[:, 1], s=0.3, c="lightgray", alpha=0.6, linewidths=0)
        for cx, cy, _ in removed:
            ax.add_patch(
                mpatches.Rectangle(
                    (cx - half, cy - half),
                    chunk_size,
                    chunk_size,
                    linewidth=1.5,
                    edgecolor="black",
                    facecolor="none",
                    linestyle="--",
                    alpha=0.4,
                )
            )
        for i, (cx, cy, _) in enumerate(kept):
            color = palette[i % len(palette)]
            ax.add_patch(
                mpatches.Rectangle(
                    (cx - half, cy - half),
                    chunk_size,
                    chunk_size,
                    linewidth=2,
                    edgecolor=color,
                    facecolor=color,
                    alpha=0.3,
                )
            )
            ax.text(cx, cy, str(i), ha="center", va="center", fontsize=10, fontweight="bold")
        ax.set_aspect("equal")
        ax.set_title(f"{len(kept)} kept / {len(removed)} dropped  (chunk_size={chunk_size:.2f} m)")
        fig.tight_layout()
        fig.savefig(Path(out_path) / "chunk_layout.png", dpi=100)
        plt.close(fig)

        return kept

    def get_chunks(
        self,
        path: str | Path,
        out_path: str | Path,
    ) -> tuple[list[list[float]], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        clean_kwargs = {
            k: getattr(self, k)
            for k in ("stat_std_ratio", "radius_nb_points", "radius_m")
            if getattr(self, k, None) is not None
        }
        clean_points = self._get_clean_points(path, out_path, **clean_kwargs)
        x_min, x_max, y_min, y_max, z_floor, z_ceiling, delta_z = self._get_room_size(clean_points)
        manual_xy = getattr(self, "manual_xy_bounds", None)
        if manual_xy is not None:
            x_min, x_max, y_min, y_max = manual_xy
        chunk_size = delta_z * self.chunk_size_factor
        chunk_centers = self._define_centers(x_min, x_max, y_min, y_max, z_floor, chunk_size)
        red_chunk_centers = self._remove_empty_chunks(
            chunk_centers,
            chunk_size,
            clean_points,
            out_path,
            n=getattr(self, "min_points_per_chunk", 500),
        )
        m_original_to_chunks, m_chunk_to_originals, relative_translations = self._get_transforms(
            red_chunk_centers, chunk_size, out_path
        )
        return red_chunk_centers, m_original_to_chunks, m_chunk_to_originals, relative_translations


class SageGtChunker(BaseChunker, SageGtMixin):
    pass


class ScannetGtChunker(BaseChunker, ScannetGtMixin):
    def __init__(self, min_overlap_factor: int = 4, chunk_size_factor: float = 1.2) -> None:
        super().__init__(min_overlap_factor=min_overlap_factor)
        self.chunk_size_factor = chunk_size_factor


class ScannetChunker(BaseChunker, ScannetMixin):
    def __init__(self, min_overlap_factor: int = 4, chunk_size_factor: float = 1.08) -> None:
        super().__init__(min_overlap_factor=min_overlap_factor)
        self.chunk_size_factor = chunk_size_factor


class ScannetIphoneMixin(ScannetMixin):
    """Same COLMAP-points pipeline as ScannetMixin, but reads from
    <scene>/iphone/colmap/ instead of <scene>/dslr/colmap/.

    ScanNet++ iPhone COLMAP exports strip the per-point TRACK[] field
    (every point has track_len=0; cf. the file header's "mean track length:
    0.0"), so we disable the track-length filter."""

    colmap_subdir: str = "iphone"
    default_min_track_len: int = 0


class ScannetIphoneChunker(BaseChunker, ScannetIphoneMixin):
    def __init__(
        self,
        min_overlap_factor: int = 4,
        chunk_size_factor: float = 1.08,
        min_points_per_chunk: int | None = None,
        skip_point_cleaning: bool = False,
    ) -> None:
        super().__init__(min_overlap_factor=min_overlap_factor)
        self.chunk_size_factor = chunk_size_factor
        if min_points_per_chunk is not None:
            self.min_points_per_chunk = min_points_per_chunk
        self.skip_point_cleaning = skip_point_cleaning


class IphoneMixin(ScannetIphoneMixin):
    """MASt3R-SfM / hloc COLMAP export at <scene>/<colmap_subdir>/.

    Differs from ScannetIphoneMixin only in path layout: the COLMAP txt files
    live in <scene>/<colmap_subdir>/ (not <scene>/iphone/colmap/). MASt3R writes
    ERROR=0.0 for every point and no TRACK info, so we keep the existing
    track-length filter at 0 and lift the reprojection-error filter.
    """

    colmap_subdir: str = "colmap_mast3r"
    default_min_track_len: int = 0
    default_max_reproj_error: float = float("inf")

    def _points3d_path(self, path: str | Path) -> Path:
        return Path(path) / self.colmap_subdir / "points3D.txt"


class IphoneChunker(BaseChunker, IphoneMixin):
    def __init__(
        self,
        min_overlap_factor: int = 4,
        chunk_size_factor: float = 1.08,
        colmap_subdir: str = "colmap_mast3r",
        stat_std_ratio: float | None = None,
        radius_nb_points: int | None = None,
        radius_m: float | None = None,
        manual_xy_bounds: tuple[float, float, float, float] | None = None,
        min_points_per_chunk: int | None = None,
    ) -> None:
        super().__init__(min_overlap_factor=min_overlap_factor)
        self.chunk_size_factor = chunk_size_factor
        self.colmap_subdir = colmap_subdir
        self.stat_std_ratio = stat_std_ratio
        self.radius_nb_points = radius_nb_points
        self.radius_m = radius_m
        self.manual_xy_bounds = manual_xy_bounds
        if min_points_per_chunk is not None:
            self.min_points_per_chunk = min_points_per_chunk
