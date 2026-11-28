"""The standard files a run writes into each scene folder, and how to read them back.

    cameras.json  one camera per input view: intrinsics, world-to-camera pose, image size
    depth.npz     one depth map per input view, at the size given in cameras.json
    points.ply    a point cloud, e.g. COLMAP's sparse points
    cells.npz     the occupied cells of GenRecon's sparse-structure grid
    mesh.ply      the reconstructed surface (mesh.ply.zst if zstd-compressed)

All of them are in the frame of the run's cameras. That frame is either the dataset's world frame in
metres (`frame = "dataset"`, e.g. a run on the true cameras) or any frame and scale (`"arbitrary"`, e.g.
VGGT-Omega). The evaluation aligns an arbitrary frame to the dataset's from the cameras (see align.py).

Cameras are OpenCV: x right, y down, z forward.

Only numpy is needed, except for meshes and points (trimesh), so the paper envs can write these files.
"""

import io
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FRAMES = ("dataset", "arbitrary")


@dataclass
class Cameras:
    names: list[str]  # image file names, as the dataset names its views
    K: np.ndarray  # (S, 3, 3) intrinsics in pixels of a width x height image
    world_to_camera: np.ndarray  # (S, 3, 4)
    width: np.ndarray  # (S,)
    height: np.ndarray  # (S,)
    frame: str

    def __post_init__(self):
        if self.frame not in FRAMES:
            raise ValueError(f"frame must be one of {FRAMES}, not {self.frame!r}")

    @property
    def centres(self) -> np.ndarray:
        """(S, 3) camera centres, -R^T t."""
        return np.einsum("nji,nj->ni", self.world_to_camera[:, :3, :3], -self.world_to_camera[:, :3, 3])

    def subset(self, names: list[str]) -> "Cameras":
        index = [self.names.index(name) for name in names]
        return Cameras(list(names), self.K[index], self.world_to_camera[index], self.width[index],
                       self.height[index], self.frame)

    def save(self, path: Path):
        views = [{"name": name, "width": int(w), "height": int(h), "K": K.tolist(), "world_to_camera": P.tolist()}
                 for name, w, h, K, P in zip(self.names, self.width, self.height, self.K, self.world_to_camera)]
        Path(path).write_text(json.dumps({"frame": self.frame, "views": views}, indent=1) + "\n")

    @classmethod
    def load(cls, path: Path) -> "Cameras":
        data = json.loads(Path(path).read_text())
        views = data["views"]
        return cls(names=[v["name"] for v in views],
                   K=np.array([v["K"] for v in views], dtype=float),
                   world_to_camera=np.array([v["world_to_camera"] for v in views], dtype=float),
                   width=np.array([v["width"] for v in views]),
                   height=np.array([v["height"] for v in views]),
                   frame=data["frame"])


@dataclass
class Depth:
    names: list[str]  # the views of cameras.json the maps belong to
    depth: np.ndarray  # (S, H, W) distance along the camera's z axis, 0 where there is no prediction
    confidence: np.ndarray | None = None  # (S, H, W), higher is more confident; its scale depends on the model

    def save(self, path: Path):
        arrays = {"names": np.array(self.names), "depth": self.depth.astype(np.float32)}
        if self.confidence is not None:
            arrays["confidence"] = self.confidence.astype(np.float32)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> "Depth":
        data = np.load(path)
        return cls([str(n) for n in data["names"]], data["depth"],
                   data["confidence"] if "confidence" in data.files else None)


@dataclass
class Cells:
    """Occupied cells of a grid made of cubic chunks of chunk_cells^3 cells, which may overlap.

    Cell (i, j, k) spans origin + [i, i + 1) * cell_size along x, and so on. Chunk c covers the cells
    [chunk_offsets[c], chunk_offsets[c] + chunk_cells) along each axis.
    """
    cells: np.ndarray  # (M, 3) int
    origin: np.ndarray  # (3,)
    cell_size: float
    chunk_offsets: np.ndarray  # (C, 3) int
    chunk_cells: int = 32

    @property
    def shape(self) -> tuple:
        """Cells per axis of the box that holds every chunk."""
        return tuple(int(n) for n in self.chunk_offsets.max(axis=0) + self.chunk_cells)

    def centres(self) -> np.ndarray:
        return self.origin + (self.cells + 0.5) * self.cell_size

    def save(self, path: Path):
        np.savez(path, cells=self.cells.astype(np.int32), origin=self.origin, cell_size=self.cell_size,
                            chunk_offsets=self.chunk_offsets.astype(np.int32), chunk_cells=self.chunk_cells)

    @classmethod
    def load(cls, path: Path) -> "Cells":
        data = np.load(path)
        return cls(data["cells"].astype(int), data["origin"], float(data["cell_size"]),
                   data["chunk_offsets"].astype(int), int(data["chunk_cells"]))


def load_mesh(path: Path):
    """A mesh from mesh.ply or mesh.ply.zst, with vertices and faces as stored."""
    import trimesh

    path = Path(path)
    if path.suffix == ".zst":
        ply = subprocess.run(["zstd", "-dc", path], capture_output=True, check=True).stdout
        return trimesh.load(io.BytesIO(ply), file_type="ply", process=False)
    return trimesh.load(path, process=False)


def find_mesh(scene_dir: Path) -> Path | None:
    for name in ("mesh.ply", "mesh.ply.zst"):
        if (Path(scene_dir) / name).exists():
            return Path(scene_dir) / name
    return None


def save_points(path: Path, points: np.ndarray, colors: np.ndarray | None = None):
    """points (N, 3); colors (N, 3) as uint8, optional."""
    import trimesh

    trimesh.PointCloud(points, colors=colors).export(path)


def load_points(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    import trimesh

    cloud = trimesh.load(path, process=False)
    colors = cloud.colors[:, :3] if len(cloud.colors) else None
    return np.asarray(cloud.vertices), colors
