"""GenRecon's cell grid, and the cells stage 2 gets from a source other than stage 1 (--cells scan or depth).

GenRecon cuts a scene into overlapping cubic chunks. Stage 1 marks which cells of each chunk's 32^3 grid
contain surface, and stages 2 and 3 generate geometry only inside those cells. Here the chunks are read as
one grid over the scene (as GenRecon's FullSceneImagesTo3DPipeline._global_grid_layout lays them out):
chunk i covers the cells [offset_i, offset_i + 32) along each axis.

Sources of cells, both with the dataset's cameras, on the grid of the run:
    scan   the cells with scan surface that a view GenRecon uses sees: in front of the camera, inside the
           square crop GenRecon takes from the image, and not hidden behind other scan surface
    depth  the cells that hold depth points of at least min_views views. The depth is rendered from the scan
           at the input views (depth = scan), or predicted (depth = a cameras run): then each view's depth is
           scaled by the median of COLMAP depth / predicted depth over the points of the dataset's own COLMAP
           model the view observes (ScanNet++: built from all images of the scene). With keep_unseen, stage
           1's cells outside every input image are kept too.

Runs in GenRecon's env (inject.py) and in the root env (experiment.py).
"""

import json
from pathlib import Path

import numpy as np

from evaluation import datasets
from evaluation.formats import Cameras, Cells, Depth
from evaluation.metrics.cells import true_cells
from evaluation.metrics.mesh import raycasting_scene, render_depth

R = 32  # cells per chunk side
SCAN_DEPTH_SIZE = (624, 416)  # width, height of the depth rendered from the scan (VGGT-Omega's resolution)


def layout(run_dir: Path) -> Cells:
    """The grid of a GenRecon run, from its chunk_transforms.json, without cells."""
    chunks = json.loads((run_dir / "chunk_transforms.json").read_text())["chunks"]
    centres = np.array([c["crop_center"] for c in chunks])
    size = chunks[0]["M_chunk_to_original"][0][0]
    rel = (centres - centres[0]) / size  # chunk centres in units of the chunk size, multiples of 1/16
    offsets = np.round((rel - rel.min(axis=0)) * R).astype(int)
    origin = centres[0] + (rel.min(axis=0) - 0.5) * size  # position of the grid's corner
    return Cells(np.zeros((0, 3), int), origin, size / R, offsets, R)


def stage2_cells(run_dir: Path, grid: Cells) -> np.ndarray:
    """The cells stage 2 got, from the coords_<chunk>.ply (cell centres) GenRecon wrote."""
    import trimesh

    centres = np.concatenate([trimesh.load(p).vertices for p in sorted(run_dir.glob("coords_*.ply"))])
    return np.unique(np.round((centres - grid.origin) / grid.cell_size - 0.5).astype(int), axis=0)


def from_chunk_coords(coords: list[np.ndarray], grid: Cells) -> np.ndarray:
    """Cells of the grid from each chunk's local [0, 32)^3 coordinates."""
    return np.unique(np.concatenate([c + offset for c, offset in zip(coords, grid.chunk_offsets)]), axis=0)


def chunk_coords(cells: np.ndarray, grid: Cells) -> list[np.ndarray]:
    """Each chunk's cells in its local [0, 32)^3 coordinates.

    Stage 2 fails on a chunk without cells, so such a chunk gets one: the corner of the chunk farthest from
    every cell, preferring corners no other chunk covers.
    """
    coords = []
    for i, offset in enumerate(grid.chunk_offsets):
        local = cells - offset
        local = local[((local >= 0) & (local < R)).all(axis=1)]
        if len(local) == 0:
            corners = offset + np.array([[x, y, z] for x in (0, R - 1) for y in (0, R - 1) for z in (0, R - 1)])
            shared = [any(((c >= o) & (c < o + R)).all() for j, o in enumerate(grid.chunk_offsets) if j != i)
                      for c in corners]
            distance = np.array([np.abs(cells - c).max(axis=1).min() for c in corners])
            best = np.lexsort((-distance, shared))[0]
            local = corners[best:best + 1] - offset
        coords.append(local)
    return coords


def genrecon_views(run_dir: Path) -> Cameras:
    """The views GenRecon uses, from its cameras.json: OpenCV world-to-camera matrices in the chunk-0 frame,
    and intrinsics normalized to the square crop, so a point is inside the crop if its u and v are in [0, 1].
    """
    chunk0 = json.loads((run_dir / "chunk_transforms.json").read_text())["chunks"][0]
    size = chunk0["M_chunk_to_original"][0][0]
    world_to_chunk0 = np.diag([1 / size] * 3 + [1.0])
    world_to_chunk0[:3, 3] = -np.array(chunk0["crop_center"]) / size
    views = json.loads((run_dir / "cameras.json").read_text())["scene"]
    return Cameras([str(i) for i in range(len(views))], np.array([v["intrinsics"] for v in views]),
                   np.array([(np.array(v["extrinsics_c0"]) @ world_to_chunk0)[:3] for v in views]),
                   np.ones(len(views), int), np.ones(len(views), int), "dataset")


def scan_cells(spec: dict, grid: Cells, run_dir: Path) -> np.ndarray:
    dataset = datasets.get(spec["dataset"])
    return true_cells(dataset.surface(spec["scene"]), grid, genrecon_views(run_dir))["visible"]


def project(points: np.ndarray, P: np.ndarray, K: np.ndarray, width: int, height: int) -> tuple:
    """Pixel indices and depth of points, and whether each lies in front of the camera inside the image."""
    cam = points @ P[:, :3].T + P[:, 3]
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.floor(K[0, 0] * cam[:, 0] / z + K[0, 2])
        v = np.floor(K[1, 1] * cam[:, 1] / z + K[1, 2])
    return u, v, z, (z > 0.05) & (u >= 0) & (u < width) & (v >= 0) & (v < height)


def colmap_points(colmap: Path, names: list[str]) -> list[np.ndarray]:
    """For each view, the positions of the COLMAP points it observes."""
    with open(colmap / "points3D.txt") as f:
        rows = np.array([line.split(maxsplit=4)[:4] for line in f if line[0] != "#"], dtype=np.float64)
    positions = dict(zip(rows[:, 0].astype(int), rows[:, 1:]))
    observed = {}
    with open(colmap / "images.txt") as f:
        lines = (line for line in f if line[0] != "#")
        for header in lines:  # two lines per image: its name, then x, y and the point's ID per observation
            observations = next(lines)
            name = header.split()[9]
            if name in names:
                observed[name] = np.array([positions[i] for i in map(int, observations.split()[2::3]) if i in positions])
    return [observed[name] for name in names]


def depth_cells(spec: dict, grid: Cells, stage1: np.ndarray | None) -> np.ndarray:
    dataset = datasets.get(spec["dataset"])
    cameras = Cameras.load(spec["cameras"])
    if spec["depth"] == "scan":
        width, height = SCAN_DEPTH_SIZE
        scan = raycasting_scene(dataset.surface(spec["scene"]))
    else:
        depth = Depth.load(spec["depth"])
        cameras = cameras.subset(depth.names)
        height, width = depth.depth.shape[1:]
    K = cameras.K.copy()
    K[:, :2] *= (width / cameras.width)[:, None, None]  # at the depth's size
    P = cameras.world_to_camera

    if spec["depth"] == "scan":
        cam_to_world = np.tile(np.eye(4), (len(P), 1, 1))
        cam_to_world[:, :3, :3] = P[:, :, :3].transpose(0, 2, 1)
        cam_to_world[:, :3, 3] = cameras.centres
        depths = [render_depth(scan, k, c, width, height) for k, c in zip(K, cam_to_world)]
    else:
        depths = []
        for d, points, k, p in zip(depth.depth, colmap_points(dataset.colmap_dir(spec["scene"]), cameras.names), K, P):
            u, v, z, inside = project(points, p, k, width, height)
            depths.append(np.median(z[inside] / d[v[inside].astype(int), u[inside].astype(int)]) * d)

    # Cells inside a chunk that hold depth points of at least min_views views
    u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    per_view = []
    for d, k, p in zip(depths, K, P):
        valid = np.isfinite(d) & (d > 0.05)
        rays = np.stack([(u[valid] - k[0, 2]) / k[0, 0], (v[valid] - k[1, 2]) / k[1, 1], np.ones(valid.sum())], axis=1)
        points = (rays * d[valid, None] - p[:, 3]) @ p[:, :3]
        cells = np.floor((points - grid.origin) / grid.cell_size).astype(int)
        per_view.append(np.unique(cells[grid.inside(cells)], axis=0))
    cells, views = np.unique(np.concatenate(per_view), axis=0, return_counts=True)
    cells = cells[views >= spec["min_views"]]

    if stage1 is not None:  # keep stage 1's cells that no input image shows
        centres = grid.origin + (stage1 + 0.5) * grid.cell_size
        covered = np.zeros(len(stage1), dtype=bool)
        for k, p in zip(K, P):
            covered |= project(centres, p, k, width, height)[3]
        cells = np.unique(np.concatenate([cells, stage1[~covered]]), axis=0)
    return cells
