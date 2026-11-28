"""ScanNet++: the 50 nvs_sem_val scenes, their undistorted DSLR images and laser scans.

Input views: data/scannetpp/eval_inputs/<N>views/<scene>/, written by scripts/select_views_scannetpp.py.
Test views: the scene's official DSLR test frames. True surface: the laser scan mesh without the faces
that were blurred out for privacy.
"""

import json
from pathlib import Path

import numpy as np
import trimesh

from evaluation.datasets.base import Dataset, View

ROOT = Path(__file__).resolve().parents[2]

# ScanNet++'s nerfstudio poses are in a world frame with x and y swapped and z flipped
# relative to the scan (the same matrix as GenRecon's inference/get_images.py).
NERFSTUDIO_TO_SCAN = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=float)
# Nerfstudio cameras look along -z with y up. OpenCV cameras look along +z with y down.
OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


class ScanNetPP(Dataset):
    name = "scannetpp"
    crop_distance = 0.15  # as GenRecon's evaluation

    def __init__(self, root: Path = ROOT / "data/scannetpp"):
        self.root = Path(root)

    def scenes(self) -> list[str]:
        return (self.root / "splits/nvs_sem_val.txt").read_text().split()

    def input_views(self, scene: str, n: int) -> list[View]:
        return self._views(scene, self.root / f"eval_inputs/{n}views/{scene}/dslr/nerfstudio/transforms_undistorted.json",
                           "frames")

    def test_views(self, scene: str) -> list[View]:
        return self._views(scene, self.root / f"data/{scene}/dslr/nerfstudio/transforms_undistorted.json", "test_frames")

    def surface(self, scene: str) -> trimesh.Trimesh:
        scans = self.root / f"data/{scene}/scans"
        scan = trimesh.load(scans / "mesh_aligned_0.05.ply", process=False)
        # vertices in regions blurred out for privacy
        anonymized = [int(i) for i in (scans / "mesh_aligned_0.05_mask.txt").read_text().split()]
        keep = ~np.isin(scan.faces, anonymized).any(axis=1)
        return trimesh.Trimesh(scan.vertices, scan.faces[keep], process=False)

    def input_dir(self, scene: str, n: int) -> Path:
        """The scene folder with only the n input views, as GenRecon's ScanNet++ mode reads it."""
        return self.root / f"eval_inputs/{n}views/{scene}"

    def colmap_dir(self, scene: str) -> Path:
        """ScanNet++'s own COLMAP model of the scene, built from all its DSLR images."""
        return self.root / f"data/{scene}/dslr/colmap"

    def _views(self, scene: str, transforms: Path, key: str) -> list[View]:
        meta = json.loads(transforms.read_text())
        K = np.array([[meta["fl_x"], 0, meta["cx"]], [0, meta["fl_y"], meta["cy"]], [0, 0, 1]])
        dslr = self.root / f"data/{scene}/dslr"
        return [View(name=Path(f["file_path"]).name,
                     image=dslr / "resized_undistorted_images" / f["file_path"],
                     K=K.copy(), width=meta["w"], height=meta["h"],
                     cam_to_world=NERFSTUDIO_TO_SCAN @ np.array(f["transform_matrix"]) @ OPENGL_TO_OPENCV,
                     mask=dslr / "resized_undistorted_masks" / f["mask_path"])
                for f in meta[key]]
