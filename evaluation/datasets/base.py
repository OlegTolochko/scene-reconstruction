"""What a dataset gives the evaluation and the experiments.

Per scene: the N input views a method gets, the test views of the 2D mesh metrics, and the true surface.
True cameras are OpenCV (x right, y down, z forward) and map into the dataset's world frame, in metres.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation.formats import Cameras


@dataclass
class View:
    name: str  # image file name, unique within a scene
    image: Path
    K: np.ndarray  # (3, 3) intrinsics in pixels of a width x height image
    width: int
    height: int
    cam_to_world: np.ndarray  # (4, 4)
    mask: Path | None = None  # image whose pixels > 0 are valid; None: every pixel is valid

    def scaled(self, scale: float) -> "View":
        """The same view at scale times the resolution."""
        K = self.K.copy()
        K[:2] *= scale
        return replace(self, K=K, width=round(self.width * scale), height=round(self.height * scale))

    def valid(self) -> np.ndarray:
        """(height, width) bool, the pixels that count (False e.g. where a person was blurred out)."""
        if self.mask is None:
            return np.ones((self.height, self.width), dtype=bool)
        return np.array(Image.open(self.mask).resize((self.width, self.height), Image.NEAREST)) > 0


class Dataset(ABC):
    name: str
    # Predicted surface farther than this (metres) from the true surface is not scored, e.g. what a
    # method sees through a window where the scan has nothing.
    crop_distance: float

    @abstractmethod
    def scenes(self) -> list[str]:
        """The evaluation scenes."""

    @abstractmethod
    def input_views(self, scene: str, n: int) -> list[View]:
        """The n views every method gets as input."""

    @abstractmethod
    def test_views(self, scene: str) -> list[View]:
        """The views the 2D mesh metrics render, none of them an input view."""

    @abstractmethod
    def surface(self, scene: str):
        """The true surface as a trimesh.Trimesh."""


def true_cameras(views: list[View]) -> Cameras:
    """The dataset's cameras of the views, in the standard format."""
    return Cameras([v.name for v in views], np.stack([v.K for v in views]),
                   np.stack([np.linalg.inv(v.cam_to_world)[:3] for v in views]),
                   np.array([v.width for v in views]), np.array([v.height for v in views]), "dataset")
