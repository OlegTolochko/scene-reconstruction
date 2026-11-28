"""Fit a run's cameras to the dataset's, to bring a run in an arbitrary frame and scale into the dataset's.

The fit is a similarity transform (scale, rotation, translation) from the predicted camera centres onto
the true ones of the same images, least squares over all cameras (Umeyama), as WorldRecon (arXiv 2610.18846,
App. C.2) evaluates. It uses only the camera centres, never the geometry, and only at evaluation.
"""

from dataclasses import dataclass

import numpy as np

from evaluation.formats import Cameras


@dataclass
class Similarity:
    scale: float
    rotation: np.ndarray  # (3, 3)
    translation: np.ndarray  # (3,)

    def apply(self, points: np.ndarray) -> np.ndarray:
        return self.scale * points @ self.rotation.T + self.translation

    def inverse(self) -> "Similarity":
        return Similarity(1 / self.scale, self.rotation.T, -self.rotation.T @ self.translation / self.scale)


def umeyama(x: np.ndarray, y: np.ndarray) -> Similarity:
    """The similarity minimising sum |s R x_i + t - y_i|^2 over the point pairs (Umeyama 1991)."""
    mx, my = x.mean(axis=0), y.mean(axis=0)
    U, D, Vt = np.linalg.svd((y - my).T @ (x - mx) / len(x))
    S = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])  # a rotation, not a reflection
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / ((x - mx) ** 2).sum(axis=1).mean()
    return Similarity(s, R, my - s * R @ mx)


def transform_cameras(cameras: Cameras, fit: Similarity, frame: str) -> Cameras:
    """The cameras in the frame fit maps to. Camera coordinates are scaled with the world, so they stay rigid."""
    R, t = cameras.world_to_camera[:, :, :3], cameras.world_to_camera[:, :, 3]
    R_new = R @ fit.rotation.T
    t_new = fit.scale * t - R_new @ fit.translation
    return Cameras(cameras.names, cameras.K, np.concatenate([R_new, t_new[:, :, None]], axis=2), cameras.width,
                   cameras.height, frame)
