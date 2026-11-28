"""A run's cameras against the dataset's true cameras of the same images.

Pairwise, independent of the world frame and scale (standard for VGGT, CO3D, RealEstate10K):
    rre_deg_median  median relative rotation error between two cameras, in degrees
    rte_deg_median  median angle between the predicted and true direction from one camera to the other
    auc30           area under the accuracy curve of max(rre, rte) up to 30 degrees (higher is better)
After the similarity fit of all predicted camera centres onto the true ones (align.umeyama):
    ate_cm          RMS distance between aligned and true camera centres, in cm
    rot_deg_median  median angle between each aligned camera's orientation and the true one, in degrees
    scale           scale of the fit
"""

import numpy as np
from scipy.spatial.transform import Rotation

from evaluation.align import umeyama
from evaluation.datasets import View
from evaluation.formats import Cameras


def angle_deg(R: np.ndarray) -> np.ndarray:
    """Rotation angle of each 3x3 matrix in R (N, 3, 3), in degrees."""
    return np.degrees(Rotation.from_matrix(R).magnitude())


def evaluate(pred: Cameras, views: list[View]) -> dict:
    """views: the true cameras; every predicted camera must have one of the same name."""
    true = {view.name: view for view in views}
    R_p, C_p = pred.world_to_camera[:, :3, :3], pred.centres
    R_g = np.stack([true[name].cam_to_world[:3, :3].T for name in pred.names])
    C_g = np.stack([true[name].cam_to_world[:3, 3] for name in pred.names])

    # Pairwise errors over all ordered pairs i != j
    i, j = np.where(~np.eye(len(pred.names), dtype=bool))
    rel_p = R_p[j] @ R_p[i].transpose(0, 2, 1)
    rel_g = R_g[j] @ R_g[i].transpose(0, 2, 1)
    rre = angle_deg(rel_p @ rel_g.transpose(0, 2, 1))
    dir_p = np.einsum("nab,nb->na", R_p[i], C_p[j] - C_p[i])  # direction to camera j, in camera i's frame
    dir_g = np.einsum("nab,nb->na", R_g[i], C_g[j] - C_g[i])
    cos = (dir_p * dir_g).sum(1) / (np.linalg.norm(dir_p, axis=1) * np.linalg.norm(dir_g, axis=1) + 1e-12)
    rte = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    err = np.maximum(rre, rte)
    auc30 = float(np.mean([(err < t).mean() for t in range(1, 31)]))

    fit = umeyama(C_p, C_g)
    ate = float(np.sqrt(((fit.apply(C_p) - C_g) ** 2).sum(1).mean()))
    # Aligned camera-to-world rotation R @ R_p^T against the true R_g^T
    rot = angle_deg(R_g @ fit.rotation @ R_p.transpose(0, 2, 1))

    return {
        "views": len(pred.names),
        "auc30": auc30,
        "rre_deg_median": float(np.median(rre)),
        "rte_deg_median": float(np.median(rte)),
        "ate_cm": 100 * ate,
        "rot_deg_median": float(np.median(rot)),
        "scale": float(fit.scale),
    }
