# Adapted from DVLT (https://github.com/nv-tlabs/dvlt), Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
# Modified for Solv3R: only the conversions used at inference.
#
# quat_to_mat is adapted from PyTorch3D (https://github.com/facebookresearch/pytorch3d,
# pytorch3d/transforms/rotation_conversions.py), Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.,
# BSD 3-Clause License (below); DVLT switched it to scalar-last (xyzw) quaternions.
#
# The ray-to-camera fitting (homography RANSAC, QL decomposition, rays_to_pose) is adapted from Depth-Anything-3
# (https://github.com/ByteDance-Seed/Depth-Anything-3, utils/ray_utils.py), Copyright (c) 2025 ByteDance Ltd. and/or its
# affiliates, Apache-2.0.
#
# BSD 3-Clause License
#
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Cameras from the network outputs: pose encodings (camera head) and dense rays (ray head)."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# Pose encoding: [Tx, Ty, Tz, qx, qy, qz, qw, fov_h, fov_w], principal point at the image centre
# =============================================================================


def quat_to_mat(quaternions: Tensor) -> Tensor:
    """Quaternions (..., 4), scalar-last (xyzw) -> rotation matrices (..., 3, 3)."""
    i, j, k, r = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def to4x4(pose: Tensor) -> Tensor:
    """(..., 3, 4) -> (..., 4, 4) with the homogeneous row [0, 0, 0, 1]."""
    assert pose.shape[-2:] == (3, 4), f"Expected 3x4 pose, got {pose.shape}"
    batch_shape = pose.shape[:-2]
    constants = torch.tensor([0.0, 0.0, 0.0, 1.0], device=pose.device, dtype=pose.dtype)
    constants = constants.view(*([1] * len(batch_shape)), 1, 4).expand(*batch_shape, 1, 4)
    return torch.cat([pose, constants], dim=-2)


_FOV_EPS = 1e-3


def pose_enc_to_extri_intri(pose_encoding: Tensor, images_hw=None):
    """Pose encodings (..., 9) -> extrinsics (..., 3, 4) and, given images_hw = (H, W), intrinsics (..., 3, 3) in pixels.
    Quaternions are normalized; fields of view are clamped to (eps, pi - eps). Runs in float32 (autocast disabled)."""
    with torch.amp.autocast(device_type="cuda", enabled=False):
        translation = pose_encoding[..., :3]
        quat_xyzw = F.normalize(pose_encoding[..., 3:7], dim=-1)
        rotation = quat_to_mat(quat_xyzw)
        extrinsics = torch.cat([rotation, translation.unsqueeze(-1)], dim=-1)
        if images_hw is None or pose_encoding.shape[-1] < 9:
            return extrinsics, None

        fov_h = pose_encoding[..., 7].clamp(_FOV_EPS, float(torch.pi) - _FOV_EPS)
        fov_w = pose_encoding[..., 8].clamp(_FOV_EPS, float(torch.pi) - _FOV_EPS)
        height, width = images_hw
        fy = (0.5 * height) / torch.tan(0.5 * fov_h)
        fx = (0.5 * width) / torch.tan(0.5 * fov_w)
        cx = torch.full_like(fx, 0.5 * width)
        cy = torch.full_like(fy, 0.5 * height)
        zero, one = torch.zeros_like(fx), torch.ones_like(fx)
        row0 = torch.stack([fx, zero, cx], dim=-1)
        row1 = torch.stack([zero, fy, cy], dim=-1)
        row2 = torch.stack([zero, zero, one], dim=-1)
        return extrinsics, torch.stack([row0, row1, row2], dim=-2)


# =============================================================================
# Rays -> cameras: homography RANSAC between identity-camera rays and predicted rays (from Depth-Anything-3)
# =============================================================================


def _ql_decomposition(A: Tensor) -> Tuple[Tensor, Tensor]:
    """QL decomposition via QR decomposition with permutation."""
    P = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]], device=A.device, dtype=A.dtype)
    A_tilde = torch.matmul(A, P)
    Q_tilde, R_tilde = torch.linalg.qr(A_tilde)
    Q = torch.matmul(Q_tilde, P)
    L = torch.matmul(torch.matmul(P, R_tilde), P)
    d = torch.diag(L)
    Q[:, 0] *= torch.sign(d[0])
    Q[:, 1] *= torch.sign(d[1])
    Q[:, 2] *= torch.sign(d[2])
    L[0] *= torch.sign(d[0])
    L[1] *= torch.sign(d[1])
    L[2] *= torch.sign(d[2])
    return Q, L


def _find_homography_weighted_batch(src_pts_batch: Tensor, dst_pts_batch: Tensor, weights_batch: Tensor) -> Tensor:
    """
    Batch weighted least squares homography estimation.

    Args:
        src_pts_batch: (B, K, 2) source points
        dst_pts_batch: (B, K, 2) destination points
        weights_batch: (B, K) confidence weights

    Returns:
        H: (B, 3, 3) homography matrices
    """
    B, K, _ = src_pts_batch.shape
    w = weights_batch.sqrt().unsqueeze(2)  # (B, K, 1)
    x = src_pts_batch[:, :, 0:1]
    y = src_pts_batch[:, :, 1:2]
    u = dst_pts_batch[:, :, 0:1]
    v = dst_pts_batch[:, :, 1:2]
    zeros = torch.zeros_like(x)

    A1 = torch.cat([-x * w, -y * w, -w, zeros, zeros, zeros, x * u * w, y * u * w, u * w], dim=2)
    A2 = torch.cat([zeros, zeros, zeros, -x * w, -y * w, -w, x * v * w, y * v * w, v * w], dim=2)
    A = torch.cat([A1, A2], dim=1)  # (B, 2K, 9)

    _, _, Vh = torch.linalg.svd(A)
    H = Vh[:, -1].reshape(B, 3, 3)
    H = H / H[:, 2:3, 2:3]
    return H


def _find_homography_weighted(src_pts: Tensor, dst_pts: Tensor, weights: Tensor) -> Tensor:
    """Single-sample weighted least squares homography estimation."""
    N = src_pts.shape[0]
    if N < 4:
        return torch.eye(3, dtype=src_pts.dtype, device=src_pts.device)

    w = weights.sqrt().unsqueeze(1)  # (N, 1)
    x = src_pts[:, 0:1]
    y = src_pts[:, 1:2]
    u = dst_pts[:, 0:1]
    v = dst_pts[:, 1:2]
    zeros = torch.zeros_like(x)

    A1 = torch.cat([-x * w, -y * w, -w, zeros, zeros, zeros, x * u * w, y * u * w, u * w], dim=1)
    A2 = torch.cat([zeros, zeros, zeros, -x * w, -y * w, -w, x * v * w, y * v * w, v * w], dim=1)
    A = torch.cat([A1, A2], dim=0)  # (2N, 9)

    _, _, Vh = torch.linalg.svd(A)
    H = Vh[-1].reshape(3, 3)
    H = H / H[-1, -1]
    return H


def _ransac_homography_batch(
    src_pts: Tensor,
    dst_pts: Tensor,
    weights: Tensor,
    n_sample: int,
    n_iter: int = 100,
    reproj_threshold: float = 0.2,
    num_sample_for_ransac: int = 8,
    max_inlier_num: int = 8000,
) -> Tensor:
    """
    Batch RANSAC homography estimation.

    Args:
        src_pts: (B, N, 2) source points
        dst_pts: (B, N, 2) destination points
        weights: (B, N) confidence weights
        n_sample: number of top-weighted points to sample from
        n_iter: RANSAC iterations
        reproj_threshold: inlier threshold
        num_sample_for_ransac: points per RANSAC sample
        max_inlier_num: max inliers for final fit

    Returns:
        H: (B, 3, 3) homography matrices
    """
    B, N, _ = src_pts.shape
    device = src_pts.device

    sorted_idx = torch.argsort(weights, descending=True, dim=1)  # (B, N)
    candidate_idx = sorted_idx[:, :n_sample]  # (B, n_sample)

    # a fixed local generator keeps the RANSAC samples, and so the cameras, deterministic
    _ransac_gen = torch.Generator(device=device).manual_seed(42)
    rand_sample_iters_idx = torch.stack(
        [torch.randperm(n_sample, device=device, generator=_ransac_gen)[:num_sample_for_ransac] for _ in range(n_iter)],
        dim=0,
    )  # (n_iter, num_sample_for_ransac)

    rand_idx = candidate_idx[:, rand_sample_iters_idx]  # (B, n_iter, num_sample_for_ransac)

    b_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, n_iter, num_sample_for_ransac)
    src_pts_batch = src_pts[b_idx, rand_idx]  # (B, n_iter, num_sample_for_ransac, 2)
    dst_pts_batch = dst_pts[b_idx, rand_idx]
    weights_batch = weights[b_idx, rand_idx]

    cB, cN = src_pts_batch.shape[:2]
    H_batch = _find_homography_weighted_batch(
        src_pts_batch.flatten(0, 1), dst_pts_batch.flatten(0, 1), weights_batch.flatten(0, 1)
    )
    H_batch = H_batch.unflatten(0, (cB, cN))  # (B, n_iter, 3, 3)

    src_homo = torch.cat([src_pts, torch.ones(B, N, 1, dtype=src_pts.dtype, device=device)], dim=2)
    src_homo_expand = src_homo.unsqueeze(1).expand(B, n_iter, N, 3)
    dst_pts_expand = dst_pts.unsqueeze(1).expand(B, n_iter, N, 2)
    weights_expand = weights.unsqueeze(1).expand(B, n_iter, N)

    H_batch_flat = H_batch.reshape(-1, 3, 3)
    src_homo_flat = src_homo_expand.reshape(-1, N, 3)
    proj = torch.bmm(src_homo_flat, H_batch_flat.transpose(1, 2))
    proj_xy = proj[:, :, :2] / proj[:, :, 2:3].clamp(min=1e-8)
    proj_xy = proj_xy.reshape(B, n_iter, N, 2)

    error = ((proj_xy - dst_pts_expand) ** 2).sum(dim=3).sqrt()
    inlier_mask = error < reproj_threshold
    total_score = (inlier_mask * weights_expand).sum(dim=2)

    best_idx = torch.argmax(total_score, dim=1)
    best_inlier_mask = inlier_mask[torch.arange(B, device=device), best_idx]

    H_inlier_list = []
    for b in range(B):
        mask = best_inlier_mask[b]
        inlier_src = src_pts[b][mask]
        inlier_dst = dst_pts[b][mask]
        inlier_weights = weights[b][mask]

        if inlier_src.shape[0] < 4:
            H_inlier_list.append(torch.eye(3, device=device, dtype=src_pts.dtype))
            continue

        if inlier_src.shape[0] > max_inlier_num:
            sorted_idx = torch.argsort(inlier_weights, descending=True)
            keep_len = max(int(len(sorted_idx) * 0.95), max_inlier_num)
            sorted_idx = sorted_idx[:keep_len]
            perm = torch.randperm(len(sorted_idx), device=device)[:max_inlier_num]
            sorted_idx = sorted_idx[perm]
            inlier_src = inlier_src[sorted_idx]
            inlier_dst = inlier_dst[sorted_idx]
            inlier_weights = inlier_weights[sorted_idx]

        H_inlier = _find_homography_weighted(inlier_src, inlier_dst, inlier_weights)
        H_inlier_list.append(H_inlier)

    return torch.stack(H_inlier_list, dim=0)


def _compute_rotation_intrinsics_batch(
    rays_origin: Tensor,
    rays_target: Tensor,
    weights: Tensor,
    z_threshold: float = 1e-4,
    reproj_threshold: float = 0.2,
    n_iter: int = 100,
    num_sample_for_ransac: int = 8,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute optimal rotation and intrinsics from ray correspondences.

    Args:
        rays_origin: (B, N, 3) identity camera rays
        rays_target: (B, N, 3) predicted ray directions
        weights: (B, N) confidence weights
        z_threshold: threshold for valid z values
        reproj_threshold: RANSAC inlier threshold

    Returns:
        R: (B, 3, 3) rotation matrices
        focal_lengths: (B, 2) focal lengths (fx, fy)
        principal_points: (B, 2) principal points (cx, cy)
    """
    device = rays_origin.device
    B, N, _ = rays_origin.shape

    z_mask = (torch.abs(rays_target[:, :, 2]) > z_threshold) & (torch.abs(rays_origin[:, :, 2]) > z_threshold)

    rays_origin = rays_origin.clone()
    rays_target = rays_target.clone()

    for i in range(B):
        mask = z_mask[i]
        rays_origin[i, mask, 0] /= rays_origin[i, mask, 2]
        rays_origin[i, mask, 1] /= rays_origin[i, mask, 2]
        rays_target[i, mask, 0] /= rays_target[i, mask, 2]
        rays_target[i, mask, 1] /= rays_target[i, mask, 2]

    rays_origin_2d = rays_origin[:, :, :2]
    rays_target_2d = rays_target[:, :, :2]

    weights = weights.clone()
    weights[~z_mask] = 0

    sample_ratio = 0.3
    n_sample = max(num_sample_for_ransac, int(N * sample_ratio))

    max_chunk_size = 2
    A_list = []
    for i in range(0, B, max_chunk_size):
        chunk_end = min(i + max_chunk_size, B)
        A = _ransac_homography_batch(
            rays_origin_2d[i:chunk_end],
            rays_target_2d[i:chunk_end],
            weights[i:chunk_end],
            n_sample=n_sample,
            n_iter=n_iter,
            reproj_threshold=reproj_threshold,
            num_sample_for_ransac=num_sample_for_ransac,
        )
        det_mask = torch.linalg.det(A) < 0
        A[det_mask] = -A[det_mask]
        A_list.append(A.to(device))

    A = torch.cat(A_list, dim=0)

    # Decompose homography into rotation and intrinsics
    R_list, f_list, pp_list = [], [], []
    for i in range(A.shape[0]):
        R, L = _ql_decomposition(A[i])
        L = L / L[2, 2]
        f = torch.stack([L[0, 0], L[1, 1]])
        pp = torch.stack([L[2, 0], L[2, 1]])
        R_list.append(R)
        f_list.append(f)
        pp_list.append(pp)

    R = torch.stack(R_list)
    f = torch.stack(f_list)
    pp = torch.stack(pp_list)

    return R, f, pp


def _create_identity_ray_grid_normalized(
    num_patches_y: int,
    num_patches_x: int,
    H: int,
    W: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """
    Create identity camera rays in normalized coordinate system.

    The grid positions must match the positions sampled by F.interpolate(align_corners=True)
    when downsampling from (H, W) to (num_patches_y, num_patches_x).

    Uses a 2x2 normalized image space with identity intrinsics fx=fy=1, cx=cy=1.

    Args:
        num_patches_y, num_patches_x: patch grid dimensions
        H, W: original image dimensions
        device, dtype: tensor device and dtype

    Returns:
        rays: (num_patches_y, num_patches_x, 3) identity ray directions in normalized space
    """
    # F.interpolate with align_corners=True samples at pixel positions:
    # x_pixel = i * (W - 1) / (num_patches_x - 1) for i in [0, num_patches_x-1]
    # y_pixel = j * (H - 1) / (num_patches_y - 1) for j in [0, num_patches_y-1]
    #
    # But our rays are generated at pixel CENTERS: ray at index (i, j) is for pixel center (i+0.5, j+0.5)
    # So the interpolated ray at patch position (m, n) corresponds to pixel center position:
    # x = m * (W - 1) / (num_patches_x - 1) + 0.5
    # y = n * (H - 1) / (num_patches_y - 1) + 0.5
    #
    # Convert to normalized [0, 2] coordinates: x_norm = x * 2 / W, y_norm = y * 2 / H

    if num_patches_x > 1:
        x_pixel = torch.linspace(0, W - 1, num_patches_x, dtype=dtype, device=device)
    else:
        x_pixel = torch.tensor([(W - 1) / 2], dtype=dtype, device=device)

    if num_patches_y > 1:
        y_pixel = torch.linspace(0, H - 1, num_patches_y, dtype=dtype, device=device)
    else:
        y_pixel = torch.tensor([(H - 1) / 2], dtype=dtype, device=device)

    # Add 0.5 to convert from pixel index to pixel center (matching compute_world_rays)
    x_pixel = x_pixel + 0.5
    y_pixel = y_pixel + 0.5

    x_norm = x_pixel * 2.0 / W
    y_norm = y_pixel * 2.0 / H

    y_grid, x_grid = torch.meshgrid(y_norm, x_norm, indexing="ij")

    # For identity intrinsics K = [[1, 0, 1], [0, 1, 1], [0, 0, 1]] in normalized 2x2 space:
    # Ray direction at normalized coord (x, y) is K^-1 @ [x, y, 1] = [x-1, y-1, 1]
    ray_x = x_grid - 1.0  # cx = 1 in normalized space
    ray_y = y_grid - 1.0  # cy = 1 in normalized space
    ray_z = torch.ones_like(ray_x)

    rays = torch.stack([ray_x, ray_y, ray_z], dim=-1)
    return rays


def _camray_to_caminfo(
    camray: Tensor,
    confidence: Tensor,
    H: int,
    W: int,
    reproj_threshold: float = 0.2,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Convert camera rays to camera parameters.

    Uses DA3's normalized coordinate system (2x2 image, fx=fy=1, cx=cy=1) for
    homography fitting, then converts results to pixel coordinates.

    Args:
        camray: (B, S, num_patches_y, num_patches_x, 6) world-space rays
        confidence: (B, S, num_patches_y, num_patches_x) confidence weights
        H, W: original image dimensions
        reproj_threshold: RANSAC reprojection threshold

    Returns:
        R: (B, S, 3, 3) rotation matrices
        T: (B, S, 3) translation vectors
        focal_lengths: (B, S, 2) focal lengths (fx, fy) in pixels
        principal_points: (B, S, 2) principal points (cx, cy) in pixels
    """
    B, S, num_patches_y, num_patches_x, _ = camray.shape
    device = camray.device
    dtype = camray.dtype

    # Grid positions must match F.interpolate(align_corners=True) sampling positions
    I_cam_rays = _create_identity_ray_grid_normalized(num_patches_y, num_patches_x, H, W, device, dtype)

    I_cam_rays = I_cam_rays.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1, -1)

    camray_flat = camray.flatten(0, 1).flatten(1, 2)  # (B*S, N, 6)
    I_cam_flat = I_cam_rays.flatten(0, 1).flatten(1, 2)  # (B*S, N, 3)
    conf_flat = confidence.flatten(0, 1).flatten(1, 2)  # (B*S, N)

    camray_flat = camray_flat.detach()
    I_cam_flat = I_cam_flat.detach()
    conf_flat = conf_flat.detach()

    # Normalize ray directions for homography fitting
    # The input rays may be unnormalized (DA3 convention), but homography fitting
    # works with normalized directions (dividing by z gives same 2D coords either way,
    # but normalized is more numerically stable)
    ray_dirs = camray_flat[:, :, :3]
    ray_dirs_normalized = F.normalize(ray_dirs, dim=-1)

    # Compute rotation and intrinsics via homography fitting
    # The homography maps normalized identity coords to predicted ray coords (both after /z)
    R, focal_lengths_norm, principal_points_norm = _compute_rotation_intrinsics_batch(
        I_cam_flat,
        ray_dirs_normalized,
        conf_flat,
        reproj_threshold=reproj_threshold,
    )

    T = (camray_flat[:, :, 3:] * conf_flat.unsqueeze(-1)).sum(dim=1) / (conf_flat.sum(dim=-1, keepdim=True) + 1e-8)

    R = R.reshape(B, S, 3, 3)
    T = T.reshape(B, S, 3)
    focal_lengths_norm = focal_lengths_norm.reshape(B, S, 2)
    principal_points_norm = principal_points_norm.reshape(B, S, 2)

    # Convert from normalized (2x2 image) to pixel coordinates
    # DA3 returns: 1/focal_lengths_norm (inverted), principal_points_norm + 1.0 (shifted)
    # These are in normalized [0, 2] coordinate space
    #
    # For normalized image [0, 2] with intrinsics K_norm = [[f_n, 0, c_n], [0, f_n, c_n], [0, 0, 1]]:
    #   f_pixel = f_norm * (pixel_dim / 2)
    #   c_pixel = c_norm * (pixel_dim / 2)

    # Invert to get normalized focal lengths (DA3 returns 1/f)
    # Clamp both min and max to avoid extreme values from ill-conditioned homographies
    # In normalized [0, 2] space: f_norm=1 corresponds to 45° FoV
    # - f_norm=0.2 -> very wide angle (~157° FoV)
    # - f_norm=10 -> telephoto (~11° FoV)
    focal_lengths_norm_inv = 1.0 / focal_lengths_norm.clamp(min=0.1, max=10.0)
    # Shift principal points (DA3 adds 1.0 to shift from [-1, 1] to [0, 2] range)
    principal_points_norm_shifted = principal_points_norm + 1.0

    focal_lengths = torch.stack(
        [
            focal_lengths_norm_inv[:, :, 0] * W / 2,
            focal_lengths_norm_inv[:, :, 1] * H / 2,
        ],
        dim=-1,
    )

    principal_points = torch.stack(
        [
            principal_points_norm_shifted[:, :, 0] * W / 2,
            principal_points_norm_shifted[:, :, 1] * H / 2,
        ],
        dim=-1,
    )

    return R, T, focal_lengths, principal_points


# =============================================================================
# Main ray-to-pose interface
# =============================================================================


def rays_to_pose(
    rays: Tensor,
    rays_conf: Tensor,
    H: int,
    W: int,
    patch_size: int = 16,
) -> Tuple[Tensor, Tensor]:
    """
    Convert world-space rays to camera extrinsics and intrinsics.

    Uses RANSAC-based homography fitting to estimate camera parameters
    from predicted world-space rays and their confidences.

    Args:
        rays: World-space rays (B, S, H, W, 6) - first 3 are direction, last 3 are origin
        rays_conf: Ray confidence (B, S, H, W)
        H, W: Image height and width
        patch_size: Patch size used for ray prediction

    Returns:
        extrinsics_c2w: (B, S, 4, 4) camera-to-world matrices
        intrinsics: (B, S, 3, 3) intrinsic matrices
    """
    B, S = rays.shape[:2]
    device = rays.device
    dtype = rays.dtype

    ph, pw = H // patch_size, W // patch_size
    rays_patch = (
        F.interpolate(
            rays.view(B * S, H, W, 6).permute(0, 3, 1, 2),
            size=(ph, pw),
            mode="bilinear",
            align_corners=True,
        )
        .permute(0, 2, 3, 1)
        .view(B, S, ph, pw, 6)
    )

    conf_patch = F.interpolate(
        rays_conf.view(B * S, 1, H, W),
        size=(ph, pw),
        mode="bilinear",
        align_corners=True,
    ).view(B, S, ph, pw)

    with torch.no_grad():
        R, T, focal_lengths, principal_points = _camray_to_caminfo(
            rays_patch, confidence=conf_patch, H=H, W=W, reproj_threshold=0.2
        )

    extrinsics_c2w = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1).clone()
    extrinsics_c2w[:, :, :3, :3] = R
    extrinsics_c2w[:, :, :3, 3] = T

    # Build intrinsics (focal_lengths and principal_points are already in pixel units)
    intrinsics = torch.zeros(B, S, 3, 3, device=device, dtype=dtype)
    intrinsics[:, :, 0, 0] = focal_lengths[:, :, 0]  # fx
    intrinsics[:, :, 1, 1] = focal_lengths[:, :, 1]  # fy
    intrinsics[:, :, 0, 2] = principal_points[:, :, 0]  # cx
    intrinsics[:, :, 1, 2] = principal_points[:, :, 1]  # cy
    intrinsics[:, :, 2, 2] = 1.0

    return extrinsics_c2w, intrinsics
