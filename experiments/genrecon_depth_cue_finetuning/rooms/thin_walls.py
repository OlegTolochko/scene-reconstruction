"""Sparse-structure latents of SAGE chunks with thin walls: only the side of each wall that faces the room.

build_rooms.py makes each wall a 0.1 m box, 2-3 cells thick in the 64^3 occupancy of a chunk; a ScanNet++
scan only has the wall's inner surface, 1 cell thick. Per chunk, this
  1. takes the occupancy get_slats.py encoded: the 64^3 coords of the chunk's shape latent,
  2. samples the room mesh of export_room_mesh.py densely in the chunk frame (crops/<room>.json), each sample
     labelled by its triangle's `keep`,
  3. drops the occupied cells that only dropped wall triangles reach (outer face, end caps, top); a cell that
     also holds a kept triangle (the inner face, furniture, a door) stays,
  4. encodes the thinned occupancy with the same sparse-structure encoder as get_slats.py.
A wrong frame shows up in step 2: the cells the samples reach must match the occupancy (IoU in thin_stats.json;
the script stops below --min-iou). The chunk frame is tried as is and with get_slats.py's +-90 deg axis fixes.

Writes <root>/ss_latents/<ss_name>_thinwalls/<chunk>.npz (z, as get_slats.py) and metadata.csv
(sha256, ss_latent_encoded), so train.py uses it via the "ss_latent" of --data_dir, and thin_stats.json.

From papers/GenRecon, in GenRecon's env, on a GPU:
    python ../../experiments/genrecon_depth_cue_finetuning/rooms/thin_walls.py --root ROOT
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.getcwd())  # papers/GenRecon
import genrecon.models as models  # noqa: E402

SS_ENCODER = "microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16"
SS_DECODER = "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16"  # config_ss.json's pretrained_ss_dec
SS_NAME = "ss_enc_conv3d_16l8_fp16_64"
SHAPE_NAME = "shape_enc_next_dc_f16c32_fp16_1024"
R = 64


def rot_x(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


AXES = {"none": np.eye(3), "x90": rot_x(90), "x-90": rot_x(-90)}


def sample(v: np.ndarray, f: np.ndarray, spacing: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """Points on the triangles at about `spacing` apart; returns points and their triangle index."""
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    n = np.ceil(2 * area / spacing**2).astype(np.int64) + 3
    tri = np.repeat(np.arange(len(f)), n)
    r1, r2 = rng.random(len(tri)), rng.random(len(tri))
    flip = r1 + r2 > 1
    r1, r2 = np.where(flip, 1 - r1, r1), np.where(flip, 1 - r2, r2)
    pts = a[tri] + r1[:, None] * (b - a)[tri] + r2[:, None] * (c - a)[tri]
    return pts, tri


def cells(pts: np.ndarray) -> np.ndarray:
    """Flat 64^3 indices of the points inside the unit cube [-0.5, 0.5]^3."""
    idx = np.floor((pts + 0.5) * R).astype(np.int64)
    inside = ((idx >= 0) & (idx < R)).all(1)
    idx = idx[inside]
    return np.unique(idx[:, 0] * R * R + idx[:, 1] * R + idx[:, 2])


def chunk_mesh(v_room, f, M):
    """The mesh in the chunk frame, only triangles whose bounding box meets the (slightly larger) unit cube."""
    v = v_room @ M[:3, :3].T + M[:3, 3]
    lo, hi = v[f].min(1), v[f].max(1)
    near = ((hi > -0.52) & (lo < 0.52)).all(1)
    return v, np.flatnonzero(near)


def decode(decoder, z: torch.Tensor) -> np.ndarray:
    """Flat 64^3 indices of the cells a sparse-structure latent [8, 16, 16, 16] decodes to (logit > 0)."""
    with torch.no_grad():
        logits = decoder(z[None].float())[0, 0]
    idx = torch.nonzero(logits > 0).cpu().numpy()
    return np.unique(idx[:, 0] * R * R + idx[:, 1] * R + idx[:, 2])


def set_iou(a: np.ndarray, b: np.ndarray) -> float:
    return len(np.intersect1d(a, b)) / max(len(np.union1d(a, b)), 1)


def voxel_ply(path: Path, flat: np.ndarray, colors: np.ndarray):
    """The cells as cubes of side 1/64 in the unit cube, one colour per cell."""
    import trimesh

    ijk = np.stack([flat // (R * R), flat // R % R, flat % R], 1)
    box = trimesh.creation.box(extents=(1.0 / R,) * 3)
    centres = (ijk + 0.5) / R - 0.5
    v = (box.vertices[None] + centres[:, None]).reshape(-1, 3)
    f = (box.faces[None] + 8 * np.arange(len(ijk))[:, None, None]).reshape(-1, 3)
    mesh = trimesh.Trimesh(v, f, process=False)
    mesh.visual.face_colors = np.repeat(colors, 12, axis=0)
    mesh.export(path)


def visualise(prefix: Path, occ, dropped, dec_orig, dec_thin, up: int):
    """<prefix>_before.ply (get_slats.py's occupancy, the dropped cells red), <prefix>_after.ply (the thinned
    latent decoded) and <prefix>_slices.png: three horizontal slices, before | after."""
    from PIL import Image

    prefix.parent.mkdir(parents=True, exist_ok=True)
    grey, red = np.array([180, 180, 180, 255]), np.array([220, 40, 40, 255])
    is_dropped = np.isin(occ, dropped)
    voxel_ply(Path(f"{prefix}_before.ply"), occ, np.where(is_dropped[:, None], red, grey))
    voxel_ply(Path(f"{prefix}_after.ply"), dec_thin, np.tile(grey, (len(dec_thin), 1)))

    def volume(flat):
        g = np.zeros(R**3, dtype=bool)
        g[flat] = True
        return np.moveaxis(g.reshape(R, R, R), up, 0)  # [up, a, b]

    before, gone, after = volume(occ), volume(dropped), volume(dec_thin)
    levels = np.flatnonzero(before.any((1, 2)))
    rows = []
    for q in (0.3, 0.5, 0.7):
        k = levels[int(q * (len(levels) - 1))]
        b = np.full((R, R, 3), 255, np.uint8)
        b[before[k]] = (90, 90, 90)
        b[gone[k]] = (220, 40, 40)
        a = np.full((R, R, 3), 255, np.uint8)
        a[after[k]] = (90, 90, 90)
        rows.append(np.concatenate([b, np.full((R, 2, 3), 0, np.uint8), a], 1))
        rows.append(np.zeros((2, 2 * R + 2, 3), np.uint8))
    Image.fromarray(np.concatenate(rows[:-1])).resize(((2 * R + 2) * 6, (3 * R + 4) * 6), Image.NEAREST).save(
        f"{prefix}_slices.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--min-iou", type=float, default=0.8, help="sampled cells vs the occupancy, below = wrong frame")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--vis", type=Path, help="write before/after .ply and slice .png per chunk here")
    ap.add_argument("--vis-rooms", type=int, default=3, help="for the first N rooms")
    args = ap.parse_args()

    out = args.root / "ss_latents" / f"{SS_NAME}_thinwalls"
    out.mkdir(parents=True, exist_ok=True)
    stats_path = out / "thin_stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    encoder = models.from_pretrained(SS_ENCODER).eval().cuda()
    decoder = models.from_pretrained(SS_DECODER).eval().cuda()
    rng = np.random.default_rng(0)
    mesh_paths = sorted((args.root / "rooms_processed").glob("*_mesh.npz"))
    vis_rooms = {p.name.removesuffix("_mesh.npz") for p in mesh_paths[:args.vis_rooms]}

    for mesh_path in mesh_paths:
        room = mesh_path.name.removesuffix("_mesh.npz")
        mesh = np.load(mesh_path)
        if "keep" not in mesh:
            raise RuntimeError(f"{mesh_path} has no wall labels: export it with the layout JSON")
        crops = json.loads((args.root / "crops" / f"{room}.json").read_text())["chunks"]
        for c in crops:
            sha = f"{room}_{int(c['index']):04d}"
            dst = out / f"{sha}.npz"
            shape = args.root / "shape_latents" / SHAPE_NAME / f"{sha}.npz"
            if (dst.exists() and not args.force) or not shape.exists():
                continue
            coords = np.load(shape)["coords"].astype(np.int64)  # [N, 3] at 64^3, as get_slats.py's ss input
            occ = np.unique(coords[:, 0] * R * R + coords[:, 1] * R + coords[:, 2])

            v, near = chunk_mesh(mesh["vertices"].astype(np.float64), mesh["faces"], np.array(c["M_original_to_chunk"]))
            f = mesh["faces"][near]
            keep = mesh["keep"][near]
            best = None
            for name, A in AXES.items():
                pts, tri = sample(v @ A.T, f, 1.0 / (4 * R), rng)
                full = cells(pts)
                iou = len(np.intersect1d(full, occ)) / max(len(np.union1d(full, occ)), 1)
                if best is None or iou > best[1]:
                    best = (name, iou, pts, tri, full)
            name, iou, pts, tri, full = best
            if iou < args.min_iou:
                raise RuntimeError(f"{sha}: sampled mesh matches the occupancy with IoU {iou:.3f} at best ({name})")
            kept = cells(pts[keep[tri]])
            dropped = np.setdiff1d(np.intersect1d(occ, full), kept)  # occupied, reached only by dropped triangles
            thin = np.setdiff1d(occ, dropped)

            grid = torch.zeros(R**3, dtype=torch.float32, device="cuda")
            grid[torch.from_numpy(thin).cuda()] = 1.0
            with torch.no_grad():
                z = encoder(grid.reshape(1, 1, R, R, R), sample_posterior=False)
            np.savez(dst, z=z[0].cpu().numpy().astype(np.float32))
            # What training sees: both latents decoded again, against the occupancy they encode.
            dec_thin = decode(decoder, z[0])
            dec_orig = decode(decoder, torch.from_numpy(
                np.load(args.root / "ss_latents" / SS_NAME / f"{sha}.npz")["z"]).cuda())
            stats[sha] = {"axis": name, "iou_mesh_vs_occupancy": round(iou, 4), "cells": int(len(occ)),
                          "cells_thin": int(len(thin)), "dropped_share": round(len(dropped) / max(len(occ), 1), 4),
                          "iou_decoded_vs_occupancy": round(set_iou(dec_orig, occ), 4),
                          "iou_decoded_thin_vs_thin": round(set_iou(dec_thin, thin), 4)}
            print(json.dumps({"chunk": sha, **stats[sha]}), flush=True)
            if args.vis and room in vis_rooms:
                up = int(np.argmax(np.abs(AXES[name] @ np.array([0.0, 0.0, 1.0]))))
                visualise(args.vis / sha, occ, dropped, dec_orig, dec_thin, up)

    stats_path.write_text(json.dumps(stats, indent=2))
    with open(out / "metadata.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sha256", "ss_latent_encoded"])
        w.writerows([[p.stem, True] for p in sorted(out.glob("*.npz"))])


if __name__ == "__main__":
    main()
