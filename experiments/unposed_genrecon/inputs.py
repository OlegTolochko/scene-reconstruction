"""Input images for infer.py (VGGT-Omega) and infer_solv3r.py (Solv3R), so both models see the same images."""

import glob
import os
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def extract_frames(video_path: str, num_frames: int, out_dir: str) -> list[str]:
    """Save num_frames evenly spaced frames from the video and return their paths."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    paths = []
    for i, index in enumerate(np.linspace(0, total - 1, num_frames).astype(int)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            continue
        path = os.path.join(out_dir, f"{i:04d}.png")
        cv2.imwrite(path, frame)
        paths.append(path)
    cap.release()
    return paths


def pick_evenly(paths: list[str], num_images: int) -> list[str]:
    """Keep num_images evenly spaced paths (all of them if there are fewer), like GenRecon does."""
    if len(paths) <= num_images:
        return paths
    indices = np.linspace(0, len(paths) - 1, num_images).round().astype(int)
    return [paths[i] for i in indices]


def select_images(input_path: str, num_images: int, tmp_dir: str) -> list[str]:
    """num_images evenly spaced images of a folder, or frames of a video saved to tmp_dir."""
    if os.path.isdir(input_path):
        all_names = sorted(p for p in glob.glob(os.path.join(input_path, "*")) if p.lower().endswith(IMAGE_EXTS))
        image_names = pick_evenly(all_names, num_images)
    else:
        image_names = extract_frames(input_path, num_images, tmp_dir)
    if not image_names:
        raise SystemExit(f"No images found in {input_path}")
    return image_names


def scene_name(input_path: str) -> str:
    """Name of the outputs: the scene ID for ScanNet++ image folders, else the folder or video name."""
    src = Path(input_path).resolve()
    return src.parents[1].name if src.parent.name in ("iphone", "dslr") else src.stem
