"""Image loading with the preprocessing used in training and in the streaming evaluation (the offline benchmark uses the
loaders of DVLT)."""
import glob
import os
import atexit
import re
import shutil
import subprocess
import tempfile
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

LONG_SIDE = 504
PATCH = 14
EXTENSIONS = (".jpg", ".jpeg", ".png")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")


def model_size(height, width, long_side=LONG_SIDE):
    """Long side resized to 504, each side rounded to a multiple of the 14-pixel patch (no crop)."""
    s = long_side / max(height, width)
    return max(PATCH, round(height * s / PATCH) * PATCH), max(PATCH, round(width * s / PATCH) * PATCH)


def resize(images, size):
    """images (S, 3, H, W) in [0, 1] -> (S, 3, h, w)."""
    return F.interpolate(images.float(), size=size, mode="bilinear", align_corners=False, antialias=True).clamp(0, 1)


def load_images(paths, size=None):
    """Images as (S, 3, h, w) in [0, 1], upright as an image viewer shows them (EXIF orientation).  All images take the
    model size of the first one unless size is given."""
    out, stretched = [], []
    for p in paths:
        try:
            im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        except (OSError, ValueError) as e:
            raise ValueError(f"could not read the image {p}: {e}") from None
        im = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255
        if size is not None and model_size(*im.shape[-2:]) != tuple(size):
            stretched.append(os.path.basename(p))
        size = size or model_size(*im.shape[-2:])
        out.append(resize(im[None], size))
    if stretched:
        warnings.warn(f"{len(stretched)} images have another aspect ratio and are stretched to {tuple(size)}: "
                      f"{', '.join(stretched[:3])}{', ...' if len(stretched) > 3 else ''}")
    return torch.cat(out)


def _file_order(path):
    """File-name order with the numbers in a name compared by value (frame_2 before frame_10)."""
    name = os.path.basename(path)
    return [int(t) if i % 2 else t for i, t in enumerate(re.split(r"(\d+)", name))], name


def _ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError("reading a video needs ffmpeg: install it, or `pip install imageio-ffmpeg`") from None


def video_frames(video, out_dir=None):
    """All frames of a video as .jpg files in a folder; returns the folder. Without out_dir the frames go to a temporary
    folder that is removed when the program exits."""
    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="solv3r_frames_")
        atexit.register(shutil.rmtree, out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    try:
        subprocess.run([_ffmpeg(), "-loglevel", "error", "-i", video, "-vsync", "0", "-q:v", "2",
                        os.path.join(out_dir, "%06d.jpg")], check=True)
    except subprocess.CalledProcessError:
        raise ValueError(f"ffmpeg could not read the video {video}") from None
    return out_dir


def list_images(folder, stride=1, max_images=None):
    """The .jpg / .jpeg / .png files of a folder in file-name order, every stride-th, at most max_images.  A video file
    (.mp4, .mov, ...) is read as its frames."""
    if os.path.isfile(folder) and folder.lower().endswith(VIDEO_EXTENSIONS):
        folder = video_frames(folder)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"no such folder or video file: {folder}")
    paths = sorted((p for p in glob.glob(os.path.join(glob.escape(folder), "*"))
                    if p.lower().endswith(EXTENSIONS) and os.path.isfile(p)), key=_file_order)
    if not paths:
        raise ValueError(f"no {', '.join(EXTENSIONS)} images in {folder}")
    return paths[::stride][:max_images]
