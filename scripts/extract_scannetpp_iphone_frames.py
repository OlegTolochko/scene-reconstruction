"""Extract the iPhone frames used by a ScanNet++ scene's COLMAP model.

GenRecon's `--mode Scannet_iphone` reads `<scene>/iphone/rgb/frame_XXXXXX.jpg`
for every image in `<scene>/iphone/colmap/images.txt`, where XXXXXX is the
frame index in `<scene>/iphone/rgb.mkv`. ScanNet++ only ships the video, so this
decodes it and keeps those frames.

Does the same job as `extract_rgb_in_colmap` in the official toolkit
(https://github.com/scannetpp/scannetpp, iphone/prepare_iphone_data.py), but
needs nothing besides ffmpeg >= 5.1. The toolkit builds one `eq(n,i)` term per
frame and passes them with `-filter_script`.

Usage:
    uv run scripts/extract_scannetpp_iphone_frames.py data/scannetpp/data/<scene_id> [...]
"""

import argparse
import math
import shutil
import subprocess
import tempfile
from pathlib import Path


def colmap_frame_names(images_txt: Path) -> list[str]:
    """Image names from a COLMAP images.txt, sorted by frame index."""
    names = []
    with images_txt.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME. Some scenes prefix
            # NAME with "video/"; GenRecon only uses the basename.
            names.append(Path(line.split(maxsplit=9)[9]).name)
            # The next line lists the image's 2D points and may be empty.
            next(f, None)
    return sorted(names, key=frame_index)


def frame_index(name: str) -> int:
    return int(Path(name).stem.split("_")[-1])


def extract_scene(scene_dir: Path) -> None:
    iphone_dir = scene_dir / "iphone"
    video = iphone_dir / "rgb.mkv"
    out_dir = iphone_dir / "rgb"
    names = colmap_frame_names(iphone_dir / "colmap" / "images.txt")

    todo = [n for n in names if not (out_dir / n).exists()]
    print(f"{scene_dir.name}: {len(names)} frames in COLMAP, {len(todo)} to extract")
    if not todo:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    # Select the frames first, first + step, ... up to the last wanted one, where
    # step is the largest spacing that hits every wanted frame (10 for the usual
    # ScanNet++ sampling), then keep only the wanted ones.
    wanted = {frame_index(n): n for n in todo}
    indices = sorted(wanted)
    first = indices[0]
    step = math.gcd(*(i - first for i in indices)) or 1
    count = (indices[-1] - first) // step + 1
    with tempfile.TemporaryDirectory(dir=iphone_dir) as tmp:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(video),
             "-vf", f"select='gte(n\\,{first})*not(mod(n-{first}\\,{step}))'",
             "-fps_mode", "vfr", "-frames:v", str(count), "-q:v", "1", f"{tmp}/%06d.jpg"],
            check=True,
        )
        # ffmpeg numbers the selected frames 1, 2, ... in decode order, so the
        # k-th file is frame first + k * step.
        extracted = sorted(Path(tmp).glob("*.jpg"))
        if len(extracted) != count:
            raise RuntimeError(f"{scene_dir.name}: ffmpeg wrote {len(extracted)} frames, expected {count}")
        for k, src in enumerate(extracted):
            name = wanted.get(first + k * step)
            if name is not None:
                shutil.move(src, out_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene_dirs", nargs="+", type=Path, help="ScanNet++ scene folders, e.g. data/scannetpp/data/<scene_id>")
    args = parser.parse_args()
    for scene_dir in args.scene_dirs:
        extract_scene(scene_dir)


if __name__ == "__main__":
    main()
