"""Runs a script of GenRecon's data_toolkit_scenes with Blender from BLENDER_DIR instead of /tmp.

The toolkit installs Blender to /tmp (with sudo apt-get for its libraries) when it is missing there. /tmp is
small and per node on our cluster, so this sets the scripts' BLENDER_PATH to a shared installation and then
runs their main() unchanged.

From papers/GenRecon, in GenRecon's env:
    python ../../experiments/genrecon_depth_cue_finetuning/rooms/toolkit.py build_rooms --root ROOT [...]
"""

import importlib
import os
import sys
from pathlib import Path

BLENDER_DIR = os.environ.get("BLENDER_DIR", "/cluster/vc17/pr26rb06/blender")
BLENDER = f"{BLENDER_DIR}/blender-5.0.1-linux-x64/blender"

if __name__ == "__main__":
    name = sys.argv[1]
    toolkit = Path("data_toolkit_scenes").resolve()
    if not (toolkit / f"{name}.py").is_file():
        raise SystemExit(f"run from papers/GenRecon: {toolkit / name}.py not found")
    if not os.path.isfile(BLENDER):
        raise SystemExit(f"Blender not found: {BLENDER}")
    sys.path.insert(0, str(toolkit))
    sys.argv = [str(toolkit / f"{name}.py"), *sys.argv[2:]]
    module = importlib.import_module(name)
    if hasattr(module, "BLENDER_PATH"):
        module.BLENDER_INSTALLATION_PATH = BLENDER_DIR
        module.BLENDER_PATH = BLENDER
    module.main()
