"""Build the pinned GenRecon CUDA extensions in the active GPU environment.

Requires torch 2.9.1+cu130, the CPU dependencies, and a CUDA 13 compiler.
Temporary source/build directories are removed on exit. No model is downloaded.
"""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[3]


def main():
    prefix = Path(sys.prefix)
    if not (prefix / "bin/nvcc").exists():
        raise SystemExit("Install the CUDA 13 compiler in this environment first.")
    environment = os.environ.copy()
    environment.update({"CUDA_HOME": str(prefix), "PATH": str(prefix / "bin") + ":" + environment["PATH"],
                        "TORCH_CUDA_ARCH_LIST": "12.0", "MAX_JOBS": "4"})
    headers = str(prefix / "targets/x86_64-linux/include")
    environment["CPATH"] = headers + (":" + environment["CPATH"] if environment.get("CPATH") else "")
    pip = [sys.executable, "-m", "pip", "install", "--no-cache-dir"]
    flash = ("https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/"
             "flash_attn-2.8.3%2Bcu13torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl")
    subprocess.run(pip + [flash], env=environment, check=True)
    dependencies = [
        "nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae",
        "nvdiffrec_render @ git+https://github.com/JeffreyXiang/nvdiffrec.git@b296927cc7fd01c2ac1087c8065c4d7248f72da4",
        "cumesh @ git+https://github.com/JeffreyXiang/CuMesh.git@12289e1062f0603f2f0d0771b02e1395d247f26f",
        "flex_gemm @ git+https://github.com/JeffreyXiang/FlexGEMM.git@6dd94a859c26ee8246888502eada3dd8ad85532e",
    ]
    # Install separately so a failed build does not discard completed packages.
    for dependency in dependencies:
        subprocess.run(pip + ["--no-build-isolation", dependency], env=environment, check=True)
    with tempfile.TemporaryDirectory(prefix="genrecon-ovoxel-") as build:
        source = Path(build) / "o-voxel"
        shutil.copytree(ROOT / "papers/GenRecon/o-voxel", source)
        subprocess.run(pip + ["--no-build-isolation", str(source)], env=environment, check=True)
    subprocess.run([sys.executable, "-c", "import o_voxel, cumesh, flex_gemm, nvdiffrast.torch, "
                    "nvdiffrec_render, flash_attn; print('GenRecon extension imports passed')"],
                   env=environment, check=True)


if __name__ == "__main__":
    main()
