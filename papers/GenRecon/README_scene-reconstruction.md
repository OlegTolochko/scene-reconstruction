# scene-reconstruction Notes for GenRecon

This directory is a copy of the official GenRecon codebase.

## Upstream

```text
Repository: https://github.com/kasothaphie/GenRecon
Imported commit: eaf1468118d20469d17079a4a19737297d2ef87b
Imported branch: main
Import date: 2026-10-19
Paper: https://arxiv.org/abs/2605.23888
Project page: https://kasothaphie.github.io/GenRecon/
Weights: https://kaldir.vc.cit.tum.de/genrecon/ (sparse_structure.pt, shape_slat.pt, texture_slat.pt)
License: MIT (see LICENSE)
```

## Local Changes

- `.gitmodules` was removed. Its only submodule, `o-voxel/third_party/eigen`
  (`https://gitlab.com/libeigen/eigen.git`, commit
  `21e4582d1739107337a03460c81412981130373e`, MPL-2.0), is used instead.
  Only the `Eigen/` headers and the license files are kept, since o-voxel only
  includes `<Eigen/Dense>`.

## Downloaded at Runtime (not tracked)

- `checkpoints/`: the GenRecon weights (see Setup).
- Hugging Face cache: `microsoft/TRELLIS.2-4B` (shape and texture decoders),
  `microsoft/TRELLIS-image-large` (sparse structure decoder) and DINOv3.
- `tmp_images/` in the working directory: undistorted iPhone frames cached by
  `--mode Scannet_iphone`.

## Setup

Requirements:

- Linux x86_64 and an NVIDIA GPU with compute capability 8.0 or higher
  (Ampere or newer).
- An NVIDIA driver that supports CUDA 13 (version 580 or newer).
- A CUDA 13.x toolkit with `nvcc`, and a host compiler that toolkit supports.
  PyTorch only builds extensions with a toolkit whose major version matches
  its own CUDA version, which is 13 here. Without one, see
  [CUDA 13 Toolkit Without Root](#cuda-13-toolkit-without-root). 
- `uv`, `curl`, and ffmpeg 5.1 or newer (ffmpeg only for iPhone data).
- Access to the gated Hugging Face repo
  `facebook/dinov3-vitl16-pretrain-lvd1689m`, the image encoder that the
  pipeline configs load, and a Hugging Face login (`hf auth login`).
- About 20 GB of disk: 13 GB of checkpoints, about 2.5 GB of Hugging Face
  downloads on the first run, and the environment. Each reconstructed scene
  adds 2 to 4 GB of outputs.
- A GPU with 24 GB or more for typical rooms. A 7 × 7 m room peaked at about
  21 GB.

### CUDA 13 Toolkit Without Root

If `nvcc --version` reports anything other than 13.x and you cannot install
system packages, first install the toolkit into a directory you own:

```bash
CUDA_RUN=/var/tmp/$USER/cuda_13.0.2_580.95.05_linux.run
mkdir -p /var/tmp/$USER
curl -fL -o $CUDA_RUN https://developer.download.nvidia.com/compute/cuda/13.0.2/local_installers/cuda_13.0.2_580.95.05_linux.run
sh $CUDA_RUN --silent --toolkit --toolkitpath=/var/tmp/$USER/cuda-13.0 --no-man-page --tmpdir=/var/tmp/$USER
rm $CUDA_RUN
/var/tmp/$USER/cuda-13.0/bin/nvcc --version    # should report release 13.0
```

Then set `CUDA_HOME=/var/tmp/$USER/cuda-13.0` in the extension step below. The
toolkit is only used to compile the extensions. Nothing compiles CUDA code at
run time, so you can delete the toolkit once the import check prints `ok`.

### Environment

Upstream pins torch 2.6 with CUDA 12.x, which has no kernels for Blackwell GPUs
(sm_120). This setup uses torch 2.9.1 with CUDA 13.0 instead, together with the
flash-attn 2.8.3 release wheel built for that pair. That wheel has kernels for
sm_80, sm_90, sm_100 and sm_120.

```bash
cd papers/GenRecon
uv venv --python 3.12 --prompt genrecon .venv
source .venv/bin/activate
uv pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130
uv pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh \
    transformers==4.57.3 tensorboard pandas lpips zstandard kornia timm pillow open3d==0.19.0 \
    matplotlib plyfile safetensors huggingface_hub setuptools wheel \
    "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"
uv pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3%2Bcu13torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

# CUDA extensions, compiled for the GPU in this machine.
export CUDA_HOME=/usr/local/cuda    # path to your CUDA 13.x toolkit
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST=$(python -c "import torch; print('%d.%d' % torch.cuda.get_device_capability())")
# Build o-voxel from a copy so build files stay out of the source tree, as setup.sh does.
OVOXEL_BUILD=$(mktemp -d)
cp -r o-voxel $OVOXEL_BUILD/
uv pip install --no-build-isolation \
    "nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae" \
    "nvdiffrec_render @ git+https://github.com/JeffreyXiang/nvdiffrec.git@b296927cc7fd01c2ac1087c8065c4d7248f72da4" \
    "cumesh @ git+https://github.com/JeffreyXiang/CuMesh.git@12289e1062f0603f2f0d0771b02e1395d247f26f" \
    "flex_gemm @ git+https://github.com/JeffreyXiang/FlexGEMM.git@6dd94a859c26ee8246888502eada3dd8ad85532e" \
    $OVOXEL_BUILD/o-voxel
rm -rf $OVOXEL_BUILD
python -c "import o_voxel, cumesh, flex_gemm, nvdiffrast.torch, nvdiffrec_render; print('ok')"
```

### Checkpoints

The released checkpoints ship without the training `config.json` that
`load_train_config` expects at `<run>/config.json` for a checkpoint at
`<run>/ckpts/<name>.pt` (upstream issue #7). Lay them out like a training run
and symlink the matching config from `configs/gen/`. The weights server drops
connections now and then (upstream issues #2 and #9), so the download retries
and resumes:

```bash
for n in sparse_structure shape_slat texture_slat; do
    mkdir -p checkpoints/$n/ckpts
    curl -fL --retry 20 --retry-all-errors -C - -o checkpoints/$n/ckpts/$n.pt \
        https://kaldir.vc.cit.tum.de/genrecon/$n.pt
done
ln -sfn ../../configs/gen/ss_flow_img/genrecon.json checkpoints/sparse_structure/config.json
ln -sfn ../../configs/gen/slat_flow_img2shape/genrecon_512.json checkpoints/shape_slat/config.json
ln -sfn ../../configs/gen/slat_flow_imgshape2tex/genrecon_512.json checkpoints/texture_slat/config.json
```

## ScanNet++ Data

In the config, set `data_root` to the absolute path of `data/scannetpp` in this repo, 
list the scenes in `download_scenes`, and set `download_assets` for the mode you want.
Scenes then end up in `data/scannetpp/data/<scene_id>/`.

- `--mode Scannet_colmap` (DSLR, the mode upstream's README uses): download
  `dslr_resized_undistorted_dir`, `dslr_nerfstudio_transform_undistorted_path`
  and `dslr_colmap_dir`. GenRecon reads `dslr/resized_undistorted_images/`,
  `dslr/nerfstudio/transforms_undistorted.json` and `dslr/colmap/points3D.txt`.
- `--mode Scannet_iphone`: download `iphone_video_path` and `iphone_colmap_dir`.
  Then, from the repo root, extract the frames that COLMAP uses from
  `iphone/rgb.mkv` into `iphone/rgb/`:

  ```bash
  uv run scripts/extract_scannetpp_iphone_frames.py data/scannetpp/data/<scene_id>
  ```

- Optional, to compare against the laser scan: `scan_mesh_path`
  (`scans/mesh_aligned_0.05.ply`).

## Running on ScanNet++

`--num_imgs_per_scene` is the number of input photos. 32 is the code default
and the value in upstream's README. The paper's ScanNet++ evaluation uses 8 input
views per scene (25 scenes) and does not say whether they are DSLR or iPhone
images, or how they were picked. The code takes evenly spaced frames along the
capture, and in iPhone mode it splits each 4:3 frame into two square crops, so
32 frames give 64 views.

Run from `papers/GenRecon`. `--mode Scannet_iphone` caches undistorted frames
in `tmp_images/<scene_id>/` under the working directory but does not create
`tmp_images/` itself, hence the `mkdir`. Outputs go to the untracked `outputs/`
folder at the repo root.

```bash
SCENE=2a1b555966
MODE=Scannet_colmap    # or Scannet_iphone
OUT=../../outputs/genrecon/$SCENE
mkdir -p tmp_images
.venv/bin/python reconstruct_scene.py \
    --mode $MODE \
    --path ../../data/scannetpp/data/$SCENE \
    --output_path $OUT \
    --ss_ckpt checkpoints/sparse_structure/ckpts/sparse_structure.pt \
    --shape_ckpt checkpoints/shape_slat/ckpts/shape_slat.pt \
    --tex_ckpt checkpoints/texture_slat/ckpts/texture_slat.pt \
    --pipeline_config configs/pipelines/original.json \
    --num_imgs_per_scene 32
PYTORCH_NO_CUDA_MEMORY_CACHING=1 .venv/bin/python chunked_to_glb.py \
    --inputs $OUT/to_glb_inputs.pt --chunk_inputs $OUT/chunk_inputs.pt --output_dir $OUT
```

`chunked_to_glb.py` first remeshes the whole scene on the GPU and keeps those
buffers while it bakes each chunk with CuMesh. CuMesh allocates GPU memory
itself, so it cannot use memory that PyTorch has freed but still holds in its
cache. Without `PYTORCH_NO_CUDA_MEMORY_CACHING=1`, a 7 × 7 m room (12 chunks)
ran out of the RTX 5090's 32 GB at the first chunk. With it, the same scene
peaked at about 20 GB. When a chunk fails, the script prints
`[chunked_to_glb] FAILED` but still exits with status 0, so check that
`scene.glb` was written. Finished chunks are kept in `$OUT/chunks/`, and a rerun
continues from them.
