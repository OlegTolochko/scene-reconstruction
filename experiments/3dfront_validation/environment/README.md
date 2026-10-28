# Environments

Run setup from the repository root.

## CPU validation

```bash
conda env create -f experiments/3dfront_validation/environment/environment.yml
conda activate 3dfront-validation
```

`environment-resolved.yml` records the captured pilot environment.

## GenRecon GPU and latent baseline

This setup targets the RTX 5090 (`sm_120`) with CUDA 13. For requirements and
other hardware, see [GenRecon setup](../../../papers/GenRecon/README_scene-reconstruction.md).

```bash
conda create -n genrecon -c conda-forge python=3.12 pip
conda activate genrecon
python -m pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130
conda install -c nvidia -c conda-forge cuda-nvcc=13.0 cuda-cudart-dev=13.0 cuda-cccl=13.0 cuda-nvrtc-dev=13.0 cuda-driver-dev=13.0
python -m pip install --no-cache-dir -r experiments/3dfront_validation/environment/genrecon-requirements.txt
python -m experiments.3dfront_validation.environment.install_genrecon_extensions
```

Image-conditioned inference also requires approved access to
`facebook/dinov3-vitl16-pretrain-lvd1689m` and `hf auth login` in `genrecon`.
