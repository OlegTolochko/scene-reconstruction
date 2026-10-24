# GenRecon setup for RTX 5090 (Blackwell, sm_120): CUDA 12.8, torch 2.7.1, xformers.
# No sudo needed. Run from the GenRecon folder with:  . ./setup2.sh --all
# (source it with "." rather than bash, so conda activate sticks in your shell)

TEMP=`getopt -o h --long help,all,new-env,cuda,basic,xformers,cumesh,o-voxel,flexgemm,nvdiffrast,nvdiffrec -n 'setup2.sh' -- "$@"`
eval set -- "$TEMP"

HELP=false
NEW_ENV=false
CUDA=false
BASIC=false
XFORMERS=false
CUMESH=false
OVOXEL=false
FLEXGEMM=false
NVDIFFRAST=false
NVDIFFREC=false
ERROR=false

if [ "$#" -eq 1 ] ; then
    HELP=true
fi

while true ; do
    case "$1" in
        -h|--help) HELP=true ; shift ;;
        --all) NEW_ENV=true ; CUDA=true ; BASIC=true ; XFORMERS=true ; NVDIFFRAST=true ; NVDIFFREC=true ; CUMESH=true ; FLEXGEMM=true ; OVOXEL=true ; shift ;;
        --new-env) NEW_ENV=true ; shift ;;
        --cuda) CUDA=true ; shift ;;
        --basic) BASIC=true ; shift ;;
        --xformers) XFORMERS=true ; shift ;;
        --cumesh) CUMESH=true ; shift ;;
        --o-voxel) OVOXEL=true ; shift ;;
        --flexgemm) FLEXGEMM=true ; shift ;;
        --nvdiffrast) NVDIFFRAST=true ; shift ;;
        --nvdiffrec) NVDIFFREC=true ; shift ;;
        --) shift ; break ;;
        *) ERROR=true ; break ;;
    esac
done

if [ "$ERROR" = true ] ; then
    echo "Error: Invalid argument"
    HELP=true
fi

if [ "$HELP" = true ] ; then
    echo "Usage: . ./setup2.sh [OPTIONS]"
    echo "Options:"
    echo "  -h, --help      Display this help message"
    echo "  --all           Do every step below, in order"
    echo "  --new-env       Create the genrecon conda env with torch 2.7.1 (cu128)"
    echo "  --cuda          Install the CUDA 12.8 compiler into the env and set env vars"
    echo "  --basic         Install basic Python dependencies"
    echo "  --xformers      Install xformers (attention backend, replaces flash-attn)"
    echo "  --nvdiffrast    Build nvdiffrast"
    echo "  --nvdiffrec     Build nvdiffrec"
    echo "  --cumesh        Build CuMesh"
    echo "  --flexgemm      Build FlexGEMM"
    echo "  --o-voxel       Build o-voxel"
    return 0 2>/dev/null || exit 0
fi

if ! command -v nvidia-smi > /dev/null; then
    echo "Error: No NVIDIA GPU found"
    return 1 2>/dev/null || exit 1
fi

WORKDIR=$(pwd)
EXT_DIR=/tmp/$USER/extensions
TORCH_INDEX=https://download.pytorch.org/whl/cu128

# Make "conda activate" work inside a script
eval "$(conda shell.bash hook)"

if [ "$NEW_ENV" = true ] ; then
    conda create -y -n genrecon python=3.10
    conda activate genrecon
    pip install torch==2.7.1 torchvision==0.22.1 --index-url $TORCH_INDEX
else
    conda activate genrecon
fi

if [ "$CUDA" = true ] ; then
    conda install -y -c "nvidia/label/cuda-12.8.1" cuda-toolkit
    # Saved in the env, so they are set again on every "conda activate genrecon"
    conda env config vars set \
        CUDA_HOME="$CONDA_PREFIX" \
        TORCH_CUDA_ARCH_LIST="12.0" \
        ATTN_BACKEND=xformers \
        SPARSE_ATTN_BACKEND=xformers
    conda deactivate
    conda activate genrecon
fi

# Needed for every build below (also covers runs without --cuda)
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="12.0"
export ATTN_BACKEND=xformers
export SPARSE_ATTN_BACKEND=xformers

if [ "$BASIC" = true ] ; then
    pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh transformers==4.57.3 gradio==6.0.1 tensorboard pandas lpips zstandard psutil plyfile
    pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
    conda install -y -c conda-forge libjpeg-turbo
    pip install pillow-simd
    pip install kornia timm
fi

if [ "$XFORMERS" = true ] ; then
    # Pinning torch stops pip from swapping it for a different version
    pip install torch==2.7.1 xformers --index-url $TORCH_INDEX
fi

# Builds below need nvcc 12.8
if [ "$NVDIFFRAST$NVDIFFREC$CUMESH$FLEXGEMM$OVOXEL" != "falsefalsefalsefalsefalse" ] ; then
    if ! nvcc --version 2>/dev/null | grep -q "release 12.8" ; then
        echo "Error: nvcc 12.8 not found. Run with --cuda first."
        return 1 2>/dev/null || exit 1
    fi
    mkdir -p $EXT_DIR
fi

if [ "$NVDIFFRAST" = true ] ; then
    rm -rf $EXT_DIR/nvdiffrast
    git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git $EXT_DIR/nvdiffrast
    pip install $EXT_DIR/nvdiffrast --no-build-isolation
fi

if [ "$NVDIFFREC" = true ] ; then
    rm -rf $EXT_DIR/nvdiffrec
    git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git $EXT_DIR/nvdiffrec
    pip install $EXT_DIR/nvdiffrec --no-build-isolation
fi

if [ "$CUMESH" = true ] ; then
    rm -rf $EXT_DIR/CuMesh
    git clone https://github.com/JeffreyXiang/CuMesh.git $EXT_DIR/CuMesh --recursive
    pip install $EXT_DIR/CuMesh --no-build-isolation
fi

if [ "$FLEXGEMM" = true ] ; then
    rm -rf $EXT_DIR/FlexGEMM
    git clone https://github.com/JeffreyXiang/FlexGEMM.git $EXT_DIR/FlexGEMM --recursive
    pip install $EXT_DIR/FlexGEMM --no-build-isolation
fi

if [ "$OVOXEL" = true ] ; then
    rm -rf $EXT_DIR/o-voxel
    cp -r $WORKDIR/o-voxel $EXT_DIR/o-voxel
    # --no-deps: CuMesh and FlexGEMM are already built above
    pip install $EXT_DIR/o-voxel --no-build-isolation --no-deps
fi

cd $WORKDIR
echo ""
echo "Checking install..."
python -c "import torch; print('torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0)); print('GPU test sum:', torch.randn(8,8,device='cuda').sum().item())"
python -c "import xformers, nvdiffrast.torch, cumesh, flex_gemm, o_voxel; print('all extensions import ok')"
