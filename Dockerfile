# FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel
# FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
        git \
        ninja-build \
        libglib2.0-0 \
        libxrender-dev \
    # && dpkg-reconfigure --frontend noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        python3-pip \
        cmake \
        python3-dev \
        python3-opencv \
        g++-9

# Set g++-9 as default g++ to ensure C++17 compatibility
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 90
RUN apt-get install -y libsparsehash-dev

RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install OpenMMLab projects
RUN pip install --no-deps \
    # mmengine==0.7.3 \
    mmengine==0.8.5 \
    mmdet==3.2.0
    # mmsegmentation==1.0.0 \
    # git+https://github.com/open-mmlab/mmdetection3d.git@22aaa47fdb53ce1870ff92cb7e3f96ae38d17f61
    # git+https://github.com/open-mmlab/mmdetection3d.git@v1.3.0 \
    # mmsegmentation==1.2.2

# RUN pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13.0/index.html --no-deps
# RUN pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html --no-deps
RUN pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html --no-deps



# Install MinkowskiEngine
# Feel free to skip nvidia-cuda-dev if minkowski installation is fine
RUN apt-get update \
    && apt-get -y install libopenblas-dev nvidia-cuda-dev
RUN TORCH_CUDA_ARCH_LIST="6.1 7.0 8.6 8.9" \
    pip install git+https://github.com/NVIDIA/MinkowskiEngine.git@02fc608bea4c0549b0a7b00ca1bf15dee4a0b228 -v --no-deps \
    --install-option="--blas=openblas" \
    --install-option="--force_cuda"

# Install torch-scatter 
# RUN pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-1.13.0+cu116.html --no-deps
RUN pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html --no-deps


# # # Install remaining python packages
RUN pip install --no-deps \
#     spconv-cu116==2.3.6 \
    spconv-cu118==2.3.8 \
    numpy==1.24.1 \
    opencv-python==4.7.0.72 \
    scipy==1.10.1 \
# #     cumm-cu116==0.4.9 \
    cumm-cu118==0.7.11 \
    trimesh==3.21.6 \
    open3d==0.17.0 \
    plyfile==1.0.2 \
    networkx==3.4.2 \
    scikit-image==0.25.2 \
    yapf==0.43.0 \
    rich==13.3.5 \
    trimesh==3.21.6 \
    git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0 \
    tensorboardX==2.6.5 \
    terminaltables==3.1.10 \
    pccm

WORKDIR /workspace/plantsegstudio
COPY --chown=containerUser:1000 . .
RUN pip install -v -e .

ENTRYPOINT [ "/bin/bash", "--login" , "-c"]
CMD ["bash"]


################################################################################
# Container description & use case...
#
# Build:
#   docker composse build
# Run as devcontainer:
#   docker run -it --rm --gpus all -v ${PWD}:/workspace/plantsegstudio --network=host plantsegstudio-interactive:latest 
# To jump into this container if already running, use
#   docker exec -it plantsegstudio-interactive:latest  bash
################################################################################