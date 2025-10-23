FROM nvcr.io/nvidia/pytorch:24.07-py3 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    NCCL_DEBUG=INFO \
    NCCL_SOCKET_IFNAME=^docker0,lo

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    vim \
    htop \
    tmux \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /workspace/llm-finetune-poc
COPY pyproject.toml .
COPY src ./src

# install the llm_finetune_poc package
RUN pip install --no-cache-dir .

COPY src /workspace/llm-finetune-poc/src

# Install the package
RUN pip install -e /workspace/llm-finetune-poc

# Create directories for data and checkpoints; TODO: revise
RUN mkdir -p /mnt/training-data/checkpoints /mnt/training-data/cache

# Set HuggingFace cache directory
ENV HF_HOME=/mnt/training-data/cache \
    TRANSFORMERS_CACHE=/mnt/training-data/cache \
    HF_DATASETS_CACHE=/mnt/training-data/cache

WORKDIR /workspace

ENTRYPOINT ["python", "-m", "torch.distributed.run"]
CMD ["--help"]
