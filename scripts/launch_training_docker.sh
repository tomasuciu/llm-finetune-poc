#!/usr/bin/env bash
#SBATCH -J elastic-demo
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -o elastic-demo.%j.out
#SBATCH -e elastic-demo.%j.err

set -euo pipefail

# ----------------- DOCKER CONFIG -----------------
DOCKER_IMAGE="your-registry/llm-finetune-poc:latest"  # Replace with your actual image
DOCKER_RUNTIME="--runtime=nvidia"  # Use nvidia runtime for GPU access

# If no IB devices, force TCP & pick default NIC
if [ ! -d /sys/class/infiniband ] || [ -z "$(ls -A /sys/class/infiniband 2>/dev/null)" ]; then
  IFACE="$(ip -o -4 route show to default | awk '{print $5}' | head -n1)"
  export TORCH_NCCL_IB_DISABLE=1
  export TORCH_NCCL_SOCKET_IFNAME="$IFACE"
  export GLOO_SOCKET_IFNAME="$IFACE"
fi

# ----------------- CONFIG -----------------
CONFIG_JSON="/mnt/training-data/llm-finetune-poc/config/mistral-7b.json"

ELASTIC_MIN_NODES=3
ELASTIC_MAX_NODES=4
PROCS_PER_NODE=${SLURM_GPUS_ON_NODE:-1}

RDZV_PORT=29401
RDZV_ID="elastic-demo-${SLURM_JOB_ID}"
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)

echo "Master: $MASTER_ADDR  RDZV_ID: $RDZV_ID"
echo "Elastic nnodes: ${ELASTIC_MIN_NODES}:${ELASTIC_MAX_NODES}, nproc_per_node=${PROCS_PER_NODE}"

# Helpers: tiny CPU, no GPUs, allow overlap with training
SRUN_SHARED="--overlap --gpus-per-task=0 --cpus-per-task=1"
SRUN_LABEL="-l -u"  # label lines by task index; still single out/err

# Prep dirs (helper step)
srun $SRUN_LABEL $SRUN_SHARED --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -lc "
  docker run --rm \
    -v /mnt/training-data:/mnt/training-data \
    ${DOCKER_IMAGE} \
    bash -c 'mkdir -p /mnt/training-data/elastic-demo/elastic || true'
"

# ------------------- TRAINING --------------------
srun $SRUN_LABEL --kill-on-bad-exit=0 --exclusive --ntasks-per-node=1 bash -lc "
set -exuo pipefail

# Export environment variables for the container
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600
export TORCH_NCCL_ENABLE_MONITORING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_SOCKET_TIMEOUT=300000
export TORCH_NCCL_IB_DISABLE=${TORCH_NCCL_IB_DISABLE:-0}
export TORCH_NCCL_SOCKET_IFNAME=${TORCH_NCCL_SOCKET_IFNAME:-}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-}

echo \"[TRAIN][\$(hostname -s)] starting torchrun in Docker at [\$(date -Ins)]\"

docker run --rm \
  ${DOCKER_RUNTIME} \
  --gpus all \
  --ipc=host \
  --network=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /mnt/training-data:/mnt/training-data \
  -e NCCL_DEBUG \
  -e PYTHONUNBUFFERED \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING \
  -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC \
  -e TORCH_NCCL_ENABLE_MONITORING \
  -e TORCH_DISTRIBUTED_DEBUG \
  -e NCCL_SOCKET_TIMEOUT \
  -e TORCH_NCCL_IB_DISABLE \
  -e TORCH_NCCL_SOCKET_IFNAME \
  -e GLOO_SOCKET_IFNAME \
  ${DOCKER_IMAGE} \
  torchrun \
    --nnodes=${ELASTIC_MIN_NODES}:${ELASTIC_MAX_NODES} \
    --nproc_per_node=${PROCS_PER_NODE} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${RDZV_PORT} \
    --rdzv_id=${RDZV_ID} \
    --rdzv-conf=timeout=600 \
    --rdzv-conf=join_timeout=300 \
    --max-restarts=3 \
    -m llm_finetune_poc.cli \
    ${CONFIG_JSON}
"
