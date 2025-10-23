# LLM Fine-Tuning POC

A production-grade proof-of-concept for distributed fine-tuning of multi-billion-parameter language models on multi-node GPU clusters, optimized for function calling capabilities.

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Infrastructure Setup](#infrastructure-setup)
4. [Technical Decisions](#technical-decisions)
5. [Deployment Guide](#deployment-guide)
6. [Docker vs Virtual Environments](#docker-vs-virtual-environments)
7. [Fault Tolerance & Resilience](#fault-tolerance--resilience)
8. [Monitoring & Observability](#monitoring--observability)
9. [Training Guide](#training-guide)
10. [Troubleshooting](#troubleshooting)

## Overview

This system provides a production-grade framework for distributed fine-tuning of large language models (billion-parameter scale) on multi-node GPU clusters. Built for the Nebius AI Cloud platform with SOperator orchestration, it demonstrates efficient training at scale with robust fault tolerance and checkpoint management.

## Training Architecture

The training system uses a **distributed data-parallel setup** built on **Fully Sharded Data Parallel (FSDP)** to efficiently scale across multiple GPUs and nodes. Each worker process (e.g., `Worker 0`, `Worker 1`, … `Worker N`) manages its own GPU and communicates with other workers through the **NCCL backend** using **All-Reduce** operations for gradient synchronization.

Two types of checkpoints are maintained to ensure reliability and flexibility during training:

- **Sharded Checkpoints** – Saved at regular intervals (e.g., `checkpoint-300`, `checkpoint-600`) and store model shards per worker.  
- **Elastic Checkpoints** – Saved less frequently (e.g., `elastic-1000`, `elastic-2000`) and capture the **full training state** for recovery or resumption.
### Component Overview

#### Training Components
- **train.py**: Main training orchestration
- **arguments.py**: Configuration dataclasses
- **elastic_checkpoint.py**: Elastic checkpoint management
- **elastic_callback.py**: Training callbacks for elastic support
- **fault_tolerance.py**: Signal handling and checkpoint validation

#### Infrastructure Components
- **Terraform modules**: Infrastructure as Code for Nebius Cloud
- **SLURM**: Job scheduling and resource management
- **Kubernetes**: Container orchestration
- **Filestore**: High-performance storage

---

## Infrastructure Setup

### Prerequisites

1. **Nebius AI Cloud Account**
   - Active account with billing enabled
   - Project and tenant IDs
   - IAM credentials

2. **Local Tools**
   ```bash
   # Install Nebius CLI
   curl -sSL https://storage.eu-north1.nebius.cloud/nebius/install.sh | bash
   
   # Install Terraform
   wget https://releases.hashicorp.com/terraform/1.5.0/terraform_1.5.0_linux_amd64.zip
   ```

3. **SSH Key Pair**
   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   ```

## Technical Decisions

### 1. PyTorch FSDP vs DeepSpeed

**Decision**: Use PyTorch FSDP (Fully Sharded Data Parallel)

**Rationale**:
- **Native PyTorch integration**: First-class support in PyTorch 2.0+
- **Simpler codebase**: Less configuration overhead than DeepSpeed
- **Better HuggingFace integration**: Seamless with Transformers library
- **Automatic sharding**: Handles model sharding transparently
- **Memory efficiency**: Similar to DeepSpeed ZeRO Stage 3

**Trade-offs**:
- DeepSpeed offers more advanced features (pipeline parallelism, ZeRO-Offload)
- FSDP is newer, less battle-tested than DeepSpeed
- DeepSpeed has more tuning options for extreme scale

### 2. Checkpoint Strategy: Hybrid Approach

**Decision**: Implement dual checkpoint system (sharded + elastic)

**Components**:

#### Sharded Checkpoints (Standard)
```python
# Saved every N steps (e.g., 300)
checkpoint-300/
  ├── config.json
  ├── trainer_state.json
  ├── pytorch_model-00001-of-00004.bin
  ├── pytorch_model-00002-of-00004.bin
  └── _SUCCESS
```

**Characteristics**:
- Fast to save (only rank 0 saves state_dict)
- Efficient for same-world-size recovery
- Cannot load if world size changes
- Stored in `/mnt/training-data/checkpoints`

#### Elastic Checkpoints (Full State Dict)
```python
# Saved every M steps (e.g., 1000) or on world size change
elastic-checkpoint-1000/
  ├── config.json
  ├── trainer_state.json
  ├── pytorch_model.bin  # FULL unsharded weights
  ├── elastic_metadata.json
  └── _ELASTIC_SUCCESS
```

**Characteristics**:
- Slower to save (all ranks gather full model)
- Can load with different world size
- Enables dynamic node scaling
- Stored in `/mnt/training-data/checkpoints/elastic`

**Rationale**:
- **Sharded checkpoints** provide fast recovery in stable conditions
- **Elastic checkpoints** enable recovery from node failures/preemptions
- **Hybrid approach** balances performance and resilience

### 3. NCCL Backend for Communication

**Decision**: Use NCCL (NVIDIA Collective Communications Library)

**Rationale**:
- **GPU-optimized**: Specifically designed for NVIDIA GPUs
- **High bandwidth**: Optimized for InfiniBand and high-speed networks
- **Ring-allreduce**: Efficient gradient synchronization
- **Industry standard**: Used by all major frameworks

**Configuration**:
```python
# In arguments.py
ddp_backend: str = "nccl"

# Environment variables
NCCL_DEBUG=INFO
NCCL_SOCKET_IFNAME=^docker0,lo  # Exclude docker and loopback
```

### 4. Flash Attention 2

**Decision**: Enable Flash Attention 2 by default

**Rationale**:
- **2-4x faster attention**: Reduces training time significantly
- **Memory efficient**: Reduces memory footprint by ~20-30%
- **Numerically stable**: Maintains training quality
- **Hardware optimized**: Leverages Tensor Cores on H100/H200

**Trade-off**: Requires CUDA 11.8+ and compatible GPUs

### 5. Gradient Checkpointing

**Decision**: Enable gradient checkpointing by default

**Rationale**:
- **Memory reduction**: Saves 30-40% GPU memory
- **Enables larger batch sizes**: More efficient training
- **Minimal slowdown**: ~15-20% increase in training time
- **Essential for large models**: Required for 7B+ models on consumer GPUs

**Implementation**:
```python
# In train.py
if training_args.gradient_checkpointing:
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
```

### 6. BFloat16 Mixed Precision

**Decision**: Use bfloat16 instead of float16

**Rationale**:
- **Better numerical stability**: Larger exponent range
- **No loss scaling required**: Simpler training loop
- **Native H100/H200 support**: Hardware acceleration
- **HuggingFace recommendation**: Better compatibility

**Configuration**:
```python
bf16: bool = True  # In CustomTrainingArguments
```

### 7. SLURM on Kubernetes

**Decision**: Use SLURM operator on Kubernetes instead of bare-metal SLURM

**Rationale**:
- **Cloud-native**: Better integration with cloud services
- **Elastic scaling**: Auto-scaling based on demand
- **Resource efficiency**: Better node utilization
- **Fault tolerance**: Kubernetes handles node failures
- **Monitoring**: Integrated with Prometheus/Grafana

**Trade-offs**:
- More complex setup than bare-metal
- Additional abstraction layer
- Requires Kubernetes expertise

### 8. Filestore vs Node-local Storage

**Decision**: Use Filestore for checkpoints, node-local for scratch

**Storage Strategy**:
```
/mnt/training-data/      # Filestore (shared, persistent)
  ├── checkpoints/       # Model checkpoints (745 GiB)
  ├── cache/            # HuggingFace cache (1024 GiB)
  └── datasets/         # Training datasets

/mnt/scratch/           # Node-local (fast, ephemeral)
  ├── tmp/             # Temporary files (186 GiB per node)
  └── preprocessed/    # Cached preprocessed data
```

**Rationale**:
- **Filestore**: Durable, accessible from all nodes, survives failures
- **Node-local**: Fast I/O, ideal for temporary data
- **Hybrid approach**: Balances performance and reliability

### 9. PyTorch Elastic Training

**Decision**: Use `torch.distributed.elastic` with custom enhancements

**Implementation**:
```python
from torch.distributed.elastic.multiprocessing.errors import record

@record  # Decorator for better error messages
def main():
    # Training code
    ...
```

**Enhancements**:
- **World size monitoring**: Detect dynamic node changes
- **Safe barriers**: Timeout-protected synchronization
- **Elastic checkpoints**: Support for world size changes
- **Graceful recovery**: Automatic restart on failure

**Rationale**:
- **Resilience**: Handles transient failures
- **Elasticity**: Supports dynamic scaling
- **Visibility**: Better error reporting
- **Industry adoption**: Used by major training systems

---
## Deployment Guide

### Step 1: Configure Environment

1. **Set up credentials**:
   ```bash
   
   # Set your credentials
   export NEBIUS_TENANT_ID="tenant-xxxxx"
   export NEBIUS_PROJECT_ID="project-xxxxx"
   export NEBIUS_REGION="eu-north1"
   
   # Source the environment
   source .envrc
   ```

2. **The `.envrc` script automatically**:
   - Validates credentials
   - Retrieves IAM token
   - Creates service account for Terraform
   - Sets up backend for Terraform state
   - Exports all necessary environment variables

### Step 2: Configure Terraform Variables

1. **Review and customize `terraform.tfvars`**:
   ```hcl
   # Company/cluster name
   company_name = "your-company"
   
   # Worker node configuration
   slurm_nodeset_workers = [{
     size = 4  # Number of GPU nodes
     resource = {
       platform = "gpu-h200"
       preset   = "1gpu-16vcpu-200gb"
     }
     boot_disk = {
       type                 = "NETWORK_SSD"
       size_gibibytes       = 256
       block_size_kibibytes = 4
     }
   }]
   
   # Storage configuration
   filestore_jail_submounts = [
     {
       name       = "training-data"
       mount_path = "/mnt/training-data"
       spec = {
         size_gibibytes       = 1024
         block_size_kibibytes = 4
       }
     },
     {
       name       = "checkpoints"
       mount_path = "/mnt/checkpoints"
       spec = {
         size_gibibytes       = 745
         block_size_kibibytes = 4
       }
     }
   ]
   
   # SSH access
   slurm_login_ssh_root_public_keys = [
     "ssh-ed25519 AAAAC3... your_email@example.com"
   ]
   ```

2. **Key variables to adjust**:
   - `slurm_nodeset_workers`: GPU node count and type
   - Storage sizes based on model and dataset requirements
   - SSH keys for cluster access
   - Monitoring/telemetry settings

### Step 3: Deploy Infrastructure

```bash
# Initialize Terraform
terraform init

# Preview changes
terraform plan

# Deploy infrastructure (takes 30-45 minutes)
terraform apply

# Save outputs
terraform output > outputs.txt
```

### Step 4: Access the Cluster

1. **Get login node IP**:
   ```bash
   terraform output slurm_login_ip
   # Output: 51.250.x.x
   ```

2. **SSH to login node**:
   ```bash
   ssh root@<login-ip>
   ```

3. **Verify SLURM**:
   ```bash
   sinfo  # Show node status
   squeue # Show job queue
   scontrol show nodes  # Detailed node info
   ```

# Step 5: Deploy Training Code

You have two options for deploying your training code: **Virtual Environment** (recommended for development) or **Docker** (recommended for production).

## Quick Comparison

| Aspect | Virtual Environment | Docker |
|--------|-------------------|---------|
| **Setup Time** | Fast (5-10 min) | Slower (20-30 min build) |
| **Iteration Speed** | Instant (edit and run) | Slow (rebuild required) |
| **Debugging** | Easy (direct access) | Harder (exec into container) |
| **Reproducibility** | Good (requirements.txt) | Excellent (frozen image) |
| **Disk Usage** | Minimal (~2 GB) | Large (~10-15 GB per image) |
| **Best For** | Development, debugging, experimentation | Production, deployment, strict reproducibility |

**Recommendation**: Start with **Option A (Virtual Environment)** for faster iteration and easier debugging. Switch to Docker for production deployments or when you need to share exact environments across teams.

---

## Option A: Virtual Environment (Recommended for Development)

This approach gives you the fastest iteration cycle and easiest debugging experience.

### Step 1: Copy code to cluster

```bash
# From your local machine
scp -r src/ root@:/home/ubuntu/llm-finetune-poc/
scp pyproject.toml root@:/home/ubuntu/llm-finetune-poc/
scp requirements.txt root@:/home/ubuntu/llm-finetune-poc/

# Alternatively:
cd /home/ubuntu && git clone https://github.com/tomasuciu/llm-finetune-poc.git
```

### Step 2: SSH to cluster and create virtual environment

```bash
# SSH to login node
ssh root@

# Navigate to shared storage (accessible from all nodes)
cd /home/ubuntu

# Create virtual environment
python3 -m venv llm-finetune-env

# Activate virtual environment
source llm-finetune-env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Step 3: Install dependencies

```bash
# Install PyTorch (adjust CUDA version as needed)
pip install torch==2.1.0+cu121 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121

# Install the training package in editable mode
pip install -e .

# Install flash attention (optional but recommended)
pip install flash-attn==2.3.3 --no-build-isolation

```

### Step 4: Submit the job

```bash
# Make script executable
chmod +x launch_training_venv.sbatch

# Submit job
sbatch launch_training_venv.sbatch

# Check job status
squeue -u $USER
```

### Advantages of Virtual Environment

- **Fast iteration**: Edit code and rerun immediately (no rebuild)
- **Easy debugging**: Direct access to Python environment
- **Quick updates**: `pip install -e .` for instant code changes
- **Lower disk usage**: No image storage overhead
- **Familiar workflow**: Standard Python development practices
- **Better error messages**: Full stack traces without container layers

### Disadvantages

- Environment may drift between nodes (mitigated by shared filesystem)
- Less reproducible than Docker
- Requires manual dependency management

---

## Option B: Docker Container (For Production Deployment)

Use this approach when you need strict reproducibility or are deploying to production.

### Step 1: Build Docker image locally

```bash
# From your project root (where Dockerfile is)
./scripts/build.sh

# Tag for registry
docker tag your-registry/llm-finetune:v1.0 your-registry/llm-finetune:latest

# Push to container registry
docker push your-registry/llm-finetune:v1.0
docker push your-registry/llm-finetune:latest
```

### Step 2: Submit Docker job

```bash
sbatch launch_training_docker.sbatch
```

### Advantages of Docker

- **Perfect reproducibility**: Exact environment every time
- **Version control**: Tag and track environment versions
- **Easy distribution**: Share images across teams/clusters
- **Isolation**: No dependency conflicts
- **Production-ready**: Standard deployment method

### Disadvantages

- Slow iteration: Must rebuild image for code changes (minutes)
- Debugging complexity: Need to exec into container or rebuild
- Larger disk footprint: 10-15 GB per image
- Build time: 20-30 minutes for initial build
- Registry setup: Need container registry (ideally via IaC)

---

