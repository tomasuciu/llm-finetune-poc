import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP
)
import transformers
from transformers import (
    TrainingArguments,
    HfArgumentParser,
    Trainer,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.elastic.multiprocessing.errors import record

from llm_finetune_poc.arguments import (
    ModelConfig,
    DataConfig,
    TrainingConfig,
)



def main():

    parser = HfArgumentParser((ModelConfig, DataConfig, TrainingConfig))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    torch.cuda.set_device(local_rank)
    
    # TODO: standardize logging
    print(f"Rank {global_rank}/{world_size}, Local rank: {local_rank}")

    if global_rank == 0:
        # TODO: create experiment, if tracking
        pass

    # select oss weights for fine-tuning
    model = None

    auto_wrap_policy = size_based_auto_wrap_policy(
        min_num_params=1e8
    )
    
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
    )
    
    return model




if __name__ == "__main__":
    main()
