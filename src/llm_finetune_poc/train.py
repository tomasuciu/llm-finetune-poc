import os
import logging

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
    AutoModelForCausalLM,

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
    ModelArguments,
    DataArguments,
    TrainingArguments,
)

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def prepare_dataset():
    pass


def main():

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    transformers.set_seed(training_args.seed)


    print(model_args)

    #if not dist.is_initialized():
    #    dist.init_process_group(backend="nccl")
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # init process group if not already done
    if local_rank != -1 and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    
    # TODO: standardize logging
    print(f"Rank {global_rank}/{world_size}, Local rank: {local_rank}")

    if global_rank == 0:
        logger.info("=" * 80)
        logger.info("Training Configuration:")
        logger.info("=" * 80)
        logger.info(f"Model: {model_args.model_name_or_path}")
        logger.info(f"Dataset: {data_args.dataset_name}")
        logger.info(f"Output: {training_args.output_dir}")
        logger.info(f"Epochs: {training_args.num_train_epochs}")
        logger.info(f"Batch size per device: {training_args.per_device_train_batch_size}")
        logger.info(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
        logger.info(f"Learning rate: {training_args.learning_rate}")
        logger.info(f"World size: {world_size}")

        effective_batch_size = (
            training_args.per_device_train_batch_size * 
            training_args.gradient_accumulation_steps * 
            world_size
        )
        logger.info(f"Effective batch size: {effective_batch_size}")
        logger.info("=" * 80)

    # select oss weights for fine-tuning
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
        attn_implementation="flash_attention_2" if model_args.use_flash_attention_2 else None,
        trust_remote_code=True,
    )
    
    # Enable gradient checkpointing for memory efficiency
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

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
