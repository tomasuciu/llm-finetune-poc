import os
import logging
import functools
from typing import Any, Dict, List

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
    AutoTokenizer,
)
from datasets import load_dataset
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


def preprocess_function_calling_dataset(
    examples: Dict[str, List],
    tokenizer: AutoTokenizer,
    max_seq_length: int,
) -> Dict[str, List]:
    processed_texts = []
    
    for idx in range(len(examples['system'])):
        conversation = []
        
        if examples['system'][idx]:
            conversation.append({
                "role": "system",
                "content": examples['system'][idx]
            })
        
        if examples['chat'][idx]:
            conversation.append({
                "role": "user", 
                "content": examples['chat'][idx]
            })
        
        text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )
        processed_texts.append(text)
    
    tokenized = tokenizer(
        processed_texts,
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
        return_tensors="pt",
    )
    
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized

def setup_model_and_tokenizer(model_args: ModelArguments, training_args: TrainingArguments):
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )
    
    # pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logger.info(f"Loading model: {model_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
        attn_implementation="flash_attention_2" if model_args.use_flash_attention_2 else None,
        trust_remote_code=True,
    )
    
    # Enable gradient checkpointing for memory efficiency
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

        # This is important for gradient checkpointing to work properly
        model.config.use_cache = False
    
    return model, tokenizer


def prepare_dataset(
    data_args: DataArguments,
    tokenizer: AutoTokenizer,
    max_seq_length: int,
):
    logger.info(f"Loading dataset: {data_args.dataset_name}")
    
    try:
        # Load dataset
        dataset = load_dataset(
            data_args.dataset_name,
            split=data_args.dataset_split,
            trust_remote_code=True,
        )
        
        logger.info(f"Dataset loaded: {len(dataset)} examples")
        
        # Create train/validation split
        if data_args.validation_split_percentage > 0:
            split_dataset = dataset.train_test_split(
                test_size=data_args.validation_split_percentage / 100,
                seed=42,
            )
            train_dataset = split_dataset["train"]
            eval_dataset = split_dataset["test"]
        else:
            train_dataset = dataset
            eval_dataset = None
        
        logger.info("Preprocessing dataset...")
        
        preprocess_fn = functools.partial(
            preprocess_function_calling_dataset,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
        
        train_dataset = train_dataset.map(
            preprocess_fn,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing training data",
        )
        
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(
                preprocess_fn,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=eval_dataset.column_names,
                desc="Tokenizing validation data",
            )
        
        logger.info(f"Training examples: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"Validation examples: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def main():

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    transformers.set_seed(training_args.seed)
    
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

    try:
        model, tokenizer = setup_model_and_tokenizer(model_args, training_args)
        logger.info("Model and tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    try:
        train_dataset, eval_dataset = prepare_dataset(
            data_args,
            tokenizer,
            model_args.max_seq_length,
        )
    except Exception as e:
        logger.error(f"Failed to prepare dataset: {e}")
        raise
    
    # Configure FSDP mixed precision
    # Use BF16 for better numerical stability on H100/H200
    if training_args.bf16:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    else:
        mp_policy = None
    
    # Configure FSDP wrapping policy
    # Wrap transformer blocks for optimal memory/compute tradeoff
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy,
        min_num_params=1e8,  # Wrap layers with >100M params
    )
    
    # FSDP configuration
    training_args.fsdp = "full_shard auto_wrap"
    training_args.fsdp_config = {
        "auto_wrap_policy": auto_wrap_policy,
        "mixed_precision_policy": mp_policy,
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        "sharding_strategy": ShardingStrategy.FULL_SHARD,
        "cpu_offload": False,
        "limit_all_gathers": True,
    }
    
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # Resume from checkpoint if available
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        logger.info(f"Resuming from checkpoint: {checkpoint}")
    elif os.path.exists(training_args.output_dir):
        # Find latest checkpoint
        checkpoints = [
            os.path.join(training_args.output_dir, d)
            for d in os.listdir(training_args.output_dir)
            if d.startswith("checkpoint-")
        ]
        if checkpoints:
            checkpoint = max(checkpoints, key=os.path.getctime)
            logger.info(f"Found checkpoint: {checkpoint}")
    
    # run training prcoess
    try:
        logger.info("Starting training...")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
        # Save final model on main process
        if global_rank == 0:
            logger.info("Saving final model...")
            trainer.save_model()
            trainer.save_state()
            
            # Save metrics
            metrics = train_result.metrics
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            
            logger.info("Training completed successfully!")
            logger.info(f"Final model saved to: {training_args.output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        return 1
    
    finally:
        # Cleanup distributed training
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
