import os
import sys
import signal
import logging
import functools
from typing import Dict, List
from datetime import timedelta

import torch
import torch.distributed as dist
import transformers
from transformers import (
    TrainingArguments,
    HfArgumentParser,
    Trainer,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from datasets import load_dataset
from torch.distributed.elastic.multiprocessing.errors import record

from llm_finetune_poc.arguments import (
    ModelArguments,
    DataArguments,
    DistributedArguments,
    CustomTrainingArguments,
)
from llm_finetune_poc.fault_tolerance import (
    FaultToleranceCallback,
    signal_handler,
    handle_sigusr1,
    validate_checkpoint,
    cleanup_old_checkpoints,
    is_rank_zero,
)
from llm_finetune_poc.elastic_callback import ElasticTrainingCallback, ElasticWorldSizeMonitor
from llm_finetune_poc.elastic_checkpoint import ElasticCheckpointManager, safe_barrier, is_dist_functional

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class EpochLogger(TrainerCallback):
    """Callback for cleaner epoch logging"""
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        if is_rank_zero():
            # state.epoch is the epoch that's starting (0-indexed during training)
            current_epoch = int(state.epoch) + 1 if state.epoch is not None else 1
            logger.info(f"========== Epoch {current_epoch} Started ==========")

    def on_epoch_end(self, args, state, control, **kwargs):
        if is_rank_zero():
            # state.epoch is the epoch that just finished
            current_epoch = int(state.epoch) if state.epoch is not None else 1
            logger.info(f"========== Epoch {current_epoch} Completed ==========")


def preprocess_function_calling_dataset(
    examples: Dict[str, List],
    tokenizer: AutoTokenizer,
    max_seq_length: int,
) -> Dict[str, List]:
    processed_texts = []

    for idx in range(len(examples['system'])):
        # for models without chat template, use concatenation
        if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
            text = ""
            if examples['system'][idx]:
                text += examples['system'][idx] + "\n\n"
            if examples['chat'][idx]:
                text += examples['chat'][idx]
            processed_texts.append(text)
        else:
            conversation = []
            
            # Add system message if present
            if examples['system'][idx]:
                conversation.append({
                    "role": "system",
                    "content": examples['system'][idx]
                })
            
            # Add chat messages
            if examples['chat'][idx]:
                conversation.append({
                    "role": "user", 
                    "content": examples['chat'][idx]
                })
            
            # Apply chat template
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
        use_fast=True,
    )
    
    # pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if is_rank_zero():
        logger.info(f"Loading model: {model_args.model_name_or_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
        attn_implementation="flash_attention_2" if model_args.use_flash_attention_2 else None,
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
    if is_rank_zero():
        logger.info(f"Loading dataset: {data_args.dataset_name}")
    
    try:
        # Load dataset
        dataset = load_dataset(
            data_args.dataset_name,
            split=data_args.dataset_split,
        )
        
        if is_rank_zero():
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
            num_proc=data_args.preprocessing_num_workers,   # type: ignore
            remove_columns=train_dataset.column_names,      # type: ignore
            desc="Tokenizing training data",                # type: ignore
        )
        
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(
                preprocess_fn,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=eval_dataset.column_names,
                desc="Tokenizing validation data",
            )
        
        if is_rank_zero():
            logger.info(f"Training examples: {len(train_dataset)}")
        if eval_dataset:

            if is_rank_zero():
                logger.info(f"Validation examples: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def setup_elastic_training(
    training_args: TrainingArguments,
    distributed_args: DistributedArguments,
) -> tuple[ElasticCheckpointManager | None, ElasticTrainingCallback | None]:
    """
    Setup elastic training components if enabled.
    
    Args:
        training_args: Training configuration
        distributed_args: Distributed training configuration
        
    Returns:
        Tuple of (elastic_manager, elastic_callback) or (None, None) if disabled
    """
    if not distributed_args.enable_elastic_training:
        if is_rank_zero():
            logger.info("Elastic training disabled - using standard sharded checkpoints only")
        return None, None
    
    if is_rank_zero():
        logger.info("=" * 80)
        logger.info("Elastic Training Configuration:")
        logger.info("=" * 80)
        logger.info(f"  Enabled: {distributed_args.enable_elastic_training}")
        logger.info(f"  Checkpoint interval: {distributed_args.elastic_checkpoint_interval} steps")
        logger.info(f"  Min nodes: {distributed_args.elastic_min_nodes}")
        logger.info(f"  Max nodes: {distributed_args.elastic_max_nodes}")
        logger.info(f"  Save on resize: {distributed_args.elastic_save_on_resize}")
        logger.info(f"  Checkpoint dir: {distributed_args.elastic_checkpoint_dir or 'output_dir/elastic'}")
        logger.info("=" * 80)
    
    # Initialize elastic checkpoint manager
    elastic_manager = ElasticCheckpointManager(
        output_dir=training_args.output_dir,
        elastic_checkpoint_dir=distributed_args.elastic_checkpoint_dir,
        checkpoint_interval=distributed_args.elastic_checkpoint_interval,
        save_on_resize=distributed_args.elastic_save_on_resize,
        keep_last_n=2,  # Keep last 2 elastic checkpoints
    )
    
    # Initialize elastic callback
    elastic_callback = ElasticTrainingCallback(
        elastic_manager=elastic_manager,
        enabled=True,
    )
    
    return elastic_manager, elastic_callback


@record
def main():
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)   # restart-triggering path
    signal.signal(signal.SIGUSR1, handle_sigusr1)


    # Parse command-line arguments and/or configuration file
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, CustomTrainingArguments, DistributedArguments) # type: ignore
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Load from JSON
        model_args, data_args, training_args, distributed_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, distributed_args = parser.parse_args_into_dataclasses()

    transformers.set_seed(training_args.seed)
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Create process group if not already done
    if local_rank != -1 and not dist.is_initialized():

        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "3")
        os.environ.setdefault("TORCH_NCCL_DEBUG", "WARN")
        os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "1")
        os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "120")
        os.environ.pop("TORCH_NCCL_BLOCKING_WAIT", None)

        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=distributed_args.process_group_timeout)
        )
        # Fence all ranks after relaunch before any DDP/accelerate collective
        try:
            dist.monitored_barrier(timeout=timedelta(seconds=120))
        except Exception as e:
            logger.warning(f"Post-init monitored_barrier failed: {e}")
    
    logging.info(f"Rank {global_rank}/{world_size}, Local rank: {local_rank}")

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

        if is_rank_zero():
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
    
    # FSDP configuration
    if world_size > 1:
        # For elastic training, we need to be able to save/load full state dict
        # But we still use sharded state dict for regular checkpoints (fast)
        training_args.fsdp = "full_shard auto_wrap"
        training_args.fsdp_transformer_layer_cls_to_wrap = None
        training_args.fsdp_config = {
            "xla": False,
            "state_dict_type": "SHARDED_STATE_DICT",
            "use_orig_params": True,
            "sync_module_states": True,
            "forward_prefetch": True,
            "limit_all_gathers": True,
        }

        if is_rank_zero():
            logger.info(f"FSDP enabled with config: {training_args.fsdp_config}")

    
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Setup elastic training if enabled
    elastic_manager, elastic_callback = setup_elastic_training(
        training_args,
        distributed_args,
    )
    
    if is_rank_zero():
        logger.info("Initializing Trainer...")

    callbacks = [
        FaultToleranceCallback(),
        EpochLogger(),
    ]
    
    # Add elastic training callback or world size monitor
    if elastic_callback is not None:
        callbacks.append(elastic_callback)
    else:
        # Still monitor world size changes even if elastic training is disabled
        callbacks.append(ElasticWorldSizeMonitor())

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset, # type: ignore
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # give the callback a handle to the Trainer so it can save elastic checkpoints
    if elastic_callback is not None:
        elastic_callback.set_trainer(trainer)


    # Resume from checkpoint if available
    checkpoint = None
    resume_from_step = None
    
    if elastic_manager is not None:
        elastic_checkpoint = elastic_manager.find_latest_elastic_checkpoint()
        if elastic_checkpoint:
            logger.info(f"Found elastic checkpoint: {elastic_checkpoint}")
            
            # Synchronize before loading elastic checkpoint - use safe barrier
            if world_size > 1 and is_dist_functional():
                logger.info("Synchronizing before elastic checkpoint load...")
                if not safe_barrier(timeout_seconds=120):
                    logger.warning("Pre-load barrier failed - proceeding with caution")
            
            # Load elastic checkpoint (handles world size changes)
            if elastic_manager.load_elastic_checkpoint(trainer, elastic_checkpoint):
                checkpoint = None  # Don't load regular checkpoint
                # Get the step from elastic checkpoint metadata
                metadata_path = os.path.join(elastic_checkpoint, "elastic_metadata.json")
                if os.path.exists(metadata_path):
                    import json
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    resume_from_step = metadata.get('step', 0)
                    logger.info(f"Resumed from elastic checkpoint at step {resume_from_step}")
                
                # Update trainer state to ensure proper resumption
                if hasattr(trainer, 'state') and resume_from_step is not None:
                    trainer.state.global_step = resume_from_step
                    logger.info(f"Set trainer.state.global_step to {resume_from_step}")
                
                # Synchronize after loading - use safe barrier
                if world_size > 1 and is_dist_functional():
                    logger.info("Synchronizing after elastic checkpoint load...")
                    if not safe_barrier(timeout_seconds=120):
                        logger.warning("Post-load barrier failed - ranks may be out of sync")
            else:
                logger.warning("Failed to load elastic checkpoint, will try regular checkpoint")
    
    # Priority 2: Try regular checkpoint if elastic not loaded
    if checkpoint is None and resume_from_step is None:
        if training_args.resume_from_checkpoint is not None:
            if validate_checkpoint(training_args.resume_from_checkpoint):
                checkpoint = training_args.resume_from_checkpoint
                logger.info(f"Resuming from specified checkpoint: {checkpoint}")
            else:
                logger.warning(f"Invalid checkpoint: {training_args.resume_from_checkpoint}")
        
        elif os.path.exists(training_args.output_dir):
            checkpoints = [
                os.path.join(training_args.output_dir, d)
                for d in os.listdir(training_args.output_dir)
                if d.startswith("checkpoint-") and not d.startswith("elastic-")
            ]
            
            # Filter to valid checkpoints only
            valid_checkpoints = [cp for cp in checkpoints if validate_checkpoint(cp)]
            
            if valid_checkpoints:
                checkpoint = max(valid_checkpoints, key=os.path.getctime)
                logger.info(f"Found valid checkpoint: {checkpoint}")

            elif checkpoints:
                logger.warning(f"Found {len(checkpoints)} checkpoints but none are valid")

                # Clean up invalid checkpoints
                for cp in checkpoints:
                    if not validate_checkpoint(cp):
                        logger.info(f"Removing invalid checkpoint: {cp}")
                        import shutil
                        try:
                            shutil.rmtree(cp)
                        except Exception as e:
                            logger.warning(f"Failed to remove {cp}: {e}")
    
    # Log resume status
    if resume_from_step is not None:
        logger.info(f"Training will resume from step {resume_from_step} (elastic checkpoint)")
    elif checkpoint is not None:
        logger.info(f"Training will resume from checkpoint: {checkpoint}")
    else:
        logger.info("Starting training from scratch")
    
    # Training Loop
    try:
        logger.info("Starting training...")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        # If we hit a cooperative preemption, abort non-zero to trigger elastic restart
        if os.environ.get("REQUESTED_ABORT") == "1":
            logger.warning("Elastic preemption requested; aborting to trigger restart")
            raise RuntimeError("ELASTIC_PREEMPT")
       
        # Save final model and state only on main process
        # Ensure everyone finished training before we start any rank-0 I/O
        if world_size > 1 and is_dist_functional():
            try:
                logger.info(f"Rank {global_rank}: Waiting for all ranks before final save...")
                trainer.accelerator.wait_for_everyone()
            except Exception as e:
                logger.warning(f"wait_for_everyone failed: {e}")
                # Try safe barrier instead
                safe_barrier(timeout_seconds=60)

        # Save final model and state only on main process
        if global_rank == 0:
            logger.info("Saving final model...")
            trainer.save_model()
            trainer.save_state()
            
            metrics = train_result.metrics
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            
            # Cleanup old checkpoints
            cleanup_old_checkpoints(
                training_args.output_dir,
                keep_last_n=3,
                exclude_elastic=True
            )
            
            logger.info("Training completed successfully!")
            logger.info(f"Final model saved to: {training_args.output_dir}")

        # Without this barrier, other ranks will destroy the process group while rank 0 is still saving
        if world_size > 1 and is_dist_functional():
            logger.info(f"Rank {global_rank}: Waiting at final barrier before cleanup...")

            # Use safe barrier with timeout
            success = safe_barrier(timeout_seconds=120)
            if success:
                logger.info(f"Rank {global_rank}: Passed final barrier, ready to exit")
            else:
                logger.warning(f"Rank {global_rank}: Final barrier failed - proceeding with exit anyway")
        
        return 0
    
    except RuntimeError as e:
        if "Signal" in str(e) or "SIGTERM" in str(e):
            logger.error(f"Training interrupted by signal: {e}")
            # Return non-zero to trigger elastic restart
            return 143  # Standard SIGTERM exit code
    
    # Graceful distributed cleanup
    #finally:
    #    if is_dist_functional():
    #        try:
    #            logger.info(f"Rank {global_rank}: Destroying process group...")
    #            dist.destroy_process_group()
    #            logger.info(f"Rank {global_rank}: Process group destroyed")
    #        except Exception as e:
    #            logger.warning(f"Rank {global_rank}: Cleanup error (non-fatal): {e}")
    #    else:
    #        logger.info(f"Rank {global_rank}: Distributed not functional, skipping process group cleanup")
    #    
    #    # Force flush outputs
    #    logger.info(f"Rank {global_rank}: Exiting cleanly")
    #    sys.stdout.flush()
    #    sys.stderr.flush()


if __name__ == "__main__":
    sys.exit(main())
