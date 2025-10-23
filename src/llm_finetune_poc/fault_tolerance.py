import logging
import shutil
import signal
import threading
import os

import torch.distributed as dist
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


def is_rank_zero() -> bool:
    return int(os.environ.get("RANK", "0")) == 0

PREEMPT_SAVE_REQUEST = threading.Event()


class FaultToleranceCallback(TrainerCallback):
    
    def on_train_begin(self, args, state, control, **kwargs):
        logger.info("Training started successfully")
        
    def on_step_end(self, args, state, control, **kwargs):
        # Log every 10 steps to avoid spam
        if state.global_step % 10 == 0:
            logger.info(f"Completed step {state.global_step}/{state.max_steps}")
    
    def on_save(self, args, state, control, **kwargs):
        logger.info(f"Checkpoint saved at step {state.global_step}")

        # Only rank 0 marks the checkpoint as committed
        if int(os.environ.get("RANK", "0")) == 0 and state.global_step is not None:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}") # type: ignore
            try:
                with open(os.path.join(ckpt_dir, "_SUCCESS"), "w") as f:
                    f.write("ok\n")
            except Exception as e:
                logger.warning(f"Failed to write _SUCCESS in {ckpt_dir}: {e}")
        
    def on_train_end(self, args, state, control, **kwargs):
        logger.info("Training completed successfully")


def handle_sigusr1(signum, frame):
    """
    Preemption-friendly path: ask the training loop to save at a safe point.
    No heavy work in the signal context.
    """
    rank = int(os.environ.get("RANK", "0"))
    logger.warning(f"Received SIGUSR1 on rank {rank}: will request a checkpoint at next step boundary")
    PREEMPT_SAVE_REQUEST.set()


def signal_handler(signum, frame):
    """
    Signal handler that raises exception for torchelastic recovery.
    """
    rank = int(os.environ.get("RANK", "0"))
    
    if signum == signal.SIGTERM:
        logger.error(f"Received SIGTERM on rank {rank} - triggering elastic recovery")
        # Immediately raise for elastic recovery - no coordination
        raise RuntimeError(f"SIGTERM received on rank {rank}")
    elif signum == signal.SIGINT:
        logger.error(f"Received SIGINT on rank {rank}")
        raise KeyboardInterrupt(f"SIGINT received on rank {rank}")
    else:
        raise RuntimeError(f"Signal {signum} received on rank {rank}")
    

def is_committed_checkpoint(path: str) -> bool:
    return os.path.exists(os.path.join(path, "_SUCCESS"))


def is_elastic_checkpoint(path: str) -> bool:
    """Check if checkpoint is an elastic checkpoint."""
    return os.path.exists(os.path.join(path, "_ELASTIC_SUCCESS"))


def validate_checkpoint(checkpoint_path: str, require_elastic: bool = False) -> bool:
    """
    Validate that a checkpoint is complete and committed.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        require_elastic: If True, only validate elastic checkpoints
        
    Returns:
        True if checkpoint is valid
    """
    required_files = ['trainer_state.json', 'config.json']
    
    if not os.path.exists(checkpoint_path):
        return False
    
    for file in required_files:
        if not os.path.exists(os.path.join(checkpoint_path, file)):
            logger.warning(f"Checkpoint {checkpoint_path} missing {file}")
            return False

    # model files: either a single file or shards
    has_single = any(os.path.exists(os.path.join(checkpoint_path, f))
                     for f in ["pytorch_model.bin", "model.safetensors"])
    has_shards = any(name.startswith(("pytorch_model-", "model-")) 
                     for name in os.listdir(checkpoint_path))
    if not (has_single or has_shards):
        logger.warning(f"Checkpoint {checkpoint_path} missing model weights")
        return False
    
    # Check appropriate success marker
    if require_elastic:
        if not is_elastic_checkpoint(checkpoint_path):
            logger.warning(f"Checkpoint {checkpoint_path} not marked as elastic")
            return False
    else:
        if not is_committed_checkpoint(checkpoint_path):
            logger.warning(f"Checkpoint {checkpoint_path} not committed (_SUCCESS missing)")
            return False
    
    return True


def cleanup_old_checkpoints(output_dir: str, keep_last_n: int = 3, exclude_elastic: bool = True) -> None:
    """
    Remove old checkpoints, keeping only the last N.
    
    Args:
        output_dir: Directory containing checkpoints
        keep_last_n: Number of checkpoints to keep
        exclude_elastic: If True, don't cleanup elastic checkpoints (managed separately)
    """
    checkpoints = [
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]
    
    # Exclude elastic checkpoints if requested
    if exclude_elastic:
        checkpoints = [
            cp for cp in checkpoints
            if not (os.path.basename(cp).startswith("elastic-") or is_elastic_checkpoint(cp))
        ]
    
    if len(checkpoints) <= keep_last_n:
        return
    
    # Sort by creation time
    checkpoints.sort(key=os.path.getctime)
    
    # Remove oldest checkpoints
    for checkpoint in checkpoints[:-keep_last_n]:
        try:
            shutil.rmtree(checkpoint)
            logger.info(f"Removed old checkpoint: {checkpoint}")
        except Exception as e:
            logger.warning(f"Failed to remove checkpoint {checkpoint}: {e}")


def get_checkpoint_type(checkpoint_path: str) -> str:
    """
    Determine checkpoint type.
    
    Args:
        checkpoint_path: Path to checkpoint
        
    Returns:
        'elastic', 'regular', or 'unknown'
    """
    if not os.path.exists(checkpoint_path):
        return 'unknown'
    
    if is_elastic_checkpoint(checkpoint_path):
        return 'elastic'
    elif is_committed_checkpoint(checkpoint_path):
        return 'regular'
    else:
        return 'unknown'
