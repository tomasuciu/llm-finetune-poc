import os
import logging
import json
import time
import threading
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import timedelta

import torch
import torch.distributed as dist
from transformers import Trainer

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig
    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False
    FSDP = None
    StateDictType = None
    FullStateDictConfig = None

logger = logging.getLogger(__name__)


def is_dist_functional() -> bool:
    """
    Check if distributed training is actually working.
    
    Returns:
        True if dist is initialized and functional
    """
    if not dist.is_initialized():
        return False
    try:
        _ = dist.get_world_size()
        _ = dist.get_rank()
        return True
    except Exception:
        return False


def safe_barrier(timeout_seconds: int = 60) -> bool:
    """Use monitored_barrier when available, fall back to a timed thread barrier."""
    if not is_dist_functional():
        return False
    try:
        from datetime import timedelta
        dist.monitored_barrier(timeout=timedelta(seconds=timeout_seconds))
        return True
    except Exception as e:
        logger.warning(f"monitored_barrier failed ({e}); falling back to threaded barrier")
        barrier_failed = []
        def _barrier_thread():
            try:
                dist.barrier()
                barrier_failed.append(False)
            except Exception as ee:
                logger.warning(f"Barrier failed: {ee}")
                barrier_failed.append(True)
        t = threading.Thread(target=_barrier_thread, daemon=True); t.start()
        t.join(timeout_seconds)
        if t.is_alive() or (barrier_failed and barrier_failed[0]):
            logger.error(f"Barrier timeout/failed after {timeout_seconds}s")
            return False
        return True


class ElasticCheckpointManager:
    """
    Manages elastic checkpoints with full state dict for dynamic world size support.
    
    Features:
    - Saves full state dict checkpoints periodically
    - Detects world size changes
    - Automatically loads appropriate checkpoint on resize
    - Maintains separate directory for elastic checkpoints
    - Validates checkpoint integrity
    - Robust error handling for node failures
    """
    
    def __init__(
        self,
        output_dir: str,
        elastic_checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 1000,
        save_on_resize: bool = True,
        keep_last_n: int = 2,
    ):
        """
        Initialize elastic checkpoint manager.
        
        Args:
            output_dir: Base output directory for training
            elastic_checkpoint_dir: Directory for elastic checkpoints (default: output_dir/elastic)
            checkpoint_interval: Steps between elastic checkpoints
            save_on_resize: Save checkpoint when world size changes
            keep_last_n: Number of elastic checkpoints to keep
        """
        self.output_dir = Path(output_dir)
        self.elastic_dir = Path(elastic_checkpoint_dir) if elastic_checkpoint_dir else self.output_dir / "elastic"
        self.checkpoint_interval = checkpoint_interval
        self.save_on_resize = save_on_resize
        self.keep_last_n = keep_last_n
        
        # State tracking
        self.last_world_size: Optional[int] = None
        self.last_elastic_checkpoint_step: Optional[int] = None
        
        # Create elastic checkpoint directory
        if self._is_rank_zero():
            self.elastic_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Elastic checkpoint directory: {self.elastic_dir}")
    
    @staticmethod
    def _is_rank_zero() -> bool:
        """Check if this is rank 0."""
        if not dist.is_initialized():
            return True
        try:
            return dist.get_rank() == 0
        except Exception:
            return True
    
    @staticmethod
    def _get_world_size() -> int:
        """Get current world size."""
        if not dist.is_initialized():
            return 1
        try:
            return dist.get_world_size()
        except Exception:
            return 1
    
    def should_save_checkpoint(self, step: int) -> bool:
        """
        Determine if elastic checkpoint should be saved at this step.
        
        Args:
            step: Current training step
            
        Returns:
            True if checkpoint should be saved
        """
        if self.checkpoint_interval <= 0:
            return False
        
        # Save at interval
        if step > 0 and step % self.checkpoint_interval == 0:
            return True
        
        return False
    
    def detect_world_size_change(self) -> bool:
        """
        Detect if world size has changed.
        
        Returns:
            True if world size changed since last check
        """
        current_world_size = self._get_world_size()
        
        if self.last_world_size is None:
            self.last_world_size = current_world_size
            return False
        
        changed = current_world_size != self.last_world_size
        
        if changed:
            logger.warning(
                f"World size changed: {self.last_world_size} -> {current_world_size}"
            )
            self.last_world_size = current_world_size
        
        return changed
    
    def save_elastic_checkpoint(
        self,
        trainer: Trainer,
        step: int,
        force: bool = False
    ) -> Optional[str]:
        """
        Save full state dict checkpoint for elastic resume.
        
        Args:
            trainer: HuggingFace Trainer instance
            step: Current training step
            force: Force save even if not at interval
            
        Returns:
            Path to saved checkpoint, or None if not saved
        """
        if not force and not self.should_save_checkpoint(step):
            return None
        
        checkpoint_dir = self.elastic_dir / f"elastic-checkpoint-{step}"
        
        if self._is_rank_zero():
            logger.info(f"Saving elastic checkpoint at step {step}...")
            start_time = time.time()
        
        try:
            # Synchronize before save - use safe barrier
            if is_dist_functional():
                logger.debug("Synchronizing before elastic checkpoint save...")
                if not safe_barrier(timeout_seconds=60):
                    logger.warning("Pre-save barrier failed - attempting save anyway")
            
            # Save using full state dict
            if FSDP_AVAILABLE and hasattr(trainer.model, '__class__') and \
               'FullyShardedDataParallel' in trainer.model.__class__.__name__:
                self._save_fsdp_full_checkpoint(trainer, checkpoint_dir, step)
            else:
                self._save_standard_checkpoint(trainer, checkpoint_dir, step)
            
            # Mark as committed
            if self._is_rank_zero():
                success_file = checkpoint_dir / "_ELASTIC_SUCCESS"
                success_file.touch()
                
                elapsed = time.time() - start_time
                logger.info(
                    f"Elastic checkpoint saved successfully in {elapsed:.2f}s: {checkpoint_dir}"
                )
            
            # Cleanup old checkpoints
            if self._is_rank_zero():
                self._cleanup_old_checkpoints()
            
            self.last_elastic_checkpoint_step = step
            
            # Synchronize after save - use safe barrier
            if is_dist_functional():
                logger.debug("Synchronizing after elastic checkpoint save...")
                if not safe_barrier(timeout_seconds=60):
                    logger.warning("Post-save barrier failed - checkpoint may be incomplete on some ranks")
            
            return str(checkpoint_dir)
            
        except Exception as e:
            logger.error(f"Failed to save elastic checkpoint: {e}", exc_info=True)
            
            # Try to sync even on error to prevent deadlock
            if is_dist_functional():
                logger.debug("Attempting barrier after save error...")
                safe_barrier(timeout_seconds=30)
            
            return None
    
    def _save_fsdp_full_checkpoint(
        self,
        trainer: Trainer,
        checkpoint_dir: Path,
        step: int
    ) -> None:
        """Save FSDP model with full state dict."""
        if self._is_rank_zero():
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Synchronize before save - with safety check
        if is_dist_functional():
            safe_barrier(timeout_seconds=60)
        
        try:
            # Configure FSDP to save full state dict on rank 0
            with FSDP.state_dict_type(
                trainer.model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            ):
                state_dict = trainer.model.state_dict()
            
            # Only rank 0 saves
            if self._is_rank_zero():
                # Save model state
                torch.save(
                    state_dict,
                    checkpoint_dir / "pytorch_model.bin"
                )
                
                # Save trainer state (includes step, epoch, optimizer info if available)
                trainer_state_path = checkpoint_dir / "trainer_state.json"
                trainer.save_state()
                
                # Copy trainer_state.json to checkpoint dir if not there
                default_trainer_state = trainer.args.output_dir + "/trainer_state.json"
                if os.path.exists(default_trainer_state) and not trainer_state_path.exists():
                    import shutil
                    shutil.copy2(default_trainer_state, trainer_state_path)
                
                # Save metadata
                metadata = {
                    'step': step,
                    'world_size': self._get_world_size(),
                    'elastic_checkpoint': True,
                    'timestamp': time.time(),
                    'epoch': trainer.state.epoch if hasattr(trainer, 'state') else 0,
                }
                
                with open(checkpoint_dir / "elastic_metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Save config
                if hasattr(trainer.model, 'config'):
                    trainer.model.config.save_pretrained(checkpoint_dir)
                
                logger.info(f"Saved full state dict from world size {self._get_world_size()}")
        
        except Exception as e:
            logger.error(f"Error saving FSDP checkpoint: {e}", exc_info=True)
            raise
        
        finally:
            # Always synchronize after save attempt - with safety check
            if is_dist_functional():
                safe_barrier(timeout_seconds=60)
    
    def _save_standard_checkpoint(
        self,
        trainer: Trainer,
        checkpoint_dir: Path,
        step: int
    ) -> None:
        """Save standard PyTorch checkpoint."""
        if self._is_rank_zero():
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            torch.save(
                trainer.model.state_dict(),
                checkpoint_dir / "pytorch_model.bin"
            )
            
            # Save trainer state
            trainer_state_path = checkpoint_dir / "trainer_state.json"
            trainer.save_state()
            
            # Copy trainer_state.json if needed
            default_trainer_state = trainer.args.output_dir + "/trainer_state.json"
            if os.path.exists(default_trainer_state) and not trainer_state_path.exists():
                import shutil
                shutil.copy2(default_trainer_state, trainer_state_path)
            
            # Save metadata
            metadata = {
                'step': step,
                'world_size': self._get_world_size(),
                'elastic_checkpoint': True,
                'timestamp': time.time(),
                'epoch': trainer.state.epoch if hasattr(trainer, 'state') else 0,
            }
            
            with open(checkpoint_dir / "elastic_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save config
            if hasattr(trainer.model, 'config'):
                trainer.model.config.save_pretrained(checkpoint_dir)
    
    def find_latest_elastic_checkpoint(self) -> Optional[str]:
        """
        Find the most recent valid elastic checkpoint.
        
        Returns:
            Path to latest checkpoint, or None if none found
        """
        if not self.elastic_dir.exists():
            return None
        
        checkpoints = []
        for item in self.elastic_dir.iterdir():
            if item.is_dir() and item.name.startswith("elastic-checkpoint-"):
                if self._validate_elastic_checkpoint(item):
                    checkpoints.append(item)
        
        if not checkpoints:
            return None
        
        # Sort by modification time (most recent first)
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        latest = str(checkpoints[0])
        logger.info(f"Found latest elastic checkpoint: {latest}")
        return latest
    
    def _validate_elastic_checkpoint(self, checkpoint_path: Path) -> bool:
        """
        Validate elastic checkpoint integrity.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            
        Returns:
            True if checkpoint is valid
        """
        required_files = [
            "pytorch_model.bin",
            "config.json",
            "elastic_metadata.json",
            "_ELASTIC_SUCCESS",
        ]
        
        for filename in required_files:
            if not (checkpoint_path / filename).exists():
                logger.warning(f"Checkpoint {checkpoint_path} missing {filename}")
                return False
        
        return True
    
    def load_elastic_checkpoint(
        self,
        trainer: Trainer,
        checkpoint_path: str
    ) -> bool:
        """
        Load elastic checkpoint and handle world size changes.
        
        Args:
            trainer: HuggingFace Trainer instance
            checkpoint_path: Path to elastic checkpoint
            
        Returns:
            True if successfully loaded
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not self._validate_elastic_checkpoint(checkpoint_path):
            logger.error(f"Invalid elastic checkpoint: {checkpoint_path}")
            return False
        
        try:
            logger.info(f"Loading elastic checkpoint from {checkpoint_path}")
            
            # Load metadata
            with open(checkpoint_path / "elastic_metadata.json", 'r') as f:
                metadata = json.load(f)
            
            old_world_size = metadata['world_size']
            current_world_size = self._get_world_size()
            step = metadata['step']
            
            logger.info(
                f"Loading checkpoint from world size {old_world_size} "
                f"to world size {current_world_size} at step {step}"
            )
            
            # Synchronize before loading - CRITICAL for elastic training
            if is_dist_functional():
                logger.info("Synchronizing all ranks before loading elastic checkpoint...")
                if not safe_barrier(timeout_seconds=120):
                    logger.error("Pre-load barrier failed - this may cause issues")
                    # Continue anyway - better to try than to fail
            
            # Load model state
            logger.info("Loading model state dict...")
            state_dict = torch.load(
                checkpoint_path / "pytorch_model.bin",
                map_location="cpu"
            )
            
            # Load into model (FSDP will automatically reshard if needed)
            if FSDP_AVAILABLE and hasattr(trainer.model, '__class__') and \
               'FullyShardedDataParallel' in trainer.model.__class__.__name__:
                logger.info("Loading FSDP model with automatic resharding...")
                with FSDP.state_dict_type(
                    trainer.model,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
                ):
                    # Load on all ranks for FSDP (rank0_only=False is CRITICAL!)
                    missing_keys, unexpected_keys = trainer.model.load_state_dict(state_dict, strict=False)
                    if missing_keys:
                        logger.warning(f"Missing keys when loading checkpoint: {missing_keys[:5]}...")
                    if unexpected_keys:
                        logger.warning(f"Unexpected keys when loading checkpoint: {unexpected_keys[:5]}...")
            else:
                logger.info("Loading standard model state dict...")
                trainer.model.load_state_dict(state_dict, strict=False)
            
            # CRITICAL: Load trainer state to sync step counter, epoch, etc.
            trainer_state_path = checkpoint_path / "trainer_state.json"
            if trainer_state_path.exists():
                logger.info("Loading trainer state...")
                with open(trainer_state_path, 'r') as f:
                    trainer_state_dict = json.load(f)
                
                # Update trainer's state
                if hasattr(trainer, 'state'):
                    trainer.state.global_step = trainer_state_dict.get('global_step', step)
                    trainer.state.epoch = trainer_state_dict.get('epoch', 0)
                    trainer.state.best_metric = trainer_state_dict.get('best_metric', None)
                    trainer.state.best_model_checkpoint = trainer_state_dict.get('best_model_checkpoint', None)
                    
                    logger.info(
                        f"Synced trainer state: step={trainer.state.global_step}, "
                        f"epoch={trainer.state.epoch}"
                    )
            else:
                logger.warning(f"trainer_state.json not found in {checkpoint_path}")
                # Fallback: at least set the global step
                if hasattr(trainer, 'state'):
                    trainer.state.global_step = step
                    logger.info(f"Set trainer.state.global_step to {step} (from metadata)")
            
            # Note: Optimizer state is reset when world size changes
            if old_world_size != current_world_size:
                logger.warning(
                    f"World size changed ({old_world_size} -> {current_world_size}). "
                    "Optimizer and LR scheduler state will be reset."
                )
                # The trainer will reinitialize optimizer and scheduler with new world size
            else:
                logger.info("World size unchanged, optimizer state may be preserved")
            
            # Update tracking
            self.last_world_size = current_world_size
            self.last_elastic_checkpoint_step = step
            
            # Synchronize all ranks after loading - CRITICAL
            if is_dist_functional():
                logger.info("Synchronizing all ranks after elastic checkpoint load...")
                if not safe_barrier(timeout_seconds=120):
                    logger.error("Post-load barrier failed - ranks may be out of sync")
                    # Continue anyway
            
            logger.info("Elastic checkpoint loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load elastic checkpoint: {e}", exc_info=True)
            
            # Try to synchronize even on error to avoid deadlock
            if is_dist_functional():
                logger.info("Attempting barrier after load error to prevent deadlock...")
                safe_barrier(timeout_seconds=30)
            
            return False
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old elastic checkpoints, keeping only the last N."""
        if not self._is_rank_zero():
            return
        
        if not self.elastic_dir.exists():
            return
        
        checkpoints = []
        for item in self.elastic_dir.iterdir():
            if item.is_dir() and item.name.startswith("elastic-checkpoint-"):
                checkpoints.append(item)
        
        if len(checkpoints) <= self.keep_last_n:
            return
        
        # Sort by modification time
        checkpoints.sort(key=lambda x: x.stat().st_mtime)
        
        # Remove oldest
        for checkpoint in checkpoints[:-self.keep_last_n]:
            try:
                import shutil
                shutil.rmtree(checkpoint)
                logger.info(f"Removed old elastic checkpoint: {checkpoint.name}")
            except Exception as e:
                logger.warning(f"Failed to remove {checkpoint}: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get elastic checkpoint statistics.
        
        Returns:
            Dictionary with checkpoint statistics
        """
        stats = {
            'elastic_dir': str(self.elastic_dir),
            'checkpoint_interval': self.checkpoint_interval,
            'last_checkpoint_step': self.last_elastic_checkpoint_step,
            'current_world_size': self._get_world_size(),
            'last_world_size': self.last_world_size,
            'num_checkpoints': 0,
            'total_size_mb': 0,
        }
        
        if self.elastic_dir.exists():
            checkpoints = [
                item for item in self.elastic_dir.iterdir()
                if item.is_dir() and item.name.startswith("elastic-checkpoint-")
            ]
            stats['num_checkpoints'] = len(checkpoints)
            
            # Calculate total size
            total_size = 0
            for checkpoint in checkpoints:
                for file in checkpoint.rglob("*"):
                    if file.is_file():
                        total_size += file.stat().st_size
            stats['total_size_mb'] = total_size / (1024 * 1024)
        
        return stats
