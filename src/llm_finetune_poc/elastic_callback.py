import logging
import os
import weakref
from typing import Optional

import torch
import torch.distributed as dist
from transformers import TrainerCallback, TrainerState, TrainerControl

from llm_finetune_poc.elastic_checkpoint import ElasticCheckpointManager, safe_barrier, is_dist_functional
from llm_finetune_poc.fault_tolerance import PREEMPT_SAVE_REQUEST

logger = logging.getLogger(__name__)


class ElasticTrainingCallback(TrainerCallback):
    """
    Callback for elastic training with dynamic world size support.
    
    Features:
    - Monitors world size changes
    - Saves periodic elastic checkpoints
    - Triggers checkpoint save on world size change
    - Provides elastic training statistics
    - Robust error handling for node failures
    """
    
    def __init__(
        self,
        elastic_manager: ElasticCheckpointManager,
        enabled: bool = True,
    ):
        """
        Initialize elastic training callback.
        
        Args:
            elastic_manager: ElasticCheckpointManager instance
            enabled: Whether elastic training is enabled
        """
        self.elastic_manager = elastic_manager
        self.enabled = enabled
        self.world_size_changes = []
        self._trainer_ref = None
        
        if not self.enabled:
            logger.info("Elastic training disabled - using standard checkpoints only")
        else:
            logger.info("Elastic training enabled - hybrid checkpoint strategy active")
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        if not self.enabled:
            return
        
        # Initialize world size tracking
        self.elastic_manager.detect_world_size_change()
        
        # Try to resume from elastic checkpoint if world size changed
        elastic_checkpoint = self.elastic_manager.find_latest_elastic_checkpoint()
        
        if elastic_checkpoint and state.global_step == 0:
            # Check if we should load from elastic checkpoint
            # (e.g., if world size is different from last run)
            logger.info("Elastic checkpoint available at training start")
            # Note: Actual loading is handled in train.py to avoid conflicts
    
    def _get_trainer_from_kwargs(self, **kwargs):
        """
        Extract trainer from callback kwargs.
        
        HuggingFace Trainer passes itself to callbacks under different keys
        depending on the callback method. This helper tries all known locations.
        
        Returns:
            Trainer instance or None
        """
        # Most common: 'model' key (despite the name, it's actually the trainer)
        trainer = kwargs.get('model')
        if trainer is not None and hasattr(trainer, 'save_state'):
            return trainer
        
        # Try other common keys
        for key in ['trainer', 'self']:
            trainer = kwargs.get(key)
            if trainer is not None and hasattr(trainer, 'save_state'):
                return trainer
        
        # Search through all kwargs for something that looks like a trainer
        for value in kwargs.values():
            if hasattr(value, 'save_state') and hasattr(value, 'model'):
                return value
        
        return self._trainer_ref() if self._trainer_ref else None


    def set_trainer(self, trainer):
        """Attach a Trainer instance so we can save elastic checkpoints."""
        self._trainer_ref = weakref.ref(trainer)
    
    def on_step_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each training step."""
        if not self.enabled:
            return
        
        # Detect world size changes
        if self.elastic_manager.detect_world_size_change():
            # World size changed!
            change_info = {
                'step': state.global_step,
                'old_world_size': self.elastic_manager.last_world_size,
                'new_world_size': self.elastic_manager._get_world_size(),
            }
            self.world_size_changes.append(change_info)
            
            logger.warning(
                f"World size change detected at step {state.global_step}: "
                f"{change_info['old_world_size']} -> {change_info['new_world_size']}"
            )
            
            # Save checkpoint on resize if configured
            if self.elastic_manager.save_on_resize:
                logger.info("Saving elastic checkpoint due to world size change...")
                # Get trainer from kwargs
                trainer = self._get_trainer_from_kwargs(**kwargs)
                self._save_elastic_checkpoint_if_needed(state, force=True, trainer=trainer, **kwargs)
    

    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step."""
        if not self.enabled:
            return
         
        # Existing periodic elastic checkpointing:
        trainer = self._get_trainer_from_kwargs(**kwargs)
        self._save_elastic_checkpoint_if_needed(state, force=False, trainer=trainer, **kwargs)
    
    def _save_elastic_checkpoint_if_needed(
        self,
        state: TrainerState,
        force: bool = False,
        trainer=None,
        **kwargs
    ) -> None:
        """
        Save elastic checkpoint if conditions are met.
        
        Args:
            state: TrainerState instance
            force: Force save regardless of interval
            trainer: Trainer instance (passed explicitly or via kwargs)
            **kwargs: Additional arguments from callback
        """
        # Try to get trainer from multiple sources
        if trainer is None:
            trainer = self._get_trainer_from_kwargs(**kwargs)
        
        if not trainer or not hasattr(trainer, 'save_state'):
            logger.debug("Trainer not available for elastic checkpoint save")
            return
        
        # Check if we should save
        if force or self.elastic_manager.should_save_checkpoint(state.global_step):
            try:
                checkpoint_path = self.elastic_manager.save_elastic_checkpoint(
                    trainer=trainer,
                    step=state.global_step,
                    force=force
                )
                
                if checkpoint_path:
                    logger.info(f"Elastic checkpoint saved: {checkpoint_path}")
            except Exception as e:
                logger.error(f"Failed to save elastic checkpoint: {e}", exc_info=True)
                # Don't raise - we don't want to stop training for checkpoint failures
    
    def on_save(self, args, state, control, **kwargs):
        """
        Called when a checkpoint is saved.
        
        Note: This is called for regular (sharded) checkpoints.
        Elastic checkpoints are saved separately via on_step_end.
        """
        if not self.enabled:
            return
        
        # Regular checkpoint saved - log for tracking
        if self.elastic_manager._is_rank_zero():
            logger.debug(
                f"Regular checkpoint at step {state.global_step}. "
                f"Last elastic checkpoint at step {self.elastic_manager.last_elastic_checkpoint_step}"
            )
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        if not self.enabled:
            return
        
        # Piggy-back: we can opportunistically save an elastic checkpoint
        # at the same cadence (respecting the interval logic).
        trainer = self._get_trainer_from_kwargs(**kwargs)


        if trainer and hasattr(trainer, 'save_state'):
            logger.info("Saving final elastic checkpoint...")
            try:
                self.elastic_manager.save_elastic_checkpoint(
                    trainer=trainer,
                    step=state.global_step,
                    force=True
                )
            except Exception as e:
                logger.error(f"Failed to save final elastic checkpoint: {e}", exc_info=True)
        
        # Log statistics
        if self.elastic_manager._is_rank_zero():
            stats = self.elastic_manager.get_statistics()
            logger.info("=" * 80)
            logger.info("Elastic Training Statistics:")
            logger.info("=" * 80)
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")
            logger.info(f"  world_size_changes: {len(self.world_size_changes)}")
            if self.world_size_changes:
                logger.info("  World size change events:")
                for change in self.world_size_changes:
                    logger.info(f"    Step {change['step']}: {change['old_world_size']} -> {change['new_world_size']}")
            logger.info("=" * 80)


class ElasticWorldSizeMonitor(TrainerCallback):
    """
    Lightweight callback to monitor world size changes.
    
    Use this when elastic training is disabled but you still want
    to track world size changes for debugging.
    """
    
    def __init__(self):
        self.last_world_size: Optional[int] = None
        self.changes = []
    
    @staticmethod
    def _get_world_size() -> int:
        """Get current world size."""
        if not dist.is_initialized():
            return 1
        try:
            return dist.get_world_size()
        except Exception:
            return 1
    
    def on_step_begin(self, args, state, control, **kwargs):
        """Monitor world size at each step."""
        current_ws = self._get_world_size()
        
        if self.last_world_size is None:
            self.last_world_size = current_ws
            return
        
        if current_ws != self.last_world_size:
            change = {
                'step': state.global_step,
                'old': self.last_world_size,
                'new': current_ws,
            }
            self.changes.append(change)
            logger.warning(
                f"World size changed at step {state.global_step}: "
                f"{self.last_world_size} -> {current_ws}"
            )
            self.last_world_size = current_ws
    
    def on_train_end(self, args, state, control, **kwargs):
        """Log final statistics."""
        if self.changes:
            logger.info("=" * 80)
            logger.info(f"Detected {len(self.changes)} world size changes during training:")
            for change in self.changes:
                logger.info(f"  Step {change['step']}: {change['old']} -> {change['new']}")
            logger.info("=" * 80)
