from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Optional,
)

@dataclass
class ModelConfig:
    model_name: str = "mistralai/Mistral-7B-v0.1"
    tokenizer_name: Optional[str] = None
    use_flash_attention: bool = True
    max_seq_length: int = 2048
    

@dataclass
class DataConfig:
    dataset_name: str = "glaiveai/glaive-function-calling-v2"
    dataset_split: str = "train"
    validation_split_percentage: int = 5
    preprocessing_num_workers: int = 8
    

@dataclass
class TrainingConfig:
    output_dir: str = "/mnt/training-data/checkpoints"
    train_epochs: int = 3
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    run_name: Optional[str] = None
