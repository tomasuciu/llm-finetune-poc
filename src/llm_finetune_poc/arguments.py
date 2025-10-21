from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Optional,
)
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="mistralai/Mistral-7B-v0.1",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_flash_attention_2: bool = field(
        default=True,
        metadata={"help": "Whether to use Flash Attention 2"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length for input text"}
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: str = field(
        default="glaiveai/glaive-function-calling-v2",
        metadata={"help": "The name of the dataset to use (via the datasets library)"}
    )
    dataset_split: str = field(
        default="train",
        metadata={"help": "Dataset split to use for training"}
    )
    validation_split_percentage: int = field(
        default=5,
        metadata={"help": "Percentage of training data to use for validation"}
    )
    preprocessing_num_workers: int = field(
        default=8,
        metadata={"help": "Number of processes to use for data preprocessing"}
    )

@dataclass
class CustomTrainingArguments(TrainingArguments):
    """
    Extended TrainingArguments with custom options.
    Inherits all standard HuggingFace TrainingArguments!
    """
    output_dir: str = field(
        default="/mnt/training-data/checkpoints",
        metadata={"help": "The output directory where checkpoints will be written"}
    )
    num_train_epochs: float = field(
        default=3.0,
        metadata={"help": "Total number of training epochs to perform"}
    )
    per_device_train_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training"}
    )
    per_device_eval_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation"}
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass"}
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "The initial learning rate for AdamW optimizer"}
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay to apply (if not zero) to all layers except bias/LayerNorm"}
    )
    warmup_steps: int = field(
        default=100,
        metadata={"help": "Number of steps used for a linear warmup from 0 to learning_rate"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "Save checkpoint every X updates steps"}
    )
    eval_steps: int = field(
        default=500,
        metadata={"help": "Run an evaluation every X steps"}
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Whether to use bf16 16-bit (mixed) precision training instead of 32-bit"}
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass"}
    )
    dataloader_num_workers: int = field(
        default=4,
        metadata={"help": "Number of subprocesses to use for data loading"}
    )
    run_name: Optional[str] = field(
        default=None,
        metadata={"help": "An optional descriptor for the run. Notably used for wandb logging"}
    )
    
    ddp_backend: str = field(
        default="nccl",
        metadata={"help": "The backend to use for distributed training"}
    )
    ddp_find_unused_parameters: bool = field(
        default=False,
        metadata={"help": "When using distributed training, whether to find unused parameters"}
    )
    dataloader_pin_memory: bool = field(
        default=True,
        metadata={"help": "Whether to pin memory in data loaders"}
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Remove columns not required by the model when using an nlp.Dataset"}
    )
