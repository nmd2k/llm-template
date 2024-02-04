from typing import Optional, List
from dataclasses import dataclass, field

# ==================================================
#                   DATA ARGUMENTS
# ==================================================

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(default=None)
    trust_remote_code: Optional[bool] = field(default=False)
    load_in_8bit: Optional[bool] = field(default=False)
    # lora config
    lora: Optional[bool] = field(default=False)
    
    # dtype: Optional[str] = field(default="bfloat16")
    

# ==================================================
#               LOAD DATASET & PREPROCESS
# ==================================================

@dataclass
class DataArguments:
    dataset_name_or_path: str = field(default=None, metadata={"help": "Path to the training data."})
    num_proc: Optional[int] = field(default=None, metadata={"help": "Number of processes."})
    label_pad_token_id: Optional[int] = field(default=-100)
    prefix_prompt: Optional[str] = field(default="")
    postfix_prompt: Optional[str] = field(default="")
    max_length: Optional[int] = field(
        default=512,
        metadata={"help": "Maximum sequence length."}
    )

# ==================================================
#               LOAD DATASET & PREPROCESS
# ==================================================
@dataclass
class GeneratorArguments:
    batch_size: Optional[int] = field(
        default=32,
        metadata={"help": "Batch size."},
    )
    temperature: Optional[float] = field(default=1.0)
    top_p: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=0)
    do_sample: bool = field(default=True)
    num_return_sequences: Optional[int] = field(default=32) 
    num_gen_iterations: Optional[int] = field(default=1) 
