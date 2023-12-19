from typing import Optional, List
from dataclasses import dataclass, field

# ==================================================
#                   DATA ARGUMENTS
# ==================================================

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, 
        metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the perplexity on (a text file)."},
    )
    train_split_set: Optional[str] = field(default='train')
    valid_split_set: Optional[str] = field(default='valid')
    test_split_set: Optional[str] = field(default='test')
    instruction_column: Optional[str] = field(default=None, metadata={"help": "The instruction column training data file (a text file)."})
    input_column: Optional[str] = field(default=None, metadata={"help": "The input column training data file (a text file)."})
    output_column: Optional[str] = field(default=None, metadata={"help": "The output column training data file (a text file)."})
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    max_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, "
                "truncate the number of training examples to this value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, "
                "truncate the number of evaluation examples to this value if set."
            )
        },
    )
    padding_side: Optional[str] = field(default='right')
    new_tokens: Optional[str] = field(default=None)
    num_proc: Optional[int] = field(
        default=8,
    )
    label_pad_token_id: Optional[int] = field(
        default=-100,
        metadata={"help": "Token id to ignore in calculating loss"},
    )
    prefix_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "Prefix prompt for input"},
    )
    postfix_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "Postfix prompt for input"},
    )

    def __post_init__(self):
        pass
    

# ==================================================
#               LOAD DATASET & PREPROCESS
# ==================================================
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, 
    or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_base: Optional[str] = field(default=None, metadata={"help": "model base"})
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, "
                "then only materialize its parameters when the pretrained "
                "weights are loaded. set True will benefit LLM loading time "
                "and RAM consumption."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={
            "help": (
                "activate 8 bit training"
            )
        },
    )
    device_map: Optional[str] = field(
        default=None,
        metadata={"help": "Specify the device do you want to load the pretrained models"},
    )
    # Lora config
    lora: Optional[str] = field(default="")
    model_type: Optional[str] = field(default="casual")
    
    def __post_init__(self):
        if self.model_base is None:
            self.model_base = self.model_name_or_path
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
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
    temperature: Optional[float] = field(
        default=1.0,
    )
    top_p: Optional[float] = field(
        default=1.0,
    )
    top_k: Optional[int] = field(
        default=0,
    )
    do_sample: bool = field(
        default=True,
    )
    num_return_sequences: Optional[int] = field(
        default=32,
        metadata={"help": "Number of sequence to generate"},
    ) 
    num_gen_iterations: Optional[int] = field(
        default=1,
    ) 
