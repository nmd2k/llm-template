import os
import json
import time
import logging
import warnings

from utils.utils import EosListStoppingCriteria
warnings.filterwarnings("ignore")

import random
from tqdm import tqdm
from typing import Optional
from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, \
    HfArgumentParser, GenerationConfig, DataCollatorWithPadding

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ==================================================
#                   SETUP ARGUMENT
# ==================================================
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    prefix_prompt: Optional[str] = field(
        default="", metadata={"help": "prefix prompt"}
    )
    postfix_prompt: Optional[str] = field(
        default="", metadata={"help": "postfix prompt"}
    )
    def __post_init__(self):
        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name_or_path


@dataclass
class DataTrainingArguments:
    dataset_name_or_path: str = field(
        default=None, 
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    output_dir: str = field(
        metadata={"help": "The output directory where data will be saved."},
        default=None,
    )
    max_eval_samples: Optional[int] = field(
        default=None, metadata={"help": ("For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set.")},
    )
    max_length: Optional[int] = field(
        default=None, metadata={"help": ("max_input_length")},
    )
    num_proc: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    label_pad_token_id: Optional[int] = field(
        default=-100,
        metadata={"help": "Token id to ignore in calculating loss"},
    )
    ignore_input_token_label: bool = field(
        default=True, metadata={"help": "ignore loss for the input sequence"}
    )
    padding_side: Optional[str] = field(
        default="right",
        metadata={"help": "Whether to pad on the left or right side of the input."}
    )


@dataclass
class GeneratorArguments:
    batch_size: Optional[int] = field(default=32)
    temperature: Optional[float] = field(default=1.0)
    top_p: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=0)
    do_sample: bool = field(default=True)
    num_return_sequences: Optional[int] = field(
        default=1, metadata={"help": "Number of sequence to generate"})
    do_passk: bool = field(default=False)
    data_pass_k: Optional[str] = field(default="human_eval")
    num_beams: Optional[int] = field(default=1)
    early_stopping: bool = field(default=False)

# ==================================================
#               LOAD DATASET & PREPROCESS
# ==================================================

def load_data(dataset_name_or_path, cache_dir: str=None):
    """Load data from huggingface hub or load local json file"""
    try:
        dataset = load_dataset(dataset_name_or_path, cache_dir=cache_dir)
        return dataset["test"]
    except Exception:
        logger.info(
            "Failed to load dataset from huggingface hub. "
            f"Loading local json file {dataset_name_or_path}.")
        
        # TODO: change the name of test_set file (not "test.jsonl")
        dataset = load_dataset("json", data_files={"test": dataset_name_or_path})
        
        return dataset["test"]

def preprocess_function(tokenizer, model_args, data_args):
    def _preprocess_function(examples):
        new_inputs = []
        for idx in range(len(examples['code'])):
            input_str = (
                model_args.prefix_prompt + 
                f"\n### Code documentation:\n{examples['original_docstring'][idx]}" +
                f"\n### Code snippet:\n```{examples['code'][idx]}```" +
                model_args.postfix_prompt
            )
            new_inputs.append(input_str)
        # new_str = f"<s>[INST]{new_str}[/INST]</s>"  # mixtral template
            
        return tokenizer(new_inputs, 
                        truncation=True, 
                        max_length=data_args.max_length,
                        padding=False,
                        return_tensors=None)
    
    return _preprocess_function

# ==================================================
#                       MAIN
# ==================================================

if __name__ == "__main__":
    # Load settings
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, GeneratorArguments))
    model_args, data_args, gen_args = parser.parse_args_into_dataclasses()
    
    accelerator = Accelerator()
    if accelerator.is_main_process:
        logger.info("Num process: %d | %d",
                    int(accelerator.state.num_processes),
                    int(accelerator.num_processes))
        
    # Load model
    logger.info("Loading model ...")
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, 
                                                 cache_dir=model_args.cache_dir,
                                                 trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)
    
    special_tokens = {}
    if tokenizer.pad_token_id is None:
        logger.info(
            "Tokenizer does not has pad token. Set the pad_token to eos_token.")
        special_tokens['pad_token'] = tokenizer.eos_token
        # tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is None:
        logger.info(
            "Tokenizer does not has pad token. Set the pad_token to eos_token.")
        special_tokens['bos_token'] = "<s>"
        tokenizer.add_bos_token = True
    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
    
    tokenizer.padding_side = data_args.padding_side
    
    generator_config = GenerationConfig(**gen_args.__dict__,
                                        max_length   = data_args.max_length,
                                        pad_token_id = tokenizer.pad_token_id,
                                        bos_token_id = tokenizer.bos_token_id,
                                        eos_token_id = tokenizer.eos_token_id)
    
    dataset = load_data(data_args.dataset_name_or_path, model_args.cache_dir)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer,
                                            padding=True,
                                            max_length=data_args.max_length)
    preprocess_fn = preprocess_function(tokenizer, model_args, data_args)
    tokenized_dataset = dataset.map(preprocess_fn,
                                    batched=True,
                                    num_proc=data_args.num_proc,
                                    remove_columns=dataset.column_names,
                                    # load_from_cache_file=True,
                                    desc="Running tokenizer on dataset")

    ds_loader = DataLoader(tokenized_dataset,
                           collate_fn=data_collator,
                           batch_size=gen_args.batch_size,
                           pin_memory=True, shuffle= False)

    model, ds_loader = accelerator.prepare(model, ds_loader)
    os.makedirs(data_args.output_dir, exist_ok=True)
    
    logger.info("Demo input:\n" + tokenizer.decode(tokenized_dataset[0]['input_ids']))
    
    # ===========================================
    #               START GENERATING
    # ===========================================
    total = int(len(ds_loader.dataset) / gen_args.batch_size)
    
    start_time = time.time()
    logger.info("Generating ...")
    results = []
    for batch_id, batch in tqdm(enumerate(ds_loader), total=total):
        with torch.no_grad():
            outputs = accelerator.unwrap_model(model).generate(
                        input_ids=batch["input_ids"], 
                        generation_config=generator_config,
                        bos_token_id=tokenizer.bos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id= tokenizer.eos_token_id,
                        stopping_criteria=[
                            EosListStoppingCriteria([tokenizer.eos_token_id])
                        ])
        
        # generated_tasks = batch["ids"].repeat(gen_args.num_return_sequences)
        generated_tokens = accelerator.pad_across_processes(
            outputs, dim=1, pad_index=tokenizer.pad_token_id
        )
        
        writer = open(os.path.join(data_args.output_dir, f"{batch_id}.jsonl"), "w")
        inputs = tokenizer.batch_decode(batch["input_ids"], 
                                        clean_up_tokenization_spaces=True)
        results = tokenizer.batch_decode(generated_tokens, 
                               clean_up_tokenization_spaces=True)
        print(inputs[0])
        print("="*50)
        print(results[0])
        
        for idx in range(len(results)):
            writer.write(json.dumps({"input": inputs[idx], "output": results[idx]}) + "\n")
    
    logger.info("Completion time: %d min", (time.time() - start_time) // 60)
    
