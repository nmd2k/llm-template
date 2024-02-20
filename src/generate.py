import os
import json
import time
import logging
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
from tqdm import tqdm
from typing import Optional
from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from accelerate import Accelerator, PartialState
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, \
    HfArgumentParser, GenerationConfig, DataCollatorWithPadding

from prompt_utils.prepare_prompt import createBreadthPrompt, createDepthPrompt
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(default=None)
    trust_remote_code: Optional[bool] = field(default=False)
    load_in_8bit: Optional[bool] = field(default=False)

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
    batch_size: Optional[int] = field(default=16)
    num_proc: Optional[int] = field(default=1)
    
@dataclass
class GenerationArguments:
    max_length: Optional[int] = field(default=20)
    max_new_tokens: Optional[int] = field(default=50)
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    temperature: Optional[float] = field(default=1.0)
    top_p: Optional[float] = field(default=0.9)
    repetition_penalty: Optional[float] = field(default=1.2)
    num_return_sequences: Optional[int] = field(default=1)


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

def preprocess_function(examples, tokenizer, data_args):
    bs = len(examples['instruction'])
    new_inputs = []
    for idx in range(bs):
        new_inputs.append(createBreadthPrompt(examples['instruction'][idx]))
    
    model_inputs = tokenizer(new_inputs, padding=True, return_tensors='pt')        
    return model_inputs

# ==================================================
#                       MAIN
# ==================================================

if __name__ == "__main__":
    accelerator = Accelerator()
    # Load settings
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, GenerationArguments))
    model_args, data_args, gen_args = parser.parse_args_into_dataclasses()
    generation_config = GenerationConfig(**gen_args.__dict__)
    os.makedirs(data_args.output_dir, exist_ok=True)
    
    # Load model
    logger.info("Loading model ...")
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, 
                                                 cache_dir=model_args.cache_dir,
                                                 load_in_8bit=model_args.load_in_8bit,
                                                 trust_remote_code=model_args.trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                              trust_remote_code=model_args.trust_remote_code)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    dataset = load_data(data_args.dataset_name_or_path, model_args.cache_dir)
    tokenized_dataset = dataset.map(preprocess_function,
                                    batched=True,
                                    num_proc=data_args.num_proc,
                                    remove_columns=dataset.column_names,
                                    load_from_cache_file=True,
                                    desc="Running tokenizer on dataset",
                                    fn_kwargs={
                                        "data_args": data_args,
                                        "tokenizer": tokenizer, 
                                    })

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    ds_loader = DataLoader(tokenized_dataset.with_format("torch"),
                           collate_fn=data_collator,
                           batch_size=data_args.batch_size, 
                           shuffle=False)
    
    model, ds_loader = accelerator.prepare(model, ds_loader)
    logger.info("Demo input:\n" + tokenizer.decode(tokenized_dataset[0]['input_ids']))
    
    # ===========================================
    #               START GENERATING
    # ===========================================
    start_time = time.time()
    logger.info("Generating ...")
    save_path = os.path.join(data_args.output_dir, 
                            f"batch_{accelerator.process_index}.jsonl")
    for batch_id, batch in tqdm(enumerate(ds_loader), total=len(ds_loader)):
        with torch.no_grad():
            outputs = accelerator.unwrap_model(model).generate(**batch, 
                                    generation_config=generation_config,
                                    pad_token_id=tokenizer.eos_token_id)
        
        batch_results = tokenizer.batch_decode(outputs, 
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=True)
        
        if batch_id == 0:
            logger.info("======== Demo output:\n" + batch_results[0])
            logger.info("======== ")
        
        with open(save_path, "a") as writer:
            for item in batch_results:
                json.dump(dict(generation=item), writer)
                writer.write("\n")
    
    logger.info("Completion time: %d min", (time.time() - start_time) // 60)
    
