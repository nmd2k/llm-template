import os
import sys
import logging
import warnings
import random
import copy

from typing import Optional, Dict, Union, List, Any
from dataclasses import dataclass, field

import torch
import datasets
import transformers
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, TrainingArguments, \
    AutoModelForCausalLM, AutoTokenizer, AutoConfig, \
    DataCollatorForLanguageModeling, DataCollatorWithPadding, \
    Trainer, \
    set_seed

# from peft import LoraConfig, get_peft_model, \
#     prepare_model_for_int8_training, PeftModel, \
#     get_peft_model_state_dict

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")  # Ignore Transformers' caching warning
IGNORE_INDEX = -100

# ==================================================
#                   SETUP ARGUMENT
# ==================================================

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(default=None)
    trust_remote_code: Optional[bool] = field(default=False)
    # dtype: Optional[str] = field(default="bfloat16")

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
#                     LOAD MODEL
# ==================================================
def load_model(args):
    model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                trust_remote_code=args.trust_remote_code,
                # torch_dtype=args.dtype,
                cache_dir=args.cache_dir)

    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)
    
    return model

# ==================================================
#              LOAD & PROCESSING DATASET
# ==================================================
def load_data(args):
    try:
        dataset = datasets.load_dataset(
            args.dataset_name_or_path,
            num_proc=args.num_proc,
        )
    except Exception: #Not found dataset
        assert os.path.isfile(args.dataset_name_or_path), "Local path not found"
        data_files = {'train': args.dataset_name_or_path}
        dataset = datasets.load_dataset('json', 
                                        data_files=data_files, 
                                        num_proc=args.num_proc)
    
    return dataset

def preprocess_fn(
    examples, 
    args: DataArguments,
    tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    prefix_prompt_tokens , postfix_prompt_tokens = [], []
    if args.prefix_prompt is not None:
        prefix_prompt_tokens = tokenizer.encode(
            args.prefix_prompt, 
            add_special_tokens=False)
    if args.postfix_prompt is not None:
        postfix_prompt_tokens = tokenizer.encode(
            args.postfix_prompt, 
            add_special_tokens=False)
    
    # ============ Customize function ==============
    bs = len(examples['text'])
    inputs = [item.strip() for item in examples['text']]
    
    max_length = args.max_length - len(prefix_prompt_tokens) - len(postfix_prompt_tokens)
    model_inputs = tokenizer(inputs,
                            truncation=True, 
                            max_length=max_length,
                            padding="max_length")
                            # return_tensors='pt')
    
    for i in range(bs):
        model_inputs["input_ids"][i] = (prefix_prompt_tokens + 
                                        model_inputs["input_ids"][i] + 
                                        postfix_prompt_tokens)
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        assert len(model_inputs["input_ids"][i]) <= args.max_length
    
    model_inputs["labels"] = copy.deepcopy(model_inputs["input_ids"])
    
    return model_inputs


def main():
    torch.cuda.empty_cache()
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    accelerator = Accelerator()
    
    # ============ Prepare experiment ========
    save_dir = os.path.join(training_args.output_dir, training_args.run_name)
    training_args.output_dir = save_dir
    logging.basicConfig(level=logging.INFO,
                        handlers=[
                            logging.FileHandler(os.path.join(save_dir, "exp.log")),
                            logging.StreamHandler()
                        ])
    
    # ============ Load dataset ==============
    dataset = load_data(data_args)
    
    # ============ Load & setup tokenizer ====
    # config = AutoConfig.from_pretrained(model_args.model_base)
    tokenizer = AutoTokenizer.from_pretrained(
                        model_args.model_name_or_path,
                        cache_dir=model_args.cache_dir,
                        padding_side="right",
                        trust_remote_code=model_args.trust_remote_code)
    
    logger.info(f"PAD Token: {tokenizer.pad_token}")
    logger.info(f"BOS Token: {tokenizer.bos_token}")
    logger.info(f"EOS Token: {tokenizer.eos_token}")
    
    if tokenizer.pad_token_id is None:
        logger.info("Tokenizer does not has pad token. Set the pad_token to eos_token.")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # ============ Load model ==============
    set_seed(42)  #  Set seed efore initializing model
    model = load_model(model_args)
    
    tokenized_dataset = dataset.map(
        preprocess_fn, 
        batched=True,
        num_proc=data_args.num_proc,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=True,
        desc="Running Encoding",
        fn_kwargs={
            "args": data_args,
            "tokenizer": tokenizer, 
        }
    )
    
    train_dataset = tokenized_dataset["train"].shuffle(seed=42)
    # eval_dataset = tokenized_dataset["eval"].shuffle(seed=42)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    model, train_dataset = accelerator.prepare(model, train_dataset)
    
    logger.info(f"Training dataset samples: {len(train_dataset)}")
    if training_args.local_rank == 0:
        torch.distributed.barrier()
        logger.info(f"Training dataset samples: {len(train_dataset)}")
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: \n{train_dataset[index]['input_ids']}, {train_dataset[index]['labels']}.")
            logger.info(f"Sample {index} of the training set: \n{tokenizer.decode(list(train_dataset[index]['input_ids']))}.")
    
    trainer = Trainer(model=model, 
                      args=training_args,
                      tokenizer=tokenizer,
                      data_collator=data_collator,
                      train_dataset=train_dataset,
                      compute_metrics=None)
    # model.config.use_cache = False

    trainer.train()
    trainer.save_state()
    # ============ Save model state ==============
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(training_args.output_dir, state_dict=cpu_state_dict)  # noqa


if __name__ == "__main__":
    main()