import os
import copy
import random
import logging
import warnings
from typing import Optional, Dict, Union, List, Any, Sequence
from dataclasses import dataclass, field

import torch
import datasets
import transformers
from transformers import TrainingArguments, HfArgumentParser, TrainingArguments
    

# from transformers import , \
#     AutoModelForCausalLM, AutoTokenizer, AutoConfig, \
#     DataCollatorForLanguageModeling, DataCollatorWithPadding, \
#     Trainer, \
#     set_seed
from trl import SFTTrainer
from unsloth import FastLanguageModel

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")  # Ignore Transformers' caching warning
IGNORE_INDEX = -100


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(default=None)
    trust_remote_code: Optional[bool] = field(default=False)
    load_in_8bit: Optional[bool] = field(default=False)
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
                                        split="train",
                                        data_files=data_files, 
                                        num_proc=args.num_proc)


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
    bs = len(examples['input'])
    inputs = [item.strip() for item in examples['input']]
    outputs = [item.strip() for item in examples['output']]
    
    examples = [s + t for s, t in zip(inputs, outputs)]
    
    max_length = args.max_length - len(prefix_prompt_tokens) - len(postfix_prompt_tokens)
    model_inputs, tokenized_source = [tokenizer(strings, 
                                                truncation=True, 
                                                max_length=max_length,) 
                                                # padding=True)
                                                for strings in (examples, inputs)]

    model_inputs["labels"] = []
    for i in range(bs):
        model_inputs["input_ids"][i] = (prefix_prompt_tokens + 
                                        model_inputs["input_ids"][i] + 
                                        postfix_prompt_tokens)
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        
        label = copy.deepcopy(model_inputs["input_ids"][i])
        label[:len(tokenized_source["input_ids"][i])] = (IGNORE_INDEX,) * len(tokenized_source["input_ids"][i])
        
        model_inputs["labels"].append(label)
    
        assert len(model_inputs["input_ids"][i]) <= args.max_length

    return model_inputs

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # ============ Prepare experiment ========
    save_dir = os.path.join(training_args.output_dir, training_args.run_name)
    training_args.output_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        handlers=[
                            logging.FileHandler(os.path.join(save_dir, "exp.log")),
                            logging.StreamHandler()
                        ])
    
    # ============ Load dataset ==============
    dataset = load_data(data_args)
    
    tokenized_dataset = dataset.map(
        preprocess_fn, 
        batched=True,
        num_proc=data_args.num_proc,
        remove_columns=dataset.column_names,
        load_from_cache_file=True,
        desc="Running Encoding",
        fn_kwargs={
            "args": data_args,
            "tokenizer": tokenizer, 
        }
    )
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    train_dataset = tokenized_dataset["train"].shuffle(seed=42)
    # eval_dataset = tokenized_dataset["eval"].shuffle(seed=42)
    
    logger.info(f"Training dataset samples: {len(train_dataset)}")
    if training_args.local_rank == 0:
        logger.info(f"Training dataset samples: {len(train_dataset)}")
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: \n{train_dataset[index]['input_ids']}, {train_dataset[index]['labels']}.")
            logger.info(f"Sample {index} of the training set: \n{tokenizer.decode(list(train_dataset[index]['input_ids']))}.")


    # ============ Load model ============
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_args.model_name_or_path,
        max_seq_length = data_args.max_length,
        dtype = None, # None for auto detection
        # load_in_4bit = True, # Use 4bit quantization to reduce memory usage. Can be False
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    # Do model patching and add fast LoRA weights
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Dropout = 0 is currently optimized
        bias = "none",    # Bias = "none" is currently optimized
        use_gradient_checkpointing = True,
        random_state = 3407,
    )

    logger.info("Start training ...")
    trainer = SFTTrainer(
        model = model,
        args = training_args,
        train_dataset = dataset,
        data_collator=data_collator,
        # dataset_text_field = "text",
        max_seq_length = data_args.max_length,
    )
    trainer.train()
    trainer.save_state()


if __name__ == "__main__":
    main()