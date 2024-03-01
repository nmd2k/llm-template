import os
import sys
import logging
import warnings
import random

import torch
import datasets
from torch.utils.data import DataLoader, random_split
from transformers import HfArgumentParser, TrainingArguments, \
    AutoModelForCausalLM, AutoTokenizer, AutoConfig, \
    DataCollatorForLanguageModeling, DataCollatorWithPadding, \
    Trainer, \
    set_seed
    
from utils.train_args import ModelArguments, DataArguments
from utils.data_preprocessor import preprocess_fn
from utils.data_collator import DataCollatorForSupervisedDataset
from utils.callbacks import WandbPredictionProgressCallback
from utils.utils import print_trainable_parameters

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")  # Ignore Transformers' caching warning
IGNORE_INDEX = -100


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
    
    return dataset


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # accelerator = Accelerator()
    
    # ============ Prepare experiment ========
    save_dir = os.path.join(training_args.output_dir, training_args.run_name)
    training_args.output_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        handlers=[
                            logging.FileHandler(os.path.join(save_dir, "exp.log")),
                            logging.StreamHandler()
                        ])
    
    # ============ Load & setup tokenizer ====
    # config = AutoConfig.from_pretrained(model_args.model_base)
    tokenizer = AutoTokenizer.from_pretrained(
                        model_args.model_name_or_path,
                        cache_dir=model_args.cache_dir,
                        padding_side="left",
                        model_max_length=data_args.max_length,
                        trust_remote_code=model_args.trust_remote_code,)
                        # add_eos_token=True,
                        # add_bos_token=True)
    
    logger.info(f"PAD Token: {tokenizer.pad_token}")
    logger.info(f"BOS Token: {tokenizer.bos_token}")
    logger.info(f"EOS Token: {tokenizer.eos_token}")
    
    if tokenizer.pad_token_id is None:
        logger.info("Tokenizer does not has pad token. Set the pad_token to eos_token.")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # ============ Load model ==============
    set_seed(42)  #  Set seed efore initializing model
    model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            load_in_8bit=model_args.load_in_8bit,
            # torch_dtype=model_args.dtype,
            cache_dir=model_args.cache_dir)
    
    # model = torch.compile(model)  # Pytorch >= 2.0
    
    if model_args.lora:
        from peft import LoraConfig, get_peft_model, \
            prepare_model_for_kbit_training

        lora_config = LoraConfig( # mixtral config
            r=8,
            lora_alpha=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ],
            bias="none",
            lora_dropout=0.05,  # Conventional
            task_type="CAUSAL_LM",
        )
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        print_trainable_parameters(model)
        
        
    if training_args.local_rank > 0: 
        torch.distributed.barrier()
        
    # ============ Load dataset ==============
    dataset = load_data(data_args)
    
    tokenized_dataset = dataset.map(
        preprocess_fn, 
        batched=True,
        num_proc=data_args.num_proc,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
        desc="Running Encoding",
        fn_kwargs={
            "args": data_args,
            "tokenizer": tokenizer, 
        }
    )
    if training_args.local_rank == 0:
        torch.distributed.barrier()
    
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer,)
    train_dataset, eval_dataset = tokenized_dataset.train_test_split(test_size=0.2).values()
    train_dataset = train_dataset.shuffle(seed=42)
    
    if training_args.local_rank == 0:
        logger.info(f"Training dataset samples: {len(train_dataset)}")
        logger.info(f"Evaluation dataset samples: {len(eval_dataset)}")
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: \n{train_dataset[index]['input_ids']}, {train_dataset[index]['labels']}.")
            logger.info(f"Sample {index} of the training set: \n{tokenizer.decode(list(train_dataset[index]['input_ids']))}.")

    # ============ Training ==============
    logger.info("Start training ...")
    model.is_parallelizable = True
    model.model_parallel = True
    
    trainer = Trainer(model=model, 
                      args=training_args,
                      tokenizer=tokenizer,
                      data_collator=data_collator,
                      train_dataset=train_dataset,
                      eval_dataset=eval_dataset,
                      compute_metrics=None)
    
    model.config.use_cache = False
    
    # Instantiate the WandbPredictionProgressCallback
    if training_args.report_to:
        progress_callback = WandbPredictionProgressCallback(
            trainer=trainer,
            tokenizer=tokenizer,
            val_dataset=eval_dataset,
            num_samples=100,
            freq=1,
        )

    # Add the callback to the trainer
    trainer.add_callback(progress_callback)

    trainer.train()
    trainer.save_state()
    # # ============ Save model state ==============
    # state_dict = trainer.model.state_dict()
    # if trainer.args.should_save:
    #     cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
    #     del state_dict
    #     trainer._save(training_args.output_dir, state_dict=cpu_state_dict)  # noqa


if __name__ == "__main__":
    main()
