import os
import sys
import logging
import warnings
import yaml

import torch
import datasets
import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    BitsAndBytesConfig,
    AutoModel,
    LlamaForCausalLM,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    LlamaTokenizer,
    CodeLlamaTokenizer,
    AutoTokenizer,
    AutoConfig,
    set_seed
)
from peft import LoraConfig, get_peft_model, \
    prepare_model_for_int8_training, PeftModel, \
    get_peft_model_state_dict

from src.utils.logger import create_logger
from src.utils.training_utils import train, \
    smart_tokenizer_and_embedding_resize, SavePeftModelCallback
from src.models.data_args import DataTrainingArguments
from src.models.model_args import ModelArguments
from src.models.gen_args import GeneratorArguments
from src.utils.utils import display_params


def load_model(args, config, is_train=True):
    def _issue_warnings_after_load(load_result):
        if len(load_result.missing_keys) != 0:
            if model._keys_to_ignore_on_save is not None and set(load_result.missing_keys) == set(
                model._keys_to_ignore_on_save
            ):
                model.tie_weights()
            else:
                print(f"There were missing keys in the checkpoint model loaded: {load_result.missing_keys}.")
        if len(load_result.unexpected_keys) != 0:
            print(
                f"There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}."
            )
    torch_dtype = (
        args.torch_dtype
        if args.torch_dtype in ["auto", None]
        else getattr(torch, args.torch_dtype)
    )
    if args.model_type == "enc_dec":
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path,
                                    trust_remote_code=True,
                                    config=config,
                                    resume_download=True,
                                    load_in_8bit=args.load_in_8bit, 
                                    low_cpu_mem_usage=args.low_cpu_mem_usage,
                                    torch_dtype=torch_dtype,
                                    cache_dir=args.cache_dir,
                                    device_map=args.device_map,)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                    trust_remote_code=True,
                                    config=config,
                                    resume_download=True,
                                    load_in_8bit=args.load_in_8bit, 
                                    low_cpu_mem_usage=args.low_cpu_mem_usage,
                                    torch_dtype=torch_dtype,
                                    cache_dir=args.cache_dir,
                                    device_map=args.device_map) #, force_download=True, resume_download=False)
    
    if args.lora:
        assert os.path.exists(args.lora), "Lora config is missing."
        with open(args.lora, "r") as lora_config_file:
            lora_config = yaml.safe_load(lora_config_file)
        
        lora_config = LoraConfig(
            r=lora_config["lora_r"],
            lora_alpha=lora_config["lora_alpha"],
            lora_dropout=lora_config["lora_dropout"],
            task_type=lora_config["task_type"],
            target_modules=lora_config["target_modules"],
            bias="none",
        )
        print(model)
            
        if is_train:
            model.train()
            if args.load_in_8bit:
                model = prepare_model_for_int8_training(model)
            model.enable_input_require_grads()
            model = get_peft_model(model, lora_config)
        
        else:

            if "adapter_model.bin" in os.listdir(args.model_name_or_path):
                model = PeftModel.from_pretrained(model, 
                                    args.model_name_or_path, 
                                    device_map=args.device_map,
                                    load_in_8bit=args.load_in_8bit, 
                                    low_cpu_mem_usage= args.low_cpu_mem_usage)
        
            elif "pytorch_model.bin" in os.listdir(args.model_name_or_path):
                if args.load_in_8bit:
                    model = prepare_model_for_int8_training(model)
                # Case when the checkpoint saved all the model but not only the adapter
                state_dict = torch.load(
                    os.path.join(args.model_name_or_path, "pytorch_model.bin"), 
                    map_location="cpu"
                )
                model = get_peft_model(model, lora_config)
                load_result = model.load_state_dict(state_dict, False)
                _issue_warnings_after_load(load_result)

    model.config.use_cache = False # silence the warnings. 
    # Please re-enable for inference!
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    
    return model


def main():
    torch.cuda.empty_cache()
    parser = HfArgumentParser((ModelArguments, 
                               DataTrainingArguments, 
                               GeneratorArguments,
                               TrainingArguments,))
    model_args, data_args, gen_args, training_args = parser.parse_args_into_dataclasses()
    
    # ============ Setup logger ==============
    create_logger()
    logger = logging.getLogger(__name__)
    warnings.filterwarnings("ignore")  # Ignore Transformers' caching warning
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    
    # ============ Load dataset ==============
    if data_args.dataset_name_or_path is not None:
        dataset = datasets.load_dataset(
            data_args.dataset_name_or_path,
            cache_dir=model_args.cache_dir, 
            num_proc=data_args.num_proc,
        )
    else:
        assert data_args.train_file is not None
        
        data_files = {'train': data_args.train_file}
        if data_args.validation_file:
            data_files['valid'] = data_args.validation_file
        
        dataset = datasets.load_dataset('json', 
                                        data_files=data_files, 
                                        num_proc=data_args.num_proc, 
                                        cache_dir=model_args.cache_dir)
    
    # ============ Load & setup tokenizer ==============
    config = AutoConfig.from_pretrained(model_args.model_base, 
                                        trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                            cache_dir=model_args.cache_dir,
                            padding_side=data_args.padding_side)
    
    # ============ Load model ==============
    set_seed(training_args.seed)  #  Set seed efore initializing model
    model = load_model(model_args, config)
    smart_tokenizer_and_embedding_resize(special_tokens=data_args.new_tokens, 
                                        tokenizer=tokenizer, 
                                        model=model)

    # path = "/datadrive05/dungnm31/Exp/phi15/checkpoint-3450/pytorch_model.bin"
    # model.load_state_dict(torch.load(path))
    model = model.cuda()
    # print("model", model.device)
    
    # Set use_cache to False to use gradient checkpointing
    if training_args.gradient_checkpointing:
        model.config.use_cache = False

    if training_args.deepspeed is None:
        display_params(model)
        
    if training_args.do_train:
        trainer_extra_kwargs = {}
        if model_args.lora:
            trainer_extra_kwargs["callbacks"] = [SavePeftModelCallback]
            # trainer.evaluate(tokenized_dataset["test"])
            old_state_dict = model.state_dict
            model.state_dict = (
                lambda self, *_, **__: peft.get_peft_model_state_dict(
                    self, old_state_dict()
                )
            ).__get__(model, type(model))

        train(model=model, 
              dataset=dataset, 
              tokenizer=tokenizer, 
              data_args=data_args, 
              training_args=training_args,
              **trainer_extra_kwargs)
        
    elif training_args.do_eval:
        generation(model=model, 
                   dataset=dataset, 
                   tokenizer=tokenizer, 
                   data_args=data_args,
                   model_args=model_args,
                   gen_args=gen_args,
                   output_dir=training_args.output_dir)
        
    else:
        raise NotImplemented("Inference function not implemented")


if __name__ == "__main__":
    main()