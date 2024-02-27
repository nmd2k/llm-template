import copy
from typing import Dict, List, Any

import transformers
from transformers import Seq2SeqTrainingArguments
from transformers.tokenization_utils import PreTrainedTokenizer

IGNORE_INDEX = -100

INPUT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Input:
{input}
### Response:
"""

NON_INPUT_TEMPLATE = """
Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Response:
"""

# FINE_TUNING_PROMPT = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\r\n### Instruction: {}\r\n### Response:"
FINE_TUNING_PROMPT = "Instruct: {}\r\n Output:"

def preprocess_fn(
    examples, args,
    tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    prefix_prompt_tokens , postfix_prompt_tokens = [], []
    if args.prefix_prompt is not None:
        prefix_prompt_tokens = tokenizer.encode(
            tokenizer.bos_token + args.prefix_prompt, 
            add_special_tokens=False)
    if args.postfix_prompt is not None:
        postfix_prompt_tokens = tokenizer.encode(
            args.postfix_prompt + tokenizer.eos_token, 
            add_special_tokens=False)
    
    # ============ Customize function ==============
    bs = len(examples['input'])
    # insts = [item.strip() for item in examples['instruction']]
    insts = [item.strip() for item in examples['input']]
    outputs = [item.strip() for item in examples['generation']]
    
    inputs = []
    for item in insts:
        inputs.append(FINE_TUNING_PROMPT.format(item))
            
    examples = [s + t for s, t in zip(inputs, outputs)]
    
    max_length = args.max_length - len(prefix_prompt_tokens) - len(postfix_prompt_tokens) # add eos and bos manually
    model_inputs, tokenized_source = [tokenizer(strings, 
                                                truncation=True, 
                                                max_length=max_length)
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


"""Source:
https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llmtuner/data/preprocess.py"""
def preprocess_supervised_dataset(
    examples: Dict[str, List[Any]],
    tokenizer: "PreTrainedTokenizer",
    template: "Template",
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    # for multiturn examples, we only mask the prompt part in each prompt-response pair.
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) != 1:
            continue

        messages = examples["prompt"][i] + examples["response"][i]
        input_ids, labels = [], []
        for turn_idx, (source_ids, target_ids) in enumerate(
            template.encode_multiturn(
                tokenizer, messages, examples["system"][i], examples["tools"][i], data_args.cutoff_len
            )
        ):
            if data_args.train_on_prompt:
                source_mask = source_ids
            elif turn_idx != 0 and template.efficient_eos:
                source_mask = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(source_ids) - 1)
            else:
                source_mask = [IGNORE_INDEX] * len(source_ids)

            input_ids += source_ids + target_ids
            labels += source_mask + target_ids

        if template.efficient_eos:
            input_ids += [tokenizer.eos_token_id]
            labels += [tokenizer.eos_token_id]

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)

    return model_inputs


def preprocess_unsupervised_dataset(
    examples: Dict[str, List[Any]],
    tokenizer: "PreTrainedTokenizer",
    template: "Template",
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    # build inputs with format `<bos> X` and labels with format `Y <eos>`
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1:
            continue

        if len(examples["response"][i]) == 1:
            messages = examples["prompt"][i] + examples["response"][i]
        else:
            messages = examples["prompt"][i] + [{"role": Role.ASSISTANT, "content": ""}]

        input_ids, labels = template.encode_oneturn(
            tokenizer, messages, examples["system"][i], examples["tools"][i], data_args.cutoff_len
        )

        if template.efficient_eos:
            labels += [tokenizer.eos_token_id]

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)

    return model_inputs