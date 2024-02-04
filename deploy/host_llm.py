import gradio as gr
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from argparse import ArgumentParser

global tokenizer
global model


def opt():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='microsoft/phi-1')
    parser.add_argument('--cache_dir', type=str, default='/datadrive05/dungnm31/.cache')
    parser.add_argument('--lora', action='store_true')
    return parser.parse_args()

def inference(text):
    template = "{instruct}\n"
    
    input_tokens = tokenizer(template.format(instruct=text), return_tensors="pt").to("cuda")
    output_tokens = model.generate(**input_tokens,
                                   max_length=256, 
                                   do_sample=True, 
                                   temperature=0.95, 
                                   pad_token_id=tokenizer.eos_token_id)
    output = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    output = output.replace(template.format(instruct=text), "")
    return output


if __name__ == "__main__":
    args = opt()
    if args.lora:
        from peft import AutoPeftModelForCausalLM, PeftModel, PeftConfig

        config = PeftConfig.from_pretrained(args.model)
        base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map="auto")
        model = PeftModel.from_pretrained(base_model, args.model)
        # lora_model = AutoPeftModelForCausalLM.from_pretrained(args.model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,    
            device_map="auto",
            cache_dir=args.cache_dir
        )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model) 

    interface = gr.Interface(inference, "text", "text")
    interface.launch()
