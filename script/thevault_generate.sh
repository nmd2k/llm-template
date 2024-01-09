# =========
# Script for generating the vault code generation
# =========
export TRANSFORMERS_CACHE="/app/.cache"
export HF_DATASETS_CACHE="/app/.cache"
# export https_proxy=http://10.16.29.10:8080

# mistralai/Mixtral-8x7B-Instruct-v0.1
# codellama/CodeLlama-34b-Instruct-hf
export CUDA_VISIBLE_DEVICES=0
accelerate launch \
    --config_file config/standard.yaml \
    src/generate.py \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --dataset_name_or_path /app/data/instruct-data/process_data/code_generation/small_thevault.jsonl \
    --batch_size 16 \
    --temperature 0.9 \
    --top_p 0.95 \
    --top_k 40 \
    --num_return_sequences 1 \
    --max_length 512 \
    --num_proc 50 \
    --cache_dir /app/.cache \
    --output_dir ./output \
    --padding_side="left" \
    --prefix_prompt="[INST]You are an expert at offering high-quality instruction to help developers with their tasks.\nPlease gain insight from the following code snippet to provide a dense explanation of 3 to 5 sentences for better understanding.\n" \
    --postfix_prompt="\n### Explanation:[/INST]"
    # --do_sample
    # --precision fp16
    # --device_map auto \
    # --low_cpu_mem_usage \
