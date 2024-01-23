# =========
# Script for generating the vault code generation
# =========
export HF_HOME="/datadrive05/.cache"
# export https_proxy=http://10.16.29.10:8080

# mistralai/Mixtral-8x7B-Instruct-v0.1
# codellama/CodeLlama-34b-Instruct-hf
export CUDA_VISIBLE_DEVICES=1
accelerate launch \
    --config_file config/standard.yaml \
    src/generate.py \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --dataset_name_or_path /datadrive05/dungnm31/instruct-data/data/processed/defect-detection/small_raw.jsonl \
    --batch_size 4 \
    --temperature 0.9 \
    --top_p 0.95 \
    --top_k 40 \
    --num_return_sequences 1 \
    --max_length 512 \
    --num_proc 50 \
    --cache_dir /datadrive05/.cache \
    --output_dir ./output \
    --padding_side="left" \
    --prefix_prompt="[INST]Your task is to determine whether the provided code contains any security vulnerabilities that could potentially be exploited by attackers." \
    --postfix_prompt="[/INST]"
    # --do_sample
    # --precision fp16
    # --device_map auto \
    # --low_cpu_mem_usage \
