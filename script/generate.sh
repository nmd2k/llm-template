export TRANSFORMERS_CACHE="/datadrive05/dungnm31/.cache"
export HF_DATASETS_CACHE="/datadrive05/dungnm31/.cache"
export HF_HOME="/datadrive05/dungnm31/.cache"
# export https_proxy=http://10.16.29.10:8080

# codellama/CodeLlama-34b-Instruct-hf
export CUDA_VISIBLE_DEVICES=0,1
accelerate launch \
    --config_file config/default_config.yaml \
    src/generate.py \
    --model_name_or_path microsoft/phi-2 \
    --dataset_name_or_path /datadrive05/dungnm31/llm_template/dummy_code_alpaca.json \
    --batch_size 5 \
    --num_proc 50 \
    --temperature 0.9 \
    --top_p 0.95 \
    --do_sample \
    --num_return_sequences 1 \
    --max_new_tokens 256 \
    --cache_dir /datadrive05/dungnm31/.cache \
    --output_dir ./output \
    # --precision fp16
    # --device_map auto \
    # --low_cpu_mem_usage \
    # --prefix_prompt="<comment>" \
    # --postfix_prompt="<code>" \
