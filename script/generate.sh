export TRANSFORMERS_CACHE="/cm/archive/namlh35/.cache"
export HF_DATASETS_CACHE="/cm/archive/namlh35/.cache"
# export https_proxy=http://10.16.29.10:8080

# codellama/CodeLlama-34b-Instruct-hf
export CUDA_VISIBLE_DEVICES=1
accelerate launch \
    --config_file config/standard.yaml \
    src/generate.py \
    --model_name_or_path codellama/CodeLlama-7b-Instruct-hf \
    --dataset_name_or_path ./data/multi_task_test.jsonl \
    --batch_size 4 \
    --temperature 0.9 \
    --top_p 0.95 \
    --top_k 40 \
    --do_sample \
    --num_return_sequences 2 \
    --max_length 4096 \
    --num_proc 50 \
    --cache_dir /app/.cache \
    --output_dir /app/instruct-data/data/multi_inst \
    --padding_side="left" \
    # --precision fp16
    # --device_map auto \
    # --low_cpu_mem_usage \
    # --prefix_prompt="<comment>" \
    # --postfix_prompt="<code>" \
