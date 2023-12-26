export TRANSFORMERS_CACHE="/datadrive05/dungnm31/.cache"
export HF_DATASETS_CACHE="/datadrive05/dungnm31/.cache"
# export https_proxy=http://10.16.29.10:8080

# codellama/CodeLlama-34b-Instruct-hf
export CUDA_VISIBLE_DEVICES=0,1
accelerate launch \
    --config_file config/standard.yaml \
    src/generate.py \
    --model_name_or_path /datadrive05/dungnm31/models/multitask/codellama7b_exp1/checkpoint-450/ \
    --dataset_name_or_path code_x_glue_cc_code_to_code_trans \
    --batch_size 2 \
    --temperature 0.9 \
    --top_p 0.95 \
    --top_k 40 \
    --do_sample \
    --num_return_sequences 2 \
    --max_length 512 \
    --num_proc 50 \
    --cache_dir /datadrive05/dungnm31/.cache \
    --output_dir ./output \
    --padding_side="left" \
    --prefix_prompt="Your task is to translate the provided C# code into its Java equivalent. C# code: " \
    --postfix_prompt="Java code: "
    # --precision fp16
    # --device_map auto \
    # --low_cpu_mem_usage \
    # --prefix_prompt="<comment>" \
    # --postfix_prompt="<code>" \
