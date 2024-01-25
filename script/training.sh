# Template for tranining
export HF_HOME="/datadrive05/.cache"
# export https_proxy=http://10.16.29.10:8080

export CUDA_VISIBLE_DEVICES=0
export EXPERIMENT_NAME="test-2123"

accelerate launch \
    --config_file config/default_config.yaml src/training.py \
    --model_name_or_path codellama/CodeLlama-7b-hf \
    --dataset_name_or_path /datadrive05/dungnm31/instruct-data/data/multi_inst/ver1/dataset.jsonl \
    --run_name $EXPERIMENT_NAME \
    --num_proc 50 \
    --per_device_train_batch_size 8 \
    --auto_find_batch_size \
    --logging_strategy steps \
    --evaluation_strategy no \
    --eval_steps 1000 \
    --save_steps 1000 \
    --logging_steps 100 \
    --metric_for_best_model loss \
    --num_train_epochs 1 \
    --learning_rate 3e-4 \
    --weight_decay 3e-5 \
    --warmup_steps 1000 \
    --gradient_accumulation_steps 4 \
    --save_total_limit 5 \
    --max_length 1024 \
    --output_dir /datadrive05/dungnm31/llm_template/checkpoints \
    --cache_dir /datadrive05/.cache \
    # --remove_unused_columns False
    # --gradient_checkpointing \
    # --load_best_model_at_end \