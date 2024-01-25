# Template for tranining
export HF_HOME="/datadrive05/.cache"
# export https_proxy=http://10.16.29.10:8080

export CUDA_VISIBLE_DEVICES=0
export EXPERIMENT_NAME="test-2123"

accelerate launch \
    --config_file config/default_config.yaml src/train_wizardcoder.py \
    --model_name_or_path codellama/CodeLlama-7b-hf \
    --data_path /datadrive05/dungnm31/instruct-data/data/multi_inst/ver1/dataset.jsonl \
    --cache_dir /datadrive05/dungnm31/.cache \
    --model_max_length 512 \
    --output_dir ./checkpoints/$EXPERIMENT_NAME \
