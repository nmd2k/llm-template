export HF_HOME=/datadrive05/dungnm31/.cache
export TRANSFORMERS_CACHE=/datadrive05/dungnm31/.cache

MODEL_PATH="/datadrive05/dungnm31/llm_template/checkpoints/phi2-lora/checkpoint-150"

python ./deploy/host_llm.py --model=$MODEL_PATH --lora