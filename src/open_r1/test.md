CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen2.5-1.5B-Instruct

CUDA_VISIBLE_devices=1,2,3,4,5 \
ACCELERATE_LOG_LEVEL=info \
accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes 5 \
src/open_r1/grpo.py --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo_code.yaml


nohup CUDA_VISIBLE_devices=1,2,3,4,5 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes 5 src/open_r1/grpo.py --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo_code.yaml > training.log 2>&1 &