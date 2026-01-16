#!/bin/bash

# Training script for AV 3D Grounding finetuning - Cosmos-Reason2-8B model

# Model configuration
llm="nvidia/Cosmos-Reason2-8B"

# Weights & Biases configuration
export WANDB_PROJECT="qwen3-vl_cosmos_reason2"
run_name="av_3d_grounding_sft_qwen3_vl_cr2_8b"

# Output directory
output_dir=./outputs/av_3d_grounding_sft_qwen3_vl_cr2_8b

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Training hyperparameters
lr=2e-7
batch_size=2
grad_accum_steps=8

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration - using Cosmos Reason 2 AV 3D Grounding dataset
datasets=av3dgrounding_train
eval_datasets=av3dgrounding_eval

# Cache directory
cache_dir=./cache

echo "============================================"
echo "Starting training for model: $llm"
echo "Training dataset: $datasets"
echo "Evaluation dataset: $eval_datasets"
echo "Output directory: $output_dir"
echo "============================================"

# Launch training
# Parameters using sft.sh suggested finetuning configuration
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} \
         --deepspeed ${deepspeed} \
         --model_name_or_path "${llm}" \
         --dataset_use ${datasets} \
         --eval_dataset_use ${eval_datasets} \
         --eval_on_start \
         --data_flatten True \
         --tune_mm_vision True \
         --tune_mm_mlp True \
         --tune_mm_llm True \
         --bf16 \
         --output_dir ${output_dir} \
         --cache_dir ${cache_dir} \
         --num_train_epochs 2 \
         --per_device_train_batch_size ${batch_size} \
         --per_device_eval_batch_size 1 \
         --gradient_accumulation_steps ${grad_accum_steps} \
         --save_strategy "steps" \
         --save_steps 20 \
         --save_total_limit 20 \
         --learning_rate ${lr} \
         --weight_decay 0.01 \
         --warmup_ratio 0.03 \
         --max_grad_norm 1 \
         --lr_scheduler_type "cosine" \
         --logging_steps 1 \
         --model_max_length 4096 \
         --gradient_checkpointing True \
         --dataloader_num_workers 0 \
         --seed 42 \
         --run_name ${run_name} \
         --report_to wandb

# Check if training was successful
if [ $? -ne 0 ]; then
    echo "Training failed for model: $llm"
    exit 1
fi

echo "============================================"
echo "Training completed successfully!"
echo "Model saved to: $output_dir"
echo "============================================"
