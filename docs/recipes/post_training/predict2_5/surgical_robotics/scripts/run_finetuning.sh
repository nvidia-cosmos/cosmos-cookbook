#!/bin/bash
# -- SLURM array job for continued training --
#SBATCH --job-name=def-cb-cp2.5
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --partition=batch_block1,batch_block3,batch_block4
#SBATCH --account=healthcareeng_holoscan
#SBATCH --gres=gpu:8
#SBATCH --time=4:00:00
#SBATCH --output=logs/def-cb-cp2.5-720x960_ac_%A_%a.out
#SBATCH --array=0-9%1  # Run 10 jobs (0-9), only 1 at a time
#SBATCH --dependency=singleton

# Set environment variables
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=${master_addr:-"127.0.0.1"}
export worker_list=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')
echo "MASTER_ADDR="$MASTER_ADDR
echo "JobID: $SLURM_JOB_ID | Array Task ID: $SLURM_ARRAY_TASK_ID | Full list: $worker_list"

# Prepare multi-node environment variables
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
nodes_array=("${nodes[@]}")
head_node="${nodes_array[0]}"
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Environment settings
export LOGLEVEL=INFO
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=WARN
export PYTHONFAULTHANDLER=1

echo "Head node:       $head_node"
echo "Head node IP:    $head_node_ip"
echo "All nodes:       ${nodes_array[@]}"
echo "SLURM Job ID:    $SLURM_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"

# Set the necessary variables
CODE_PATH="/lustre/fsw/portfolios/healthcareeng/users/lzbinden/git/def-cookbook-cosmos-predict2.5"
DATASET_PATH="/lustre/fsw/portfolios/healthcareeng/users/nigeln/cache/huggingface/lerobot/jhu_lerobot/suturebot_lerobot"

# Run the training script inside the container

srun --export=ALL --container-image="/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/lzbinden/images/cosmos-predict-2.5.sqsh" \
     --container-mounts="$CODE_PATH:/workspace,$DATASET_PATH:/SutureBot,/lustre/fsw/portfolios/healthcareeng/users/lzbinden:/lustre/fsw/portfolios/healthcareeng/users/lzbinden,/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan:/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan" \
     --container-workdir=/workspace \
     bash -c '
        # Set up environment variables
        echo "MASTER_ADDR="$MASTER_ADDR
        export NCCL_DEBUG=INFO
        CURRENT_RANK=${SLURM_NODEID:-"0"}
        n_node=${SLURM_JOB_NUM_NODES:-1}
        echo "JobID: $SLURM_JOB_ID | Array Task ID: $SLURM_ARRAY_TASK_ID | Full list: $worker_list | Node rank: $CURRENT_RANK of $n_node"

        cd /workspace
        
        source .venv/bin/activate

        seed=$((1234 + $SLURM_ARRAY_TASK_ID * $n_node * 8))

        torchrun --nnodes=$n_node --nproc_per_node=8 --master_port=25001 --master_addr $MASTER_ADDR --node_rank=$CURRENT_RANK -m \
            scripts.train \
              --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
              -- \
              experiment="cosmos_predict2p5_2B_action_conditioned_suturebot_13frame_4nodes_release_oss" \
              checkpoint.save_iter=200 \
              ~dataloader_train.dataloaders
     '
