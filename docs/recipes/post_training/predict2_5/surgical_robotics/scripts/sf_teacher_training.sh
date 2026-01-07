#!/bin/bash
# -- SLURM array job for continued training --
#SBATCH --job-name=cb-cp2.5
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --partition=batch_block1,batch_block3,batch_block4
#SBATCH --account=healthcareeng_holoscan
#SBATCH --gres=gpu:8
#SBATCH --time=4:00:00
#SBATCH --output=logs/cb-cp2.5-teacher_720x960_ac_%A_%a.out
#SBATCH --array=0-9%1  # Run 10 jobs (0-9), only 1 at a time
#SBATCH --dependency=singleton

TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
# Archive repository (excluding binaries/logs) - SLURM-safe, uses pyproject.toml as repo root marker
_d="${SLURM_SUBMIT_DIR:-$PWD}"; while [[ "$_d" != "/" && ! -f "$_d/pyproject.toml" ]]; do _d="$(dirname "$_d")"; done
REPO_ROOT="$_d"; [[ -f "$REPO_ROOT/pyproject.toml" ]] || { echo "ERROR: Cannot find repo root"; exit 1; }
REPO_DEST="/lustre/fsw/portfolios/healthcareeng/users/lzbinden/imaginaire/output/repo/run_${TIMESTAMP}_${SLURM_JOB_NAME:-unknown}"
mkdir -p "$REPO_DEST" && tar -czf "$REPO_DEST/repo.tar.gz" -C "$REPO_ROOT" \
  --exclude='*.log' --exclude='*.out' --exclude='*.pt' --exclude='*.pth' --exclude='*.bin' --exclude='*.onnx' \
  --exclude='*.npy' --exclude='*.npz' --exclude='*.parquet' --exclude='*.h5' --exclude='*.safetensors' \
  --exclude='*.mp4' --exclude='*.jpg' --exclude='*.jpeg' --exclude='*.png' --exclude='*.gif' --exclude='*.sqsh' \
  --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='logs' --exclude='uv.lock' \
  --exclude='.venv' --exclude='venv' --exclude='.env' --exclude='*.egg-info' --exclude='.eggs' \
  --exclude='.pytest_cache' --exclude='.mypy_cache' --exclude='.ruff_cache' --exclude='dist' --exclude='build' --exclude='datasets' . \
  && cat > "$REPO_DEST/README.txt" <<EOF
SLURM_JOB_ID..: ${SLURM_JOB_ID}
SLURM_JOB_NAME: ${SLURM_JOB_NAME}
TIMESTAMP.....: ${TIMESTAMP}
REPO_ROOT.....: ${REPO_ROOT}
LOG_FILE......: ${REPO_ROOT}/logs/cb-cp2.5-teacher_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out
EOF
  echo "[jobId=${SLURM_JOB_ID}] Repo archived to: $REPO_DEST/repo.tar.gz"

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
CODE_PATH="/lustre/fsw/portfolios/healthcareeng/users/lzbinden/git/cookbook-cosmos-predict2.5"

# Run the training script inside the container

srun --export=ALL --container-image="/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/lzbinden/images/cosmos-predict-2.5.sqsh" \
     --container-mounts="$CODE_PATH:/workspace,/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/datasets/SutureBot_v2/LeRobot:/SutureBot,/lustre/fsw/portfolios/healthcareeng/users/lzbinden:/lustre/fsw/portfolios/healthcareeng/users/lzbinden,/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan:/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan" \
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
              experiment="ac_predict2p5_video2world_2b_suturebot_training" \
              checkpoint.save_iter=200 \
              ~dataloader_train.dataloaders
     '
