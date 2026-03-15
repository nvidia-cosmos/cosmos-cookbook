#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Run finetuning on a single machine.
# Set env vars (see below), then: ./scripts/run_finetuning_standalone.sh
#
# You can run either inside a Docker container (recommended) or on the host.
# For Docker: build the image from the Cosmos-H-Surgical-Simulator repo first (see steps 1.1–1.3),
# then set COSMOS_CONTAINER_IMAGE to that image tag.

set -euo pipefail

# --- Config (override with env vars) ---
# Path to the Cosmos-H-Surgical-Simulator repo (training code).
CODE_PATH="${COSMOS_CODE_PATH:-/ephemeral/Cosmos-H-Surgical-Simulator}"
# Path to the SutureBot dataset in LeRobot format (output of convert_suturebot_to_lerobot_v3.py or create_mini_suturebot.py).
SUTUREBOT_LEROBOT_PATH="${SUTUREBOT_LEROBOT_PATH:?Set SUTUREBOT_LEROBOT_PATH to the LeRobot dataset path (e.g. \$HF_LEROBOT_HOME/suturebot_lerobot or .../suturebot_lerobot_mini)}"
# Path to the Cosmos-H-Surgical-Simulator checkpoint iter directory (e.g. iter_000023000).
# The checkpointer appends /model internally; do NOT include the model subdir in this path.
# Download from HuggingFace: hf download nvidia/Cosmos-H-Surgical-Simulator --include "checkpoints/iter_000023000/*"
COSMOS_H_CKPT_PATH="${COSMOS_H_CKPT_PATH:-}"
# Docker image built from Cosmos-H-Surgical-Simulator (see step 1.3). If unset, run on host using CODE_PATH.
COSMOS_CONTAINER_IMAGE="${COSMOS_CONTAINER_IMAGE:-}"
# Number of GPUs to use (default: all visible).
NGPUS="${NGPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
if [ "${NGPUS:-0}" -lt 1 ] 2>/dev/null; then
  echo "ERROR: No GPUs detected (NGPUS=${NGPUS}). Ensure nvidia-smi works or set NGPUS manually." >&2
  exit 1
fi
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-25001}"

export LOGLEVEL="${LOGLEVEL:-INFO}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export PYTHONFAULTHANDLER=1

echo "CODE_PATH=$CODE_PATH"
echo "SUTUREBOT_LEROBOT_PATH=$SUTUREBOT_LEROBOT_PATH"
echo "COSMOS_H_CKPT_PATH=${COSMOS_H_CKPT_PATH:-<not set — will warm-start from base Cosmos 2B>}"
echo "NGPUS=$NGPUS"
echo "MASTER_ADDR=$MASTER_ADDR"
if [[ -n "$COSMOS_CONTAINER_IMAGE" ]]; then
  echo "Running inside container: $COSMOS_CONTAINER_IMAGE"
  echo "On failure, traceback may be in \${IMAGINAIRE_OUTPUT_ROOT:-/tmp/imaginaire4-output}/torchelastic_error.json or run with NGPUS=1 to see it in the log."
fi

# Build the checkpoint override arg (empty string = use config default, i.e. base Cosmos 2B from HF).
# When COSMOS_H_CKPT_PATH points to a DCP directory (iter_XXXXXX/model), the trainer
# resumes from that distributed checkpoint; when it ends in .pt it loads a consolidated file.
_ckpt_override() {
  if [[ -n "${COSMOS_H_CKPT_PATH:-}" ]]; then
    echo "checkpoint.load_path=$COSMOS_H_CKPT_PATH"
  fi
}

_run_train() {
  # SUTUREBOT_DATASET_PATH is read by data.py to set SUTUREBOT_DATASET_SPECS[0]["path"].
  export SUTUREBOT_DATASET_PATH="$SUTUREBOT_LEROBOT_PATH"
  torchrun \
    --nnodes=1 \
    --nproc_per_node="$NGPUS" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    -m scripts.train \
    --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
    -- \
    experiment="cosmos_predict2p5_2B_action_conditioned_suturebot_13frame_4nodes_release_oss" \
    checkpoint.save_iter=200 \
    '~dataloader_train.dataloaders' \
    $(_ckpt_override)
}

if [[ -n "$COSMOS_CONTAINER_IMAGE" ]]; then
  # Run training inside Docker.
  # SUTUREBOT_DATASET_PATH is passed as an env var so data.py resolves the dataset path at import time.
  # The dataset is mounted read-only; the container writes checkpoints to OUT_ROOT on the host.
  OUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-/tmp/imaginaire4-output}"
  DOCKER_GPU_FLAG="${DOCKER_GPU_FLAG:---gpus all}"
  HF_HOME_HOST="${HF_HOME:-$HOME/.cache/huggingface}"
  ELASTIC_ERROR_FILE="$OUT_ROOT/torchelastic_error.json"
  DOCKER_ARGS=(
    $DOCKER_GPU_FLAG
    --ipc=host
    -v "$CODE_PATH:/workspace"
    -v "$SUTUREBOT_LEROBOT_PATH:$SUTUREBOT_LEROBOT_PATH:ro"
    -v "$OUT_ROOT:$OUT_ROOT"
    -e SUTUREBOT_DATASET_PATH="$SUTUREBOT_LEROBOT_PATH"
    -e NGPUS="$NGPUS"
    -e MASTER_ADDR="$MASTER_ADDR"
    -e MASTER_PORT="$MASTER_PORT"
    -e LOGLEVEL="$LOGLEVEL"
    -e NCCL_DEBUG="$NCCL_DEBUG"
    -e PYTHONFAULTHANDLER="$PYTHONFAULTHANDLER"
    -e HF_TOKEN="${HF_TOKEN:-}"
    -e WANDB_API_KEY="${WANDB_API_KEY:-}"
    -e IMAGINAIRE_OUTPUT_ROOT="$OUT_ROOT"
    -e TORCHELASTIC_ERROR_FILE="$ELASTIC_ERROR_FILE"
  )
  # Mount pre-trained checkpoint if provided
  if [[ -n "${COSMOS_H_CKPT_PATH:-}" ]]; then
    DOCKER_ARGS+=( -v "$COSMOS_H_CKPT_PATH:$COSMOS_H_CKPT_PATH:ro" )
    DOCKER_ARGS+=( -e COSMOS_H_CKPT_PATH="$COSMOS_H_CKPT_PATH" )
  fi
  # Mount HF cache so container can use host token/cache (e.g. from hf auth login)
  if [[ -d "$HF_HOME_HOST" ]]; then
    DOCKER_ARGS+=( -v "$HF_HOME_HOST:/root/.cache/huggingface" -e HF_HOME=/root/.cache/huggingface )
  fi
  # Build optional checkpoint override for Docker command string
  CKPT_OVERRIDE_ARG=""
  if [[ -n "${COSMOS_H_CKPT_PATH:-}" ]]; then
    CKPT_OVERRIDE_ARG=" checkpoint.load_path=${COSMOS_H_CKPT_PATH}"
  fi
  # Preflight: ensure GPUs are visible in container (avoids "CUDA device busy or unavailable").
  TRAIN_CMD="source .venv/bin/activate 2>/dev/null || true;
echo '=== GPU preflight ===';
nvidia-smi -L || true;
python -c \"import torch; n=torch.cuda.device_count(); print('PyTorch sees', n, 'GPU(s):', [torch.cuda.get_device_name(i) for i in range(n)] if n else 'none')\" || true;
if [[ -n \"\${WANDB_API_KEY:-}\" ]]; then wandb login --relogin \"\$WANDB_API_KEY\"; fi;
echo '=== Starting training ===';
torchrun --nnodes=1 --nproc_per_node=\"\${NGPUS:-8}\" --master_addr=\"\${MASTER_ADDR:-127.0.0.1}\" --master_port=\"\${MASTER_PORT:-25001}\" -m scripts.train --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py -- experiment=\"cosmos_predict2p5_2B_action_conditioned_suturebot_13frame_4nodes_release_oss\" checkpoint.save_iter=200 '~dataloader_train.dataloaders'${CKPT_OVERRIDE_ARG}"
  TTY_FLAG=$([ -t 0 ] && echo "-it" || echo "-i")
  docker run $TTY_FLAG --rm \
    "${DOCKER_ARGS[@]}" \
    -w /workspace \
    "$COSMOS_CONTAINER_IMAGE" \
    bash -c "$TRAIN_CMD"
else
  # Run on host: use CODE_PATH as workspace and venv there.
  cd "$CODE_PATH"
  if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
  elif [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    echo "Using conda env: $CONDA_DEFAULT_ENV"
  else
    echo "Warning: no .venv found in $CODE_PATH; ensure dependencies are installed (see setup.md)."
  fi
  _run_train
fi
