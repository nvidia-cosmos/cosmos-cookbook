# Setup Guide

## System Requirements

* NVIDIA GPUs with Ampere architecture (RTX Pro 6000, A100, H100) or newer
* NVIDIA driver >=570.124.06 compatible with [CUDA 12.8.1](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html#cuda-toolkit-major-component-versions)
* Linux x86-64
* glibc>=2.35 (e.g Ubuntu >=22.04)
* Python 3.10

## Installation

Clone the necessary repositories:

```shell
git clone https://github.com/nvidia-cosmos/cosmos-predict2.5.git
git clone https://github.com/nvidia-cosmos/cosmos-reason2.git
```

Install system dependencies for Predict2.5:

[uv](https://docs.astral.sh/uv/getting-started/installation/)

```shell
cd cosmos-predict2.5
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Install the package into a new environment:

```shell
uv sync --extra=cu128
source .venv/bin/activate
```

Or, install the package into the active environment (e.g. conda):

```shell
uv sync --extra=cu128 --active --inexact
```

CUDA variants:

* `--extra=cu128`: CUDA 12.8
* `--extra=cu129`: CUDA 12.9

## Downloading Checkpoints

1. Get a [Hugging Face Access Token](https://huggingface.co/settings/tokens) with `Read` permission
2. Install [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli): `uv tool install -U "huggingface_hub[cli]"`
3. Login: `hf auth login` and enter the token created in Step 1.
4. Accept the [NVIDIA Open Model License Agreement](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B).

Checkpoints are automatically downloaded during inference and post-training. To modify the checkpoint cache location, set the [HF_HOME](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhome) environment variable.

> **💡 Tip**: Ensure you have sufficient disk space in `HF_HOME`.

## Training Output Directory

Configure where training checkpoints and artifacts will be saved:

```bash
# Set output directory for training checkpoints and artifacts
# Default: /tmp/imaginaire4-output
export IMAGINAIRE_OUTPUT_ROOT=/path/to/your/output/directory
```

> **💡 Tip**: Ensure you have sufficient disk space in `IMAGINAIRE_OUTPUT_ROOT`.

## Weights & Biases (W&B) Logging

By default, training will attempt to log metrics to Weights & Biases. You have several options:

### Option 1: Enable W&B

To enable full experiment tracking with W&B:

1. Create a free account at [wandb.ai](https://wandb.ai)
2. Get your API key from [https://wandb.ai/authorize](https://wandb.ai/authorize)
3. Set the environment variable:

    ```bash
    export WANDB_API_KEY=your_api_key_here
    ```

> ⚠️ **Security Warning:** Store API keys in environment variables or secure vaults. Never commit API keys to source control.

### Option 2: Disable W&B

Add `job.wandb_mode=disabled` to your training command to disable wandb logging.

## Next Steps

Once the setup is complete, proceed to the [post-training tutorial](post-training.md) to learn how to train Cosmos Predict 2.5 on Gr00t Trajectories.
