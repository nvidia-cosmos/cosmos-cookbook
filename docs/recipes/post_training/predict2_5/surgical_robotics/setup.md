# Setup Guide

Complete all steps below before continuing with the [post-training tutorial](post_training.md). This guide assumes you have already cloned the repositories in [step 1.1](post_training.md#11-clone-the-cosmos-cookbook) and [step 1.3](post_training.md#13-clone-the-cosmos-h-surgical-simulator-repository).

## System Requirements

* NVIDIA GPUs with Ampere architecture (RTX 30 Series, A100) or newer
* NVIDIA driver >=570.124.06 compatible with [CUDA 12.8.1](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html#cuda-toolkit-major-component-versions)
* Linux x86-64
* glibc >= 2.35 (e.g., Ubuntu >= 22.04)
* Python 3.10

## Installation

Install system dependencies:

```shell
sudo apt install curl ffmpeg parallel tree wget
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/):

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

From the Cosmos-H-Surgical-Simulator repo cloned in [step 1.3](post_training.md#13-clone-the-cosmos-h-surgical-simulator-repository), install the package into a new environment:

```shell
cd $COSMOS_CODE_PATH   # e.g. /path/to/Cosmos-H-Surgical-Simulator
uv sync --extra=cu128
source .venv/bin/activate
```

Or, install the package into the active environment (e.g. conda):

```shell
cd $COSMOS_CODE_PATH
uv sync --extra=cu128 --active --inexact
```

For the SutureBot→LeRobot conversion and inference scripts, install additional dependencies:

```shell
uv pip install lerobot==0.3.3 mediapy torchcodec tyro
```

CUDA variants:

* `--extra=cu128`: CUDA 12.8
* `--extra=cu129`: CUDA 12.9

## Downloading Checkpoints

1. Get a [Hugging Face Access Token](https://huggingface.co/settings/tokens) with `Read` permission
2. Install [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli): `uv tool install -U "huggingface_hub[cli]"`
3. Login: `hf auth login`
4. Accept the [NVIDIA Open Model License Agreement](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B).

Checkpoints are automatically downloaded during inference and post-training. To modify the checkpoint cache location, set the [`HF_HOME`](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhome) environment variable.

## Next Steps

Once this setup is complete, return to the [post-training tutorial](post_training.md) and continue from **step 1.5 Build the Docker Image (for Containerized Runs)** onward.
