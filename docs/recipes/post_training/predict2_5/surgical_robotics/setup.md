# Setup Guide

Run this setup **after** you clone the repository in [README §1.1](README.md#11-clone-the-repository). Complete all steps below before continuing with the rest of the post-training tutorial.

## System Requirements

* NVIDIA GPUs with Ampere architecture (RTX 30 Series, A100) or newer
* NVIDIA driver >=570.124.06 compatible with [CUDA 12.8.1](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html#cuda-toolkit-major-component-versions)
* Linux x86-64
* glibc>=2.35 (e.g Ubuntu >=22.04)
* Python 3.10

## Installation

Install system dependencies:

```shell
sudo apt install curl ffmpeg parallel tree wget
```

[uv](https://docs.astral.sh/uv/getting-started/installation/)

```shell
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

For the SutureBot→LeRobot conversion (Post-training §2.3), install lerobot 0.3.3 in the same environment:

```shell
uv pip install lerobot==0.3.3
```

CUDA variants:

* `--extra=cu128`: CUDA 12.8
* `--extra=cu129`: CUDA 12.9

## Downloading Checkpoints

1. Get a [Hugging Face Access Token](https://huggingface.co/settings/tokens) with `Read` permission
2. Install [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli): `uv tool install -U huggingface_hub`
3. Login: `hf auth login`
4. Accept the [NVIDIA Open Model License Agreement](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B).

Checkpoints are automatically downloaded during inference and post-training. To modify the checkpoint cache location, set the [`HF_HOME`](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhome) environment variable.

## Optional: Slurm

To run finetuning with **Slurm** (`sbatch scripts/run_finetuning.sh`) on your own cluster, see [Slurm setup](setup_slurm.md). If you have a single server without Slurm, use [run_finetuning_standalone.sh](scripts/run_finetuning_standalone.sh) instead (see [README §4](README.md#4-finetuning)).

## Next Steps

Once this setup is complete, return to the [post-training tutorial](README.md) and continue from **§1.5 Hugging Face Configuration** onward.
