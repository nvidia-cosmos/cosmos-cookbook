# Leveraging World Foundation Models for Synthetic Trajectory Generation in Robot Learning

> **Author:** [Rucha Apte](https://www.linkedin.com/in/ruchaa-apte/), [Saurav Nanda](https://www.linkedin.com/in/sauravnanda/), [Jingyi Jin](https://www.linkedin.com/in/jingyi-jin/)
> **Organization:** NVIDIA

| **Model** | **Workload** | **Use Case** |
|-----------|--------------|--------------|
| [Cosmos Predict 2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) | Post-training, Inference | Synthetic Trajectory Generation |
| [Cosmos Reason 2](https://github.com/nvidia-cosmos/cosmos-reason2) | Inference | Reasoning and filtering synthetic trajectories |

This guide walks you through post-training the Cosmos Predict 2.5 model on the [PhysicalAI-Robotics-GR00T-GR1](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-GR1) open dataset to generate synthetic robot trajectories for robot learning applications. After post-training, we'll use the fine-tuned model to generate trajectory predictions on the [PhysicalAI-Robotics-GR00T-Eval](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Eval) dataset. Finally, Cosmos Reason 2 is leveraged to evaluate these generated trajectories by assessing their physical plausibility, helping to quantify and filter for valid, realistic, and successful robot motions.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Preparing Data](#1-preparing-data)
- [Post-training Predict 2.5](#2-post-training-predict2.5)
  - [Configuration](#21-configuration)
  - [Training](#22-training)
- [Inference with Post-trained Predict 2.5 checkpoint](#3-inference-with-post-trained-predict2.5-checkpoint)
  - [Converting DCP Checkpoint to Consolidated PyTorch Format](#31-converting-dcp-checkpoint-to-consolidated-pytorch-format)
  - [Running Inference](#32-running-inference)
- [Inference with Reason 2 for plausibilty check and filtering](#4-inference-with-Reason2-for-plausibility-check-and-filtering)

## Prerequisites

### 1. Environment Setup

Follow the [Setup guide](./setup.md) for general environment setup instructions, including installing dependencies.

### 2. Hugging Face Configuration

Model checkpoints are automatically downloaded during post-training if they are not present. Configure Hugging Face as follows:

```bash
# Login with your Hugging Face token (required for downloading models)
hf auth login

# Set custom cache directory for HF models
# Default: ~/.cache/huggingface
export HF_HOME=/path/to/your/hf/cache
```

> **💡 Tip**: Ensure you have sufficient disk space in `HF_HOME`.

### 3. Training Output Directory

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
