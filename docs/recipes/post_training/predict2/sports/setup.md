# Setup and System Requirements

This guide covers the setup requirements for running Cosmos Predict 2 for traffic anomaly generation using LoRA post-training.

## System Requirements

### Minimum Hardware Requirements

- **GPU**: Single node with 8 GPUs (A100, H100, or later recommended)
- **Memory**: Sufficient VRAM for model inference and training (50GB+ recommended)
- **Storage**: Adequate disk space for model weights, datasets, and checkpoints

### Software Requirements

Setup requires the Cosmos Predict 2 repository and model to be properly installed and configured.

## Installation

### Cosmos Predict 2 Setup

To set up the Cosmos Predict 2 repository and model, follow the installation and inference setup instructions in the
[Cosmos Predict 2 Setup Guide](https://github.com/nvidia-cosmos/cosmos-predict2/blob/main/documentations/setup.md).

### Verification

Before proceeding with the post-training pipeline, verify that the model is working correctly by running [inference examples](https://github.com/nvidia-cosmos/cosmos-predict2?tab=readme-ov-file#user-guide) provided in the Cosmos Predict 2 repository.

## Next Steps

Once the setup and verification are complete, proceed to the [post-training tutorial](post_training.md) to learn how to use Cosmos Predict 2 for traffic anomaly generation with LoRA adaptation.
