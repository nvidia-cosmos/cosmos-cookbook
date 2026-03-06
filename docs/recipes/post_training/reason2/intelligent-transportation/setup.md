# Setup and System Requirements

This guide covers the setup requirements for post-training Cosmos Reason 2 for intelligent transportation.

## System Requirements

### Minimum Hardware Requirements

- **GPU**: 8xA100, H100, or later GPUs (recommended minimum).
- **Memory**: Sufficient VRAM for model training and inference. We recommend at least 80GB per GPU.
- **Storage**: Adequate space for model weights and datasets.

### Software Requirements

The setup requires the [Cosmos Reason 2](https://github.com/nvidia-cosmos/cosmos-reason2) and [Cosmos-RL](https://github.com/nvidia-cosmos/cosmos-rl) repositories and [Cosmos Reason 2-8B model](https://huggingface.co/nvidia/Cosmos-Reason2-8B) to be properly installed and configured.

## Installation

### Cosmos Reason 2 Setup

To set up Cosmos Reason 2 repository and model, follow the detailed installation and post-training environment setup instructions at:

**[Cosmos Reason 2 Post-Training Installation Guide](https://github.com/nvidia-cosmos/cosmos-reason2/blob/main/examples/cosmos_rl/README.md)**
**[Cosmos-RL Installation Documentation](https://nvidia-cosmos.github.io/cosmos-rl/quickstart/installation.html)**

### Verification

After completing the installation, verify the setup by running the inference examples provided in the [Cosmos Reason 2 repository](https://github.com/nvidia-cosmos/cosmos-reason2?tab=readme-ov-file#transformers) on your custom traffic video and question to ensure the model is working correctly before proceeding with the post-training pipeline.

## Next Steps

Once the setup is complete, proceed to the [post-training tutorial](post_training.md) to learn how to post-train Cosmos Reason 2 for intelligent transportation.
