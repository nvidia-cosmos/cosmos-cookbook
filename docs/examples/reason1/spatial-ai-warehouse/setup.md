# Setup and System Requirements

This guide covers the setup requirements for post-training Cosmos-Reason1 for the Spatial AI for Warehouse application.

## System Requirements

### Minimum Hardware Requirements

- **GPU**: 8 or more GPUs (A100, H100, or later recommended)
- **Memory**: Sufficient VRAM for model inference
- **Storage**: Adequate disk space for model weights

### Software Requirements

Setup requires the Cosmos-Reason1 and cosmos-rl repositories and Cosmos-Reason1 model to be properly installed and configured.

## Installation

### Cosmos-Reason1 Setup

To set up the Cosmos-Reason1 repository and model, follow the detailed installation and setup instructions in [Cosmos Reason1 Post-Training Setup Guide](https://github.com/nvidia-cosmos/cosmos-reason1/blob/main/examples/post_training/README.md#setup).

### Verification

After completing the installation, run the Cosmos-Reason1 [inference examples](https://github.com/nvidia-cosmos/cosmos-reason1/blob/main/README.md#inference) to ensure the model is working correctly before proceeding with the post-training pipeline.

## Next Steps

Once the setup is complete, proceed to the [post-training tutorial](post_training.md) to learn how to post-train Cosmos-Reason1 for spatial AI for warehouse scenarios.
