# Setup and System Requirements

This guide covers the setup requirements for running Cosmos Predict2 for traffic anomaly generation using LoRA post-training.

## System Requirements

### Minimum Hardware Requirements

- **GPU**: Single node with 8 GPUs (A100, H100, or later recommended)
- **Memory**: Sufficient VRAM for model inference and training (50GB+ recommended)
- **Storage**: Adequate space for model weights, datasets, and checkpoints

### Software Requirements

The setup requires the Cosmos Predict2 repository and model to be properly installed and configured.

## Installation

### Cosmos Predict2 Setup

To set up Cosmos Predict2 repository and model, follow the detailed installation and inference setup instructions at:

**[Cosmos Predict2 Installation Guide](https://github.com/nvidia-cosmos/cosmos-predict2)**

### Verification

After completing the installation, verify the setup by running the inference examples provided in the Cosmos Predict2 repository to ensure the model is working correctly before proceeding with the post-training pipeline.

## Next Steps

Once the setup is complete, proceed to the [post-training tutorial](post_training.md) to learn how to use Cosmos Predict2 for traffic anomaly generation with LoRA adaptation.
