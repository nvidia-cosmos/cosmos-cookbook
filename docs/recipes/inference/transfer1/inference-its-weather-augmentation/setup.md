# Setup and System Requirements

This guide covers the setup requirements for running Cosmos Transfer 1 for weather augmentation of ITS images.

## System Requirements

### Minimum Hardware Requirements

- **GPU**: 1 or more GPUs (A100, H100, or later recommended)
- **Memory**: Sufficient VRAM for model inference
- **Storage**: Adequate disk space for model weights

### Software Requirements

The setup requires the Cosmos Transfer 1 repository and model to be properly installed and configured.

## Installation

### Cosmos Transfer 1 Setup

To set up Cosmos Transfer 1 repository and model, follow the detailed installation and inference setup instructions at:

**[Cosmos Transfer 1 Installation Guide](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/INSTALL.md#inference)**

The installation guide provides comprehensive steps for:

- Repository cloning and setup
- Environment configuration
- Model weight downloads
- Dependency installation
- Inference configuration

### Verification

After completing the installation, verify the setup by running the inference examples provided in the Cosmos Transfer 1 repository to ensure the model is working correctly before proceeding with the weather augmentation pipeline.

## Next Steps

Once the setup is complete, proceed to the [inference tutorial](inference.md) to learn how to use Cosmos Transfer 1 for weather augmentation of ITS images.
