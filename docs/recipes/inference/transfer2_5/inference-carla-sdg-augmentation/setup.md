# Setup and System Requirements

This guide covers the setup requirements for running Cosmos-Transfer2.5 for video augmentation of synthetic data from simulators.

## System Requirements

### Minimum Hardware Requirements

- **GPU**: 1 or more GPUs (A100, H100, or later recommended)
- **Memory**: Sufficient VRAM for model inference
- **Storage**: Adequate disk space for model weights

### Software Requirements

The setup requires the Cosmos-Transfer2.5 repository and model to be properly installed and configured.

## Installation

### Cosmos-Transfer2.5 Setup

To set up Cosmos-Transfer2.5 repository and model, follow the [Cosmos-Transfer2.5 Setup Guide](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/docs/setup.md).

The installation guide provides the following comprehensive steps:

- Repository cloning and setup
- Environment configuration
- Model weight downloads
- Dependency installation
- Inference configuration

### Verification

After completing the installation, verify setup by running the [inference examples](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/docs/inference.md) provided in the Cosmos-Transfer2.5 repository to ensure the model is working correctly before proceeding with the simulator augmentation pipeline.

## Next Steps

Once setup is complete, proceed to the [inference tutorial](inference.md) to learn how to use Cosmos-Transfer2.5 for simulator-to-real data augmentation.
