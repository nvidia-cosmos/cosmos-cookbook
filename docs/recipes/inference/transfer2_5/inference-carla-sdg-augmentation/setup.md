# Setup and System Requirements

This guide covers the setup requirements for running Cosmos Transfer 2.5 for video augmentation of synthetic data from simulators.

## System Requirements

### Minimum Hardware Requirements

- **GPU**: Single A100, H100, or later GPU (recommended minimum)
- **Memory**: Sufficient VRAM for model inference
- **Storage**: Adequate space for model weights

### Software Requirements

The setup requires the Cosmos Transfer 2.5 repository and model to be properly installed and configured.

## Installation

### Cosmos Transfer 2.5 Setup

To set up Cosmos Transfer 2.5 repository and model, follow the detailed installation and inference setup instructions at:

**[Cosmos Transfer 2.5 Installation Guide](https://github.com/nvidia-cosmos/cosmos-transfer2.5)**

The installation guide provides comprehensive steps for:

- Repository cloning and setup
- Environment configuration
- Model weight downloads
- Dependency installation
- Inference configuration

### Verification

After completing the installation, verify the setup by running the inference examples provided in the Cosmos Transfer 2.5 repository to ensure the model is working correctly before proceeding with the simulator augmentation pipeline.

## Next Steps

Once the setup is complete, proceed to the [inference tutorial](inference.md) to learn how to use Cosmos Transfer 2.5 for simulator-to-real data augmentation.
