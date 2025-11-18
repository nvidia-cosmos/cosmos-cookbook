# Setup and System Requirements

This guide covers the setup requirements for running Cosmos Predict2 to generate synthetic ITS images.

## System Requirements

### Minimum Hardware Requirements

- **GPU**: NVIDIA GPUs with Ampere architecture (RTX 30 Series, A100) or newer
- **Memory**: Sufficient VRAM for model inference
- **Storage**: Adequate space for model weights

### Software Requirements

The setup requires the Cosmos Predict2 repository and model to be properly installed and configured.

## Installation

### Cosmos Predict2 Setup

To set up the Cosmos Predict2 repository and environment, follow the official setup instructions:

**[Cosmos Predict2 Setup Guide](https://github.com/nvidia-cosmos/cosmos-predict2/blob/main/documentations/setup.md)**

The setup guide provides steps for:

- Repository cloning and setup
- Environment configuration
- Model weight downloads
- Dependency installation
- Installation via docker container
- Inference configuration

### Verification

After completing the installation, verify the setup by running the text-to-image inference walkthrough from the Cosmos Predict2 docs to ensure inference works end-to-end before proceeding with the ITS workflow:

See: [Cosmos Predict2 Text-to-Image Inference](https://github.com/nvidia-cosmos/cosmos-predict2/blob/main/documentations/inference_text2image.md)

## Next Steps

Once the setup is complete, proceed to the [inference tutorial](inference.md) to learn how to use Cosmos Predict2 for ITS image synthesis/inference.
