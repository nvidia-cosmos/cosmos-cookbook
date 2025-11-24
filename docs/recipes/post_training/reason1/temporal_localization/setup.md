# Setup and System Requirements

This guide covers the setup requirements for running Cosmos Reason 1 for MimicGen temporal localization and post-training workflows.

## System Requirements

### Minimum Hardware Requirements

- **GPU**: 1 or more GPUs (A100, H100, or later recommended)
- **Memory**: 24 GB VRAM for model inference
- **Storage**: Adequate disk space for model weights

### Software Requirements

The setup requires the Cosmos Reason 1 repository and model to be properly installed and configured.

## Installation

### Cosmos Reason 1 Setup

To set up Cosmos Reason 1 repository and model, follow the [Setup steps](https://github.com/nvidia-cosmos/cosmos-reason1/tree/main/examples/post_training_hf#setup) in the Cosmos Reason 1 Post-Training Hugging Face Example.

### Verification

After completing the installation, run the Cosmos Reason 1 [inference examples](https://github.com/nvidia-cosmos/cosmos-reason1/blob/main/README.md#inference) to ensure the model is working correctly before proceeding with the post-training pipeline.

## Next Steps

Once setup is complete, proceed to the [post-training tutorial](post_training.md) to learn how to use Cosmos Reason 1 for MimicGen temporal localization.
