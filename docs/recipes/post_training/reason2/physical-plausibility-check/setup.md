# Setup and System Requirements

This guide covers the setup requirements for post-training Cosmos Reason 2 for physical plausibility prediction.

## System Requirements

### Minimum Hardware Requirements

- **GPU**: 8 A100, H100, or later GPU (recommended minimum)
- **Memory**: Sufficient VRAM for model inference
- **Storage**: Adequate space for model weights

### Software Requirements

- **FFmpeg**: Required for video processing
  ```bash
  # Ubuntu/Debian
  sudo apt-get update && sudo apt-get install -y ffmpeg
  
  # macOS
  brew install ffmpeg
  ```

- **Cosmos Reason 2 Repository**: Clone from [GitHub](https://github.com/nvidia-cosmos/cosmos-reason2) and follow the installation instructions below.

## Installation

### Cosmos Reason 2 Setup

Follow the detailed installation and inference setup instructions at:
**[Cosmos Reason 2 Post-Training Installation Guide](https://github.com/nvidia-cosmos/cosmos-reason2/blob/main/examples/cosmos_rl/README.md)**

This includes:

- Installing uv and other dependencies
- Setting up the virtual environment or Docker container
- Installing redis (required system dependency)
- Running `uv sync` in the `examples/cosmos_rl` directory
- Optional: Setting up wandb for monitoring

### Verification

After completing the installation, verify the setup by running the inference examples provided in the Cosmos Reason 2 repository to ensure the model is working correctly before proceeding with the post-training pipeline.

## Next Steps

Once the setup is complete, proceed to the [post-training tutorial](post_training.md) to learn how to post-train Cosmos Reason 2 for physical plausibility prediction.
