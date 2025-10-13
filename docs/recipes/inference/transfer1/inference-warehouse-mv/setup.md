# Setup and System Requirements

<<<<<<< HEAD
<<<<<<< HEAD
This guide covers the setup requirements for running Cosmos Transfer 1 for warehouse multi-view inference and inference for robotics navigation tasks.
=======
This guide covers the setup requirements for running Cosmos-Transfer1 for warehouse multi-view inference.
>>>>>>> 8d9847e (Copyedit recipes along with recent change sto core_concepts)
=======
This guide covers the setup requirements for running Cosmos Transfer 1 for warehouse multi-view inference.
>>>>>>> fa549ba (Re-align model names and remove merge artifact)

## System Requirements

### Minimum Hardware Requirements

- **GPU**: 1 or more GPUs (A100, H100, or later recommended)
- **Memory**: Sufficient VRAM for model inference
- **Storage**: Adequate disk space for model weights

### Software Requirements

The setup requires both the Cosmos Transfer 1 repository and the Cosmos Cookbook to be properly installed and configured.

## Installation

### Cosmos Cookbook Setup

Set up the Cosmos Cookbook repository and its dependencies by following the comprehensive setup tutorial:

**[Getting Started Guide](../../../get_started.md)**

The getting started guide provides detailed instructions for the following:

- Repository cloning and setup
- Environment configuration
- Essential tool installation (uv, Hugging Face CLI, AWS CLI, etc.)
- Development dependencies

### Cosmos Transfer 1 Setup

To set up Cosmos Transfer 1 repository and model, follow the [Cosmos Transfer 1 Installation Guide](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/INSTALL.md#inference) for detailed installation and inference setup instructions.

The installation guide provides comprehensive steps for the following:

- Repository cloning and setup
- Environment configuration
- Model weight downloads
- Dependency installation
- Inference configuration

After completing the installation, copy the example cookbook assets into your Cosmos Transfer 1 repository. Assuming your Cosmos Transfer 1 repository root is available as `$COSMOS_TRANSFER_ROOT`, run the following command:

```bash
cp -r scripts/examples/transfer1/* "$COSMOS_TRANSFER_ROOT/examples/cookbook/"
```

Replace `$COSMOS_TRANSFER_ROOT` with the real path to the Cosmos Transfer 1 directory path, or export the environment variable with the path.

#### Install ffmpeg (required for video preprocessing/postprocessing)

Install ffmpeg into your active Conda environment:

```bash
conda install -c conda-forge ffmpeg
```

## Next Steps

Once the setup is complete, proceed to the [inference tutorial](inference.md) to learn how to use Cosmos Transfer 1 for warehouse multi-view inference.
