# Curate data for Cosmos Predict Fine-Tuning using Cosmos Curator

> **Authors:** [Hao Wang](https://www.linkedin.com/in/pkuwanghao/) • NVIDIA Cosmos Curator Team
> **Organization:** NVIDIA

| **Model** | **Workload** | **Use Case** |
|-----------|--------------|--------------|
| [Cosmos Curator](https://github.com/nvidia-cosmos/cosmos-curate) | Data Curation | Video data curation for Predict 2 post-training |

## Overview

This recipe demonstrates how to curate video data for fine-tuning Cosmos Predict 2 models using Cosmos Curator. You'll learn how to transform raw videos into a structured dataset with semantic scene splits, AI-generated captions, and quality filtering—all in the format required by Cosmos Predict 2.

This guide focuses on a minimal, end-to-end workflow that you can run locally with Docker. We'll use a real example dataset to walk through each step of the curation pipeline. For more advanced configurations and deployment options, refer to the [Cosmos Curator Documentation](https://github.com/nvidia-cosmos/cosmos-curate/blob/main/docs/README.md).

**What you'll learn**:

1. Set up Cosmos Curator with required models.
2. Prepare source video data.
3. Run the curation pipeline to generate training-ready datasets.
4. Explore advanced configuration options.

## Set up Cosmos Curator

### Clone and Install Cosmos Curator

These steps will give you a CLI that helps with the following tasks:

- Build a container image which includes the data curation pipelines.
- Download the required models from HuggingFace.
- Launch the pipeline in one of the following ways:
  - Locally using `docker`
  - On a `slurm` cluster
  - On an `NVCF` cluster

```bash
# Clone and install cosmos-curate
git clone --recurse-submodules https://github.com/nvidia-cosmos/cosmos-curate.git
cd cosmos-curate
uv venv --python=3.10 && source .venv/bin/activate
uv pip install poetry && poetry install --extras=local

# Verify you get the cosmos-curate CLI working
cosmos-curate --help
```

### Build the Container Image

Before building the container image, you need to install the following dependencies:

- [Docker](https://docs.docker.com/engine/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

You can then build the container image using the following command:

```bash
# Build the image; first build can take up to 30 minutes
cosmos-curate image build
```

### Download the Model Weights

For this recipe, you will download the following models:

- [TransnetV2](https://huggingface.co/Sn4kehead/TransNetV2) for semantically splitting a long video into short clips
- [Cosmos-Reason1-7B](https://huggingface.co/nvidia/Cosmos-Reason1-7B) for captioning the clips
- [T5-11B](https://huggingface.co/google-t5/t5-11b) for generating embeddings for caption text

```bash
cosmos-curate local launch -- pixi run python -m cosmos_curate.core.managers.model_cli download --models transnetv2,cosmos_reason1,t5_xxl
```

The model weights will be downloaded to your **cosmos-curate workspace**, which is by default at `${HOME}/cosmos_curate_local_workspace`.
If you want to change the **workspace location**, you can define environment variable `COSMOS_CURATE_LOCAL_WORKSPACE_PREFIX` to change it to `${COSMOS_CURATE_LOCAL_WORKSPACE_PREFIX}/cosmos_curate_local_workspace`.

## Prepare the Source Data

These instructions will use [nexar_collision_prediction](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/tree/main) data as an example.
You will need to download the data into the **workspace**.

```bash
# Go to the workspace
pushd .
cd "${COSMOS_CURATE_LOCAL_WORKSPACE_PREFIX:-$HOME}/cosmos_curate_local_workspace"

# Download a few videos into the workspace
mkdir -p nexar_collision_prediction/train/positive/
cd nexar_collision_prediction/train/positive/
wget https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/resolve/main/train/positive/00000.mp4
wget https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/resolve/main/train/positive/00003.mp4
wget https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/resolve/main/train/positive/00004.mp4
wget https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/resolve/main/train/positive/00005.mp4
wget https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/resolve/main/train/positive/00006.mp4
wget https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/resolve/main/train/positive/00007.mp4
wget https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/resolve/main/train/positive/00008.mp4
wget https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/resolve/main/train/positive/00010.mp4

# let's go back
popd
```

## Run the Curation Pipeline

```bash
cosmos-curate local launch -- pixi run python -m cosmos_curate.pipelines.video.run_pipeline split \
    --input-video-path /config/nexar_collision_prediction/train/positive/ \
    --output-clip-path /config/output-nexar/ \
    --no-generate-embeddings \
    --generate-cosmos-predict-dataset predict2 \
    --splitting-algorithm transnetv2 \
    --transnetv2-min-length-frames 120 \
    --captioning-algorithm cosmos_r1 \
    --limit 0
```

> **Note:** For detailed information about all available pipeline parameters and configuration options, refer to the [Video Pipeline Reference Documentation](https://github.com/nvidia-cosmos/cosmos-curate/blob/main/docs/curator/REFERENCE_PIPELINES_VIDEO.md).

When running the curation pipeline, consider the following important notes:

- The `--input-video-path` and `--output-clip-path` options expect paths **inside the container**.
  - The **workspace** directory `${COSMOS_CURATE_LOCAL_WORKSPACE_PREFIX:-$HOME}/cosmos_curate_local_workspace` mounts to `/config` in the container.
- With the `--generate-cosmos-predict-dataset predict2` option, you will get a `cosmos_predict2_video2world_dataset/` directory under your specified output path (i.e. `${COSMOS_CURATE_LOCAL_WORKSPACE_PREFIX:-$HOME}/cosmos_curate_local_workspace/output-nexar`).
  - This is the exact dataset format to [post-train/fine-tune a Cosmos Predict 2 model](https://github.com/nvidia-cosmos/cosmos-predict2/blob/main/documentations/post-training_video2world.md#1-preparing-data).
- The `--transnetv2-min-length-frames 120` option is added to specify the minimum length of clips because the Cosmos Predict 2 post-training script expects a minimum number of frames.
- The `--limit 0` option applies no limit to the number of input videos to process.
  - The data curation pipeline can fail with a system out-of-memory (OOM) error if you have two or fewer GPUs in the system or limited system memory with input videos that are too long or numerous.
  - If an out-of-memory (OOM) error occurs, you can use a small limit like `--limit 1` and run the exact same command repeatedly. Cosmos Curator can remember which commands have been executed and which remain to be done.

### Produce the WebDataset Format

Cosmos Curator has another `Shard-Dataset` pipeline that takes the output of the above `Split-Annotate` pipeline and generates a webdataset.

```bash
cosmos-curate local launch -- pixi run python -m cosmos_curate.pipelines.video.run_pipeline shard \
    --input-clip-path /config/output-nexar/ \
    --output-dataset-path /config/webdataset \
    --annotation-version v0
```

Again, the **workspace** directory `"${COSMOS_CURATE_LOCAL_WORKSPACE_PREFIX:-$HOME}/cosmos_curate_local_workspace"` mounts to `/config` in the container,
so you will find the webdataset under `"${COSMOS_CURATE_LOCAL_WORKSPACE_PREFIX:-$HOME}/cosmos_curate_local_workspace/webdataset/v0/"`.

By default, the generated webdataset is sharded by resolution, aspect ratio, and frame window index within the video clip.
More details can be found in the [documentation](https://github.com/nvidia-cosmos/cosmos-curate/blob/main/docs/curator/REFERENCE_PIPELINES_VIDEO.md#shard-dataset-pipeline).

## Advanced Options

The CLI and pipeline commands have many configurable options. Use `cosmos-curate --help` or `cosmos-curate local launch -- pixi run python -m cosmos_curate.pipelines.video.run_pipeline split --help` to explore all available parameters.

### Storage Configuration

`Cosmos Curate` supports all S3-compatible object stores (e.g. AWS, OCI, Swiftstack, GCP). You only need to configure your `~/.aws/credentials` file properly and pass in an S3 prefix.

**Documentation:** [Initial Setup - Storage Configuration](https://github.com/nvidia-cosmos/cosmos-curate/blob/main/docs/client/END_USER_GUIDE.md#initial-setup) (refer to bullet 5)

### Scaling to Production Platforms

Running locally on a desktop with one GPU is mainly for functional development. For production runs, consider deploying on multi-GPU clusters:

- [Launch Pipelines on Slurm](https://github.com/nvidia-cosmos/cosmos-curate/blob/main/docs/client/END_USER_GUIDE.md#launch-pipelines-on-slurm)
- [Deploy and Launch on NVCF](https://github.com/nvidia-cosmos/cosmos-curate/blob/main/docs/client/NVCF_GUIDE.md)
- [Reference Helm Chart for NVCF Deployment](https://github.com/nvidia-cosmos/cosmos-curate/tree/main/charts/cosmos-curate)
- A guide on vanilla K8s deployment will be added soon

### Pipeline Configuration Options

#### Splitting Algorithms

- **`transnetv2`** (default): Semantically splits videos at scene transitions.
- **`fixed-stride`**: Cuts source videos into fixed-length clips.

#### Captioning Algorithms

- **`cosmos_r1`** (default): Uses the Cosmos-Reason1-7B model.
- **`qwen`**: Uses the Qwen 2.5 VL model (requires adding `qwen2.5_vl` to the model download list).
- **`phi4`**: Uses the Phi-4 model (requires adding `phi_4` to the model download list).

#### Custom Captioning Prompts

Instead of using the default prompt, pass a custom prompt:

```bash
--captioning-prompt-text 'Describe the driving scene in detail, focusing on road conditions and vehicle behavior.'
```

#### Quality Filters

Enhance dataset quality by filtering out low-quality clips:

- **Motion filter**: `--motion-filter enable` (removes static/low-motion clips)
- **Aesthetic filter**: `--aesthetic-threshold 3.5` (filters low-quality or blurry frames)
- **VLM-based filter**: `--qwen-filter enable` (semantic filtering; requires adding `qwen2.5_vl` to model download list)

**Documentation**: [Split-Annotate Pipeline Reference](https://github.com/nvidia-cosmos/cosmos-curate/blob/main/docs/curator/REFERENCE_PIPELINES_VIDEO.md#split-annotate-pipeline-stages)

### Extending the Pipeline

If you need to add custom processing stages (e.g. new filtering logic, custom annotations), refer to the [Pipeline Design Guide](https://github.com/nvidia-cosmos/cosmos-curate/blob/main/docs/curator/PIPELINE_DESIGN_GUIDE.md).
