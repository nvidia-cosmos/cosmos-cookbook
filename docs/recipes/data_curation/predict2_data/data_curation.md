# Curate data for Cosmos-Predict Fine-Tuning using Cosmos-Curate

> **Authors:** NVIDIA Cosmos-Curate Team
>
> **Organization:** NVIDIA

## Overview

This guide provides a minimal set of commands to get the end-to-end workflow running.
For more advanced options, we will provide links to [Cosmos-Curate Documentation](https://github.com/nvidia-cosmos/cosmos-curate/blob/main/docs/README.md).

We will cover the following steps:

1. setup `cosmos-curate`
2. prepare source video data
3. run curation pipeline
4. advanced options

## Setup Cosmos-Curate

### Clone and install Cosmos-Curate

This will give you a CLI that helps

- build container image which includes the data curation pipelines
- download required models from HuggingFace
- launch the pipeline
  - locally using `docker`
  - onto `slurm` cluster
  - onto `NVCF`

```bash
# clone & install cosmos-curate
git clone --recurse-submodules https://github.com/nvidia-cosmos/cosmos-curate.git
cd cosmos-curate
uv venv --python=3.10 && source .venv/bin/activate
uv pip install poetry && poetry install --extras=local

# verify you get the cosmos-curate CLI working
cosmos-curate --help
```

### Build the Container Image

Pre-requisite:

- install [Docker](https://docs.docker.com/engine/install/)
- install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

```bash
# Build the image; first build can take up to 30 minutes
cosmos-curate image build
```

### Download model weights

For this recipe, we will only download
- [TransnetV2](https://huggingface.co/Sn4kehead/TransNetV2) for semantically splitting a long video into short clips.
- [Cosmos-Reason1-7B](https://huggingface.co/nvidia/Cosmos-Reason1-7B) for captioning the clips.
- [T5-11B](https://huggingface.co/google-t5/t5-11b) for generating embeddings for caption text.

```bash
cosmos-curate local launch -- pixi run python -m cosmos_curate.core.managers.model_cli download --models transnetv2,cosmos_reason1
```

The model weights will be downloaded to your **cosmos-curate workspace**, which by default is at `"${HOME}/cosmos_curate_local_workspace"`.
In case you want to change the **workspace location**, you can define environment variable `COSMOS_CURATE_LOCAL_WORKSPACE_PREFIX` to change it to `"${COSMOS_CURATE_LOCAL_WORKSPACE_PREFIX}/cosmos_curate_local_workspace"`.

## Prepare Source Data

Let's use [nexar_collision_prediction](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/tree/main) data as an example.
We will need to download the data into the **workspace**.

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

## Run Curation Pipeline

```bash
cosmos-curate local launch -- pixi run python -m cosmos_curate.pipelines.video.run_pipeline split \
    --input-video-path /config/nexar_collision_prediction/train/positive/ \
    --output-clip-path /config/output-nexar-v0/ \
    --no-generate-embeddings \
    --generate-cosmos-predict-dataset predict2 \
    --splitting-algorithm transnetv2 \
    --transnetv2-min-length-frames 120 \
    --captioning-algorithm cosmos_r1 \
    --limit 0
```

A few **very important** notes:

- Options `--input-video-path` & `--output-clip-path` expect paths **inside the container**.
  - The **workspace** directory `"${COSMOS_CURATE_LOCAL_WORKSPACE_PREFIX:-$HOME}/cosmos_curate_local_workspace"` mounts to `/config` in the container.
- With `--generate-cosmos-predict-dataset predict2` you will get a directory `cosmos_predict2_video2world_dataset/` under your specified output path, i.e. `"${COSMOS_CURATE_LOCAL_WORKSPACE_PREFIX:-$HOME}/cosmos_curate_local_workspace/output-nexar-v0"`.
  - This is the exact dataset format to [post-train/fine-tune Cosmos-Predict2 model](https://github.com/nvidia-cosmos/cosmos-predict2/blob/main/documentations/post-training_video2world.md#1-preparing-data).
- Option `--limit 0` applies no limit to the number of input videos to process.
  - The data curation pipeline can fail with system out-of-memory (OOM) error if
    - you have `<=2 GPUs` in the system
      - i.e. there is not enough GPU to keep at least one active worker per pipeline stage to allow videos **streaming** through, leading to much higher memory/storage requirement.
    - you have limited system memory but many/long input videos
      - i.e. `Cosmos-Curate` mainly targets systems with 4x/8x GPUs and 1TB+ memory for production runs; running on desktop workstations is primarily for pipeline development.
  - So if OOM happens, you can use a small limit like `--limit 1` and run the exact same command repeatedly.
    - `Cosmos-Curate` is smart enough to figure out what has done and what remains to be done.

## Advanced Options

Both the CLI (`cosmos-curate --help`)
and the split-annotate pipeline (`cosmos-curate local launch -- pixi run python -m cosmos_curate.pipelines.video.run_pipeline split --help`)
have many configurable options.

Below are some advanced things you can do

| Topic | Supported Options | Documentation Link |
| ----- | ----------------- | ------------------ |
| Storage | `Cosmos-Curate` supports all S3-compatible object store (e.g. AWS, OCI, Swiftstack, GCP), you only need to configure your `~/.aws/credentials` file properly and pass in an S3 prefix. | See bullet 5 in the [Initial Setup section](https://github.com/nvidia-cosmos/cosmos-curate/blob/main/docs/client/END_USER_GUIDE.md#initial-setup) |
| Target Platform | As mentioned above, running locally on desktop with one GPU is mainly for functional development; for production runs, you may want to check out how to run on Slurm and K8s-based systems. | <ul><li>[Launch Pipelines on Slurm](https://github.com/nvidia-cosmos/cosmos-curate/blob/main/docs/client/END_USER_GUIDE.md#launch-pipelines-on-slurm)</li><li>[Deploy and launch on NVCF](https://github.com/nvidia-cosmos/cosmos-curate/blob/main/docs/client/NVCF_GUIDE.md)</li><li>[Reference Helm Chart used for NVCF deployment](https://github.com/nvidia-cosmos/cosmos-curate/tree/main/charts/cosmos-curate)</li><li>Guide on vanilla K8s deployment will be added soon.</li></ul> |
| Pipeline Configuration | <ul><li>splitting algorithm: another option is `fixed-stride` to cut source videos into fixed-length clips.</li><li>captioning prompt: instead of using default prompt, you can pass in a custom prompt using `--captioning-prompt-text '...'`</li><li>filters: the reference pipeline includes<ul><li>motion-filter (`--motion-filter enable`)</li><li>aesthetic filter (`--aesthetic-threshold 3.5`)</li><li>experimental VLM-based filter (`--qwen-filter enable`); note you will need add `qwen2.5_vl` to the model download list.</li></ul></li></ul> | [Split-Annotate Pipeline](https://github.com/nvidia-cosmos/cosmos-curate/blob/main/docs/curator/REFERENCE_PIPELINES_VIDEO.md#split-annotate-pipeline-stages) |
| Adding New Functionality | If you need e.g. add new filtering stages... | [Pipeline Design Guide](https://github.com/nvidia-cosmos/cosmos-curate/blob/main/docs/curator/PIPELINE_DESIGN_GUIDE.md) |
