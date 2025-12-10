---
title: setup

---

# Setup and System Requirements

This guide covers the setup requirements for running Cosmos Reason 1 for wafer map classification task and post-training workflows.

## System Requirements

### Minimum Hardware Requirements

- **GPU**: 1 or more GPUs (A100, H100, or later recommended)
- **Memory**: Sufficient VRAM for model inference
- **Storage**: Adequate disk space for model weights

### Software Requirements

The setup requires the Cosmos Reason 1 repository and model to be properly installed and configured.

## Installation


### Install Cosmos Reason 1 Dependencies
+ Install uv and Hugging Face CLI

```
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv tool install -U "huggingface_hub[cli]"
hf auth login
```
### Clone repositories
+ Clone [cosmos-reason1](https://github.com/nvidia-cosmos/cosmos-reason1) and [cosmos-rl](https://github.com/nvidia-cosmos/cosmos-rl) inside cosmos-reason1
```
git clone https://github.com/nvidia-cosmos/cosmos-reason1
cd cosmos-reason1
git clone https://github.com/nvidia-cosmos/cosmos-rl
```
### Install the cosmos-rl Dependencies
+ **Prerequisites** : Ensure your system meets the minimum versions:
    + Python ≥ 3.9
    + CUDA ≥ 12.2
    + PyTorch ≥ 2.6.0
    + vLLM ≥ 0.8.5
+ Install Redis and other Python dependencies : 
```
apt-get update && apt-get install redis-server
cd cosmos-rl
pip install -r requirements.txt
pip install -e .
```


## Next Steps

Once setup is complete, proceed to the [post-training tutorial](post_training.md) to learn how to use Cosmos Reason 1 for physical plausibility augmentation of ITS images.
