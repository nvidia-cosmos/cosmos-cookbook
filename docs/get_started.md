# Getting Started

This guide covers the essential tools and dependencies needed to set up your development environment for working with Cosmos models. These tools provide the foundation for data curation, model post-training, evaluation, and deployment workflows across all Cosmos projects.

## Repository Setup

Clone the Cosmos Cookbook repository and install it in development mode:

```shell
git clone git@github.com:nvidia-cosmos/cosmos-cookbook.git
cd cosmos-cookbook
```

### Cookbook Structure

The Cosmos Cookbook is organized into two main directories:

- **`docs/`** - Contains the source documentation in markdown files. This includes all the technical guides, workflows, examples, and tutorials that make up the cookbook content.

- **`scripts/`** - Contains all the executable scripts referenced throughout the cookbook. This includes scripts for data processing, evaluation pipelines, configuration files for post-training tasks, and other automation tools used across the various workflows.

This structure separates the documentation from the practical implementation, making it easy to navigate between reading about workflows and executing the corresponding scripts.

**Note:** These installation steps will be updated as we prepare the external repository for public release.

## Prerequisites

Before getting started, ensure you have the following requirements:

### Hardware

**NVIDIA GPUs**: Not required for local documentation rendering. For running cookbook recipes and workflows: Ampere architecture or newer (A100, H100) - minimum 1 GPU, recommended 8 GPUs

### Software

- **Operating System**: Ubuntu 24.04, 22.04, or 20.04
- **Python**: Version 3.10+
- **NVIDIA Container Toolkit**: 1.16.2 or later
- **CUDA**: 12.4 or later
- **Docker Engine**
- **Access**: Internet connection for downloading models and dependencies

### Hardware Requirements

For specific GPU and memory requirements for each Cosmos model (Predict 1, Predict 2, Transfer 1, Reason 1), refer to the official [NVIDIA Cosmos Prerequisites](https://docs.nvidia.com/cosmos/latest/prerequisites.html) documentation.

## Generic Tool Installation

The following system dependencies are required to run the Cosmos Cookbook:

### pkgx

[pkgx](https://docs.pkgx.sh/) is a modern package manager that simplifies CLI tool installation and management. It provides isolated environments and automatic dependency resolution.

```shell
brew install pkgx || curl https://pkgx.sh | sh
```

### uv

[uv](https://docs.astral.sh/uv/) is a fast Python package installer and resolver, designed as a drop-in replacement for pip. It's essential for managing Python dependencies in Cosmos projects.

```shell
pkgm install uv
```

### Hugging Face CLI

The [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) is essential for downloading pre-trained model checkpoints and datasets from the Hugging Face Hub.

```shell
pkgm install huggingface-cli
hf auth login
```

**Note:** You'll need a Hugging Face account and access token for authentication.
