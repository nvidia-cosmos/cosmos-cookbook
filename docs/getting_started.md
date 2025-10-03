# Getting Started

This comprehensive guide covers the essential tools and dependencies needed to set up your development environment for working with Cosmos models. These tools provide the foundation for data curation, model post-training, evaluation, and deployment workflows across all Cosmos projects.

## Prerequisites

Before starting, ensure you have:

- A Unix-like operating system (Linux, macOS, or WSL on Windows)
- Administrator/sudo access for system installations
- Internet connection for downloading packages and models
- Git installed on your system

## Generic Tool Installation

### Package Managers

#### pkgx

[pkgx](https://docs.pkgx.sh/) is a modern package manager that simplifies CLI tool installation and management. It provides isolated environments and automatic dependency resolution.

```shell
brew install pkgx || curl https://pkgx.sh | sh
```

### Python Environment Management

#### uv

[uv](https://docs.astral.sh/uv/) is a fast Python package installer and resolver, designed as a drop-in replacement for pip. It's essential for managing Python dependencies in Cosmos projects.

```shell
pkgm install uv
```

### Hugging Face

The [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) is essential for downloading pre-trained model checkpoints and datasets from the Hugging Face Hub.

```shell
pkgm install huggingface-cli
hf auth login
```

**Note:** You'll need a Hugging Face account and access token for authentication.

### AWS (Required for Data Curation)

AWS services are required for data storage and processing in the curation pipeline. You'll need your own AWS S3 bucket and appropriate credentials.

#### Basic AWS CLI Setup

```shell
pkgm install aws
aws configure
```

**Configuration requirements:**

- AWS Access Key ID
- AWS Secret Access Key
- Default region (e.g., `us-west-2`)
- Output format (recommend `json`)

For detailed guidance, see the [AWS CLI User Guide](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-welcome.html).

**s5cmd CLI** - High-performance S3 operations tool for large file transfers:

```shell
uv tool install -U s5cmd
```

### Weights & Biases (Optional for Training)

[Wandb](https://github.com/wandb/wandb) provides experiment tracking, model versioning, and collaborative features for machine learning projects. Recommended for post-training workflows.

```shell
uv tool install -U wandb
wandb login
```

## Repository Setup

Clone the Cosmos Cookbook repository and install it in development mode:

```shell
git clone git@github.com:nvidia-cosmos/cosmos-cookbook.git
cd cosmos-cookbook
just install
source .venv/bin/activate
```

**Note:** These installation steps will be updated as we prepare the external repository for public release.
