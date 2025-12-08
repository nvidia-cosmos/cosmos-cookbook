# Cosmos Cookbook

[![Documentation](https://img.shields.io/badge/docs-cosmos--cookbook-blue)](https://nvidia-cosmos.github.io/cosmos-cookbook/)
[![Contributing](https://img.shields.io/badge/contributing-guide-green)](CONTRIBUTING.md)

A comprehensive guide for working with the **NVIDIA Cosmos ecosystem**—a suite of World Foundation Models (WFMs) for real-world, domain-specific applications across robotics, simulation, autonomous systems, and physical scene understanding.

**📚 [View the Full Documentation →](https://nvidia-cosmos.github.io/cosmos-cookbook/)** — Step-by-step workflows, case studies, and technical recipes

<https://github.com/user-attachments/assets/bb444b93-d6af-4e25-8bd0-ca5891b26276>

## Latest Updates

| **Recipe** | **Model** | **Description** |
|------------|-----------|-----------------|
| [Sports Video Generation](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/predict2_5/sports/post_training.html) | Cosmos Predict 2.5 | LoRA post-training for sports video generation with improved player dynamics and rule coherence |
| [Distilling Cosmos Predict 2.5](https://nvidia-cosmos.github.io/cosmos-cookbook/core_concepts/distillation/distilling_predict2.5.html) | Cosmos Predict 2.5 | Model distillation using DMD2 to create a 4-step student model |
| [Smart City SDG Pipeline](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/end2end/smart_city_sdg/workflow_e2e.html) | Cosmos Transfer 2.5 + Reason 1 | End-to-end synthetic data generation for traffic scenarios with CARLA |
| [Temporal Localization for MimicGen](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/reason1/temporal_localization/post_training.html) | Cosmos Reason 1 | Automated timestamp annotation for robot learning data generation |
| [BioTrove Moths Augmentation](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/inference/transfer2_5/biotrove_augmentation/inference.html) | Cosmos Transfer 2.5 | Domain transfer pipeline for scarce biological datasets using FiftyOne |

## Prerequisites

| Use Case | Linux (Ubuntu) | macOS | Windows |
|----------|----------------|-------|---------|
| Running cookbook recipes (GPU workflows) | ✅ Supported | ❌ | ❌ |
| Local documentation & contribution | ✅ Supported | ✅ Supported | ⚠️ WSL recommended |

### For Documentation & Contribution (All Platforms)

- **Git** with [Git LFS](#1-install-git-lfs-required)
- **Python**: Version 3.10+
- **Internet access** for cloning and dependencies

### For Running Cookbook Recipes (Ubuntu Only)

Full GPU workflows require an Ubuntu Linux environment with NVIDIA GPUs.

→ See **[Getting Started](https://nvidia-cosmos.github.io/cosmos-cookbook/getting_started/setup.html)** for complete hardware and software requirements.

## Quick Start

### 1. Install Git LFS (Required)

> ⚠️ **Important**: This repository contains many media files (videos, images, demonstrations). Git LFS is **required** to clone and work with this repository properly.

```bash
# Ubuntu/Debian (recommended)
sudo apt update && sudo apt install git-lfs

# Enable Git LFS globally
git lfs install
```

For other platforms (macOS, Windows, Fedora), see the official installation guide at **[git-lfs.com](https://git-lfs.com/)**.

If you've already cloned without LFS, fetch the media files with:

```bash
git lfs pull
```

### 2. Install System Dependencies

```bash
# Install uv (fast Python package manager)
# Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Windows (PowerShell):
# powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install just (command runner) - all platforms
uv tool install -U rust-just
```

### 3. Clone and Setup Repository

```bash
# Clone the repository
git clone https://github.com/nvidia-cosmos/cosmos-cookbook.git
cd cosmos-cookbook

# Install dependencies and setup
just install
```

### 4. Explore the Documentation

```bash
# Serve documentation locally
just serve-external  # For public documentation
# or
just serve-internal   # For internal documentation (if applicable)
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

## Repository Structure

The Cosmos Cookbook is organized into two main directories:

### `docs/`

Contains the source documentation in markdown files:

- Technical guides and workflows
- End-to-end examples and case studies
- Step-by-step recipes and tutorials
- Getting started guides

### `scripts/`

Contains executable scripts referenced throughout the cookbook:

- Data processing and curation pipelines
- Model evaluation and quality control scripts
- Configuration files for post-training tasks
- Automation tools and utilities

This structure separates documentation from implementation, making it easy to navigate between reading about workflows and executing the corresponding scripts.

## Media File Guidelines

When contributing or working with media files in this repository:

### Recommended Format: MP4 with H.264

While `.gif` files offer universal browser compatibility, they suffer from:

- **Lower quality** due to limited 256-color palette
- **Larger file sizes** compared to modern video codecs
- **No audio support**

**We strongly recommend using `.mp4` files** with the following encoding settings for optimal quality, file size, and cross-browser compatibility:

```bash
# For silent videos (screen recordings, demos) - most common for documentation
ffmpeg -i input.mov -c:v libx264 -preset slow -crf 23 \
       -an \
       -pix_fmt yuv420p -movflags +faststart \
       output.mp4

# For videos with audio (tutorials with narration)
ffmpeg -i input.mov -c:v libx264 -preset slow -crf 23 \
       -c:a aac -b:a 128k \
       -pix_fmt yuv420p -movflags +faststart \
       output.mp4
```

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `-c:v libx264` | H.264 codec | Universal browser support (Chrome, Firefox, Safari, Edge) |
| `-preset slow` | Encoding speed | Better compression (use `medium` for faster encoding) |
| `-crf 23` | Quality factor | Range 18-28; lower = higher quality (23 is balanced) |
| `-pix_fmt yuv420p` | Pixel format | Required for browser/QuickTime compatibility |
| `-movflags +faststart` | Fast start | Enables progressive playback before full download |
| `-an` | No audio | Strips audio track (smaller file size) |
| `-c:a aac -b:a 128k` | AAC audio | Include when narration/audio is needed |

> 💡 **Tip**: Most documentation demos are silent—use `-an` to skip audio encoding for smaller files. For tutorials with narration, include the audio parameters.

## Available Commands

```bash
# Development
just install          # Install dependencies and setup
just setup            # Setup pre-commit hooks
just serve-external   # Serve public documentation locally
just serve-internal   # Serve internal documentation locally

# Quality Control
just lint            # Run linting and formatting
just test            # Run all tests and validation

# Continuous Integration
just ci-lint         # Run CI linting checks
just ci-deploy-internal         # Deploy internal documentation
just ci-deploy-external         # Deploy external documentation
```

## Contributing & Support

- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute to the cookbook
- **Report Issues**: Use [GitHub Issues](https://github.com/nvidia-cosmos/cosmos-cookbook/issues) for bugs and feature requests
- **Share Success Stories**: We love hearing how you use Cosmos models creatively

## License and Contact

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

NVIDIA Cosmos source code is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).

NVIDIA Cosmos models are released under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). For a custom license, please contact [cosmos-license@nvidia.com](mailto:cosmos-license@nvidia.com).
