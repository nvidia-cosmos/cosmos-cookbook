# Cosmos Cookbook

[![Documentation](https://img.shields.io/badge/docs-cosmos--cookbook-blue)](https://nvidia-cosmos.github.io/cosmos-cookbook/)
[![Contributing](https://img.shields.io/badge/contributing-guide-green)](CONTRIBUTING.md)

A comprehensive guide for working with the **NVIDIA Cosmos ecosystem**—a suite of World Foundation Models (WFMs) for real-world, domain-specific applications across robotics, simulation, autonomous systems, and physical scene understanding.

**📚 [View the Full Documentation →](https://nvidia-cosmos.github.io/cosmos-cookbook/)** — Step-by-step workflows, case studies, and technical recipes

<https://github.com/user-attachments/assets/bb444b93-d6af-4e25-8bd0-ca5891b26276>

## Latest Updates

| **Date** | **Recipe** | **Model** | **Description** |
|----------|------------|-----------|-----------------|
| Jan 5 | [Post-train Cosmos Reason 2 for AV Video Captioning and VQA](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/reason2/video_caption_vqa/post_training.html) | Cosmos Reason 2 | Domain adaptation for autonomous vehicle video captioning with multi-benchmark evaluation |
| Jan 1 | [Egocentric Social Reasoning for Robotics](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/inference/reason2/intbot_showcase/inference.html) | Cosmos Reason 2 | Egocentric social and physical reasoning evaluation for social robotics |
| Jan 1 | [Reason 2 on Brev](https://nvidia-cosmos.github.io/cosmos-cookbook/getting_started/brev/reason2/reason2_on_brev.html) | Cosmos Reason 2 | Getting started guide for Cosmos Reason 2 inference and post-training on Brev |
| Dec 22 | [Multiview AV Generation with World Scenario Maps](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/transfer2_5/av_world_scenario_maps/post_training.html) | Cosmos Transfer 2.5 | ControlNet post-training for spatially-conditioned multiview AV video generation |
| Dec 20 | [Vision AI Gallery](https://nvidia-cosmos.github.io/cosmos-cookbook/gallery/vision_ai_inference.html) | Cosmos Transfer 2.5 | Interactive gallery showcasing weather, lighting, and object augmentations for traffic scenarios |
| Dec 20 | [Style-Guided Video Generation](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/inference/transfer2_5/inference-image-prompt/inference.html) | Cosmos Transfer 2.5 | Generate videos with style guidance from reference images using edge/depth/segmentation control |

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
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Install just (command runner)
uv tool install -U rust-just
```

For other platforms, see **[astral.sh/uv](https://astral.sh/uv/)** for installation instructions.

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

When contributing media files, prefer `.mp4` over `.gif`:

- **Better quality** — MP4 supports full color depth vs GIF's 256-color limit
- **Smaller file size** — Modern video codecs compress far more efficiently
- **Audio support** — MP4 can include narration when needed

Use **H.264** encoding for universal browser compatibility.

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
