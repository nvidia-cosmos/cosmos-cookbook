# Cosmos Cookbook

[![Documentation](https://img.shields.io/badge/docs-cosmos--cookbook-blue)](https://nvidia-cosmos.github.io/cosmos-cookbook/)
[![Contributing](https://img.shields.io/badge/contributing-guide-green)](CONTRIBUTING.md)

A comprehensive guide for working with the **NVIDIA Cosmos ecosystem**—a suite of World Foundation Models (WFMs) for real-world, domain-specific applications across robotics, simulation, autonomous systems, and physical scene understanding.

**📚 [View the Full Documentation →](https://nvidia-cosmos.github.io/cosmos-cookbook/)** — Step-by-step workflows, case studies, and technical recipes

<https://github.com/user-attachments/assets/bb444b93-d6af-4e25-8bd0-ca5891b26276>

## Latest Updates

| **Date** | **Recipe** | **Model** | **Description** |
|----------|------------|-----------|-----------------|
| Jan 30 | [Prompt Guide](https://nvidia-cosmos.github.io/cosmos-cookbook/core_concepts/prompt_guide/reason_guide.html) | Cosmos Reason 2 | Inference Prompt Guide |
| Jan 29 | [Video Search and Summarization with Cosmos Reason](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/inference/reason2/vss/inference.html) | Cosmos Reason 2 | GPU-accelerated video analysis pipeline for large-scale video summarization, Q&A, and live stream alerts across warehouses, factories, retail, and smart cities |
| Jan 28 | [Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/predict2/cosmos_policy/post_training.html) | Cosmos Predict 2 | State-of-the-art robot policy through latent frame injection for visuomotor control achieving 98.5% on LIBERO, 67.1% on RoboCasa, and 93.6% on ALOHA |
| Jan 27 | [Physical Plausibility Prediction with Cosmos Reason 2](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/reason2/physical-plausibility-check/post_training.html) | Cosmos Reason 2 | Supervised fine-tuning for physical plausibility prediction using VideoPhy-2 dataset with improved zero-shot and SFT performance |
| Jan 26 | [Intelligent Transportation Post-Training with Cosmos Reason 2](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/reason2/intelligent-transportation/post_training.html) | Cosmos Reason 2 | Post-training Cosmos Reason 2 for intelligent transportation scene understanding using WovenTraffic Safety dataset |
| Jan 6 | [Dataset Video Clustering with Time Series K-Means](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/data_curation/embedding_analysis/embedding_analysis.html) | Cosmos Curator | Advanced video clustering using Time Series K-Means on embedding vector trajectories for robotics behavior analysis |

## Upcoming Activities

### NVIDIA GTC 2026

Register for [NVIDIA GTC](https://www.nvidia.com/gtc/) happening **March 16–19, 2026**, and add the [Cosmos sessions](https://www.nvidia.com/gtc/session-catalog/?sessions=S81667,CWES81669,DLIT81644,DLIT81698,S81836,S81488,S81834,DLIT81774,CWES81733,CWES81568) to your calendar. Don't miss the must-see keynote from CEO Jensen Huang at SAP Center on Monday, March 16 at 11:00 a.m. PT.

### NVIDIA Cosmos Cookoff

Introducing the **[NVIDIA Cosmos Cookoff](https://luma.com/nvidia-cosmos-cookoff)** — a virtual, four-week physical AI challenge running **January 29 – February 26** for robotics, AV, and vision AI builders.

Build with NVIDIA Cosmos Reason and Cosmos Cookbook recipes—from egocentric robot reasoning to physical plausibility checks and traffic-aware models for a chance to win **$5,000**, an **NVIDIA DGX Spark**, and more!

**[Register Now →](https://luma.com/nvidia-cosmos-cookoff)**

Sponsored by Nebius and Milestone.

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
