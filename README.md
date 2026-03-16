# Cosmos Cookbook

[![Documentation](https://img.shields.io/badge/docs-cosmos--cookbook-blue)](https://nvidia-cosmos.github.io/cosmos-cookbook/)
[![Contributing](https://img.shields.io/badge/contributing-guide-green)](CONTRIBUTING.md)

A comprehensive guide for working with the **NVIDIA Cosmos ecosystem**—a suite of World Foundation Models (WFMs) for real-world, domain-specific applications across robotics, simulation, autonomous systems, and physical scene understanding.

**📚 [View the Full Documentation →](https://nvidia-cosmos.github.io/cosmos-cookbook/)** — Step-by-step workflows, case studies, and technical recipes

<https://github.com/user-attachments/assets/bb444b93-d6af-4e25-8bd0-ca5891b26276>

## Latest Updates

| **Date** | **Recipe** | **Model** | **Description** |
|----------|------------|-----------|-----------------|
| Mar 16 | [Cosmos-Reason2 on Jetson Thor for Edge VLM Perception](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/inference/reason2/intbot_edge_vlm/inference.html) | Cosmos Reason 2 | Deploy Cosmos-Reason2 on Jetson AGX Thor for social robots (IntBot), with FP8 quantization and TensorRT-Edge-LLM optimization |
| Mar 15 | [Content-Adaptive Video Compression for Cosmos Curator with Beamr CABR](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/data_curation/cosmos_cabr/cabr_recipe.html) | Cosmos Curator | Recipe for replacing Cosmos Curator's default CPU-based, fixed-bitrate video encoder with Beamr CABR (Content-Adaptive Bitrate) — a GPU-accelerated video optimization and encoding solution |
| Mar 15 | [Post-Training Cosmos-H-Surgical-Simulator for Surgical Robotics](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/predict2_5/surgical_robotics/post_training.html) | Cosmos Predict 2.5 | Fine-tune Cosmos Predict 2.5 as an action-conditioned surgical simulator for policy evaluation and synthetic data generation using the SutureBot dataset |
| Mar 15 | [Outlier Detection in Embedding Vector Trajectories](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/data_curation/outlier_detection/outlier_detection.html) | Cosmos Curator | Outlier detection in video embedding trajectories via Time Series K-Means + Soft-DTW distance thresholding |
| Mar 3 | [GR00T-Dreams: Synthetic Trajectory Generation for Robot Learning](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/end2end/gr00t-dreams/post-training.html) | Cosmos Predict 2.5, Reason 2 | End-to-end pipeline for synthetic robot trajectory generation: post-train Predict 2.5 on GR1 data, generate trajectories, and use Cosmos Reason 2 as video critic for rejection sampling |
| Feb 18 | [Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/predict2/cosmos_policy/post_training.html) | Cosmos Predict 2.5 | Recipe upgraded to **Cosmos Predict 2.5**: state-of-the-art robot policy via latent frame injection. Results—LIBERO 98.33%, RoboCasa **71.1%** (new SOTA, +4% over Predict2) |
| Feb 18 | [3D AV Grounding Post-Training with Cosmos Reason 1 & 2](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/reason2/av_3d_grounding/post_training.html) | Cosmos Reason 1 & 2 | 3D vehicle grounding in autonomous driving: detect and localize vehicles in 3D from camera images with SFT (Cosmos-RL and Qwen-Finetune) |
| Feb 4 | [Worker Safety in a Classical Warehouse](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/inference/reason2/worker_safety/inference.html) | Cosmos Reason 2 | Zero-shot industrial safety compliance and hazard detection in classical warehouse environments using context-aware prompt engineering |
| Jan 30 | [Prompt Guide](https://nvidia-cosmos.github.io/cosmos-cookbook/core_concepts/prompt_guide/reason_guide.html) | Cosmos Reason 2 | Inference Prompt Guide |

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

→ Or **[Deploy on Cloud](https://nvidia-cosmos.github.io/cosmos-cookbook/getting_started/cloud_platform.html)** (Nebius, Brev, and more to come) for ready-to-launch GPU instances.

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
