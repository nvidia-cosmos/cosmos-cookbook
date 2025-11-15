# Cosmos Cookbook

[![Documentation](https://img.shields.io/badge/docs-cosmos--cookbook-blue)](https://nvidia-cosmos.github.io/cosmos-cookbook/)
[![Contributing](https://img.shields.io/badge/contributing-guide-green)](CONTRIBUTING.md)

A comprehensive guide for working with the **NVIDIA Cosmos ecosystem**—a suite of World Foundation Models (WFMs) for real-world, domain-specific applications across robotics, simulation, autonomous systems, and physical scene understanding.

**📚 [View the Full Documentation](https://nvidia-cosmos.github.io/cosmos-cookbook/)**

> The complete Cosmos Cookbook with step-by-step workflows, technical recipes, and comprehensive guides is available at:
> **<https://nvidia-cosmos.github.io/cosmos-cookbook/>**

https://github.com/user-attachments/assets/bb444b93-d6af-4e25-8bd0-ca5891b26276

## Overview

This cookbook provides step-by-step workflows, technical recipes, and comprehensive guides for the complete AI development lifecycle with Cosmos models:

- **Inference**: Quick-start examples with pre-trained models
- **Gallery**: Visual showcase of Cosmos results across different domains
- **Data Curation**: Scalable data processing pipelines with Cosmos Curator
- **Post-Training**: Custom fine-tuning for domain-specific adaptation
- **Evaluation**: Quality control and model assessment workflows

The Cosmos ecosystem includes core model families: **Curator**, **Predict** (versions 2 and 2.5), **Transfer** (versions 1 and 2.5), **Reason 1**, and **RL**, each targeting specific capabilities in the AI development workflow.

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

## Quick Start

### 1. Install System Dependencies

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Install just (command runner)
uv tool install -U rust-just

# Optional useful tools
uv tool install -U s5cmd      # High-performance S3 operations
uv tool install -U streamlit  # Web app framework
uv tool install -U yt-dlp     # Video downloading
```

### 2. Clone and Setup Repository

```bash
# Clone the repository
git clone https://github.com/nvidia-cosmos/cosmos-cookbook.git
cd cosmos-cookbook

# Install dependencies and setup
just install
```

### 3. Explore the Documentation

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

## Key Features

- **End-to-End Examples**: Complete workflows from data to deployment
- **Quick Start Templates**: Get up and running in minutes
- **Modular Scripts**: Reusable components for custom workflows
- **Evaluation Tools**: Built-in quality assessment and metrics
- **Production Ready**: Scalable pipelines for real-world deployment
- **Comprehensive Docs**: Step-by-step workflows and technical recipes

## Documentation

- **[Full Documentation](https://nvidia-cosmos.github.io/cosmos-cookbook/)** - Complete guides and examples
- **[Getting Started](https://nvidia-cosmos.github.io/cosmos-cookbook/get_started.html)** - Environment setup and first steps
- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute to the cookbook

## Community & Support

- **Share Success Stories**: We love hearing how you use Cosmos models creatively
- **Report Issues**: Use GitHub issues for bugs and feature requests
- **Discussions**: Join our community discussions
- **Documentation**: Check our comprehensive guides first

## License and Contact

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

NVIDIA Cosmos source code is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).

NVIDIA Cosmos models are released under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). For a custom license, please contact [cosmos-license@nvidia.com](mailto:cosmos-license@nvidia.com).
