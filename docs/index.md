# Cosmos Playbook

## Overview

This playbook documents the **NVIDIA Cosmos ecosystem**—a suite of World Foundation Models (WFMs) for real-world, domain-specific applications. The playbook provides step-by-step workflows, technical recipes, and concrete examples across robotics, simulation, autonomous systems, and physical scene understanding.

The Cosmos ecosystem covers the complete AI development lifecycle: from **inference** with pre-trained models to **custom post-training** for domain-specific adaptation. The playbook includes quick-start inference examples, advanced post-training workflows, and proven recipes for successful model deployment and customization.

## Cosmos Model Ecosystem

The Cosmos architecture consists of five core repositories, each targeting specific capabilities in the AI development workflow:

**[Cosmos Curate](https://github.com/nvidia-cosmos/cosmos-curate)** - A GPU-accelerated video curation pipeline built on Ray. Supports multi-model analysis, content filtering, annotation, and deduplication for both inference and training data preparation.

**[Cosmos Predict2](https://github.com/nvidia-cosmos/cosmos-predict2)** - A diffusion transformer for future state prediction. Provides text-to-image and video-to-world generation capabilities, with specialized variants for robotics and simulation. Supports custom training for domain-specific prediction tasks.

**[Cosmos Transfer1](https://github.com/nvidia-cosmos/cosmos-transfer1)** - A multi-control video generation system with ControlNet and MultiControlNet conditioning (depth, segmentation, LiDAR, HDMap). Includes 4K upscaling capabilities and supports training for custom control modalities and domain adaptation.

**[Cosmos Reason1](https://github.com/nvidia-cosmos/cosmos-reason1)** - A 7B vision-language model for physically grounded reasoning. Handles spatial/temporal understanding and chain-of-thought tasks, with fine-tuning support for embodied AI applications and domain-specific reasoning.

**[Cosmos RL](https://github.com/nvidia-cosmos/cosmos-rl)** - A distributed training framework supporting both supervised fine-tuning (SFT) and reinforcement learning approaches. Features elastic policy rollout, FP8/FP4 precision support, and optimization for large-scale VLM and LLM training.

All models include pre-trained checkpoints and support custom training for domain-specific adaptation. The diagram below illustrates component interactions across inference and training workflows.

![Cosmos Overview](assets/images/cosmos_overview.png)

## Cosmos Workflows

The playbook is organized around key workflows spanning **inference** and **training** use cases:

**1. [Data Curation](data_curation/overview.md)** - Use Cosmos Curate to prepare your datasets with modular, scalable processing pipelines. This includes splitting, captioning, filtering, deduplication, task-specific sampling, and cloud-native or local execution.

**2. [Model Post-Training](post_training/overview.md)** - Fine-tune foundation models using your curated data. This covers domain adaptation for Predict2, Transfer1, Reason1, setup for supervised fine-tuning, LoRA, or reinforcement learning, and use of Cosmos RL for large-scale distributed rollouts.

**3. [Evaluation & Quality Control](evaluation/overview.md)** - Ensure your post-trained models are aligned and robust through metrics, visualization, and qualitative inspection. Leverage Cosmos Reason1 as a quality filter (e.g., for synthetic data rejection sampling).

## End-to-End Examples

The playbook includes comprehensive case studies demonstrating real-world applications across the Cosmos ecosystem:

---

### 🎯 **Cosmos Predict2**

#### Future state prediction and generation

| **Workflow** | **Description** | **Link** |
|--------------|-----------------|----------|
| **Training** | Traffic anomaly generation with improved realism and prompt alignment | [Traffic Anomaly Generation](examples/predict2/cosmos-predict2-its-accident/SUMMARY.md) |

---

### 🎬 **Cosmos Transfer1**

#### Multi-control video generation and augmentation

| **Workflow** | **Description** | **Link** |
|--------------|-----------------|----------|
| **Inference** | Weather augmentation pipeline for simulation data using multi-modal controls | [Weather Augmentation](examples/transfer1/inference-its-weather-augmentation/SUMMARY.md) |
| **Inference** | CG-to-real conversion for multi-view warehouse environments | [Warehouse Simulation](examples/transfer1/inference-warehouse-mv/SUMMARY.md) |

---

### 🧠 **Cosmos Reason1**

#### Vision-language reasoning and quality control

| **Workflow** | **Description** | **Link** |
|--------------|-----------------|----------|
| **Training** | Spatial, temporal, and causal reasoning for autonomous vehicles | [AV Visual Q&A](examples/reason1/cosmos-reason1-av-vqa/SUMMARY.md) |
| **Training** | Video critique model for realism and coherence assessment | [Video Critique](examples/reason1/reason1-video-critique/SUMMARY.md) |
| **Inference** | Quality filtering for synthetic data generation workflows | [Sampling Rejection](examples/transfer1/transfer1-reason1-sampling/SUMMARY.md) |

## Getting Started

This playbook provides flexible entry points for both **inference** and **training** workflows. Each section contains runnable scripts, technical recipes, and complete examples.

### **🚀 Quick Start Paths**

- **Inference workflows:** [Getting Started](getting_started.md) for setup and immediate model deployment
- **Data processing:** [Data Processing & Analysis](data_curation/overview.md) for content analysis workflows
- **Training workflows:** [Model Training & Fine-tuning](post_training/overview.md) for domain adaptation
- **Domain examples:** [End-to-End Examples](#end-to-end-examples) organized by application area

### **📚 Complete Workflows**

- **Inference-focused:** [Getting Started](getting_started.md) → [End-to-End Examples](#end-to-end-examples)
- **Training-focused:** [Data Curation](data_curation/overview.md) → [Model Training](post_training/overview.md) → [Evaluation](evaluation/overview.md)
- **End-to-end examples:** Follow documented examples from the showcase for complete workflows

The playbook serves as a technical reference for reproducing successful Cosmos model deployments across different domains.
