# Cosmos Cookbook

## Overview

The **NVIDIA Cosmos ecosystem** is a suite of World Foundation Models (WFMs) for real-world, domain-specific applications. This cookbook provides step-by-step workflows, technical recipes, and concrete examples across robotics, simulation, autonomous systems, and physical scene understanding. It serves as a technical reference for reproducing successful Cosmos model deployments across different domains.

The Cosmos ecosystem covers the complete AI development lifecycle: from **inference** with pre-trained models to **custom post-training** for domain-specific adaptation. The cookbook includes quick-start inference examples, advanced post-training workflows, and proven recipes for successful model deployment and customization.

<video width="100%" height="60%" controls autoplay loop muted>
  <source src="assets/images/recipe_overview.gif" type="image/gif">
  Your browser does not support the video tag.
</video>

## Open Source Community Platform

The Cosmos Cookbook is designed as an **open-source platform** where NVIDIA shares practical knowledge and proven techniques with the broader AI community. This collaborative approach enables researchers, developers, and practitioners to contribute their own workflows, improvements, and domain-specific adaptations.

**Repository:** [https://github.com/nvidia-cosmos/cosmos-cookbook](https://github.com/nvidia-cosmos/cosmos-cookbook)

We encourage community contributions including new examples, workflow improvements, bug fixes, and documentation enhancements. The open-source nature ensures that the collective knowledge and best practices around Cosmos models continue to evolve and benefit the entire ecosystem.

## End-to-End Examples

The cookbook includes comprehensive case studies demonstrating real-world applications across the Cosmos ecosystem.

### **Cosmos Predict**

#### Future state prediction and generation

| **Workflow** | **Description** | **Link** |
|--------------|-----------------|----------|
| **Training** | Traffic anomaly generation with improved realism and prompt alignment | [Traffic Anomaly Generation](examples/predict2/its-accident/post_training.md) |

### **Cosmos Transfer**

#### Multi-control video generation and augmentation

| **Workflow** | **Description** | **Link** |
|--------------|-----------------|----------|
| **Inference** | Weather augmentation pipeline for simulation data using multi-modal controls | [Weather Augmentation](examples/transfer1/inference-its-weather-augmentation/inference.md) |
| **Inference** | CG-to-real conversion for multi-view warehouse environments | [Warehouse Simulation](examples/transfer1/inference-warehouse-mv/inference.md) |
| **Inference** | CARLA simulator-to-real augmentation for traffic anomaly scenarios | [CARLA Sim2Real](examples/transfer2_5/inference-carla-sdg-augmentation/inference.md) |

### **Cosmos Reason**

#### Vision-language reasoning and quality control

| **Workflow** | **Description** | **Link** |
|--------------|-----------------|----------|
| **Training** | Physical plausibility check for video quality assessment | [Video Rewards](examples/reason1/physical-plausibility-check/post_training.md) |
| **Training** | Spatial AI understanding for warehouse environments | [Spatial AI Warehouse](examples/reason1/spatial-ai-warehouse/post_training.md) |
| **Training** | Intelligent transportation scene understanding and analysis | [Intelligent Transportation](examples/reason1/intelligent-transportation/post_training.md) |
<!--| **Training** | Visual question answering for autonomous vehicle scenarios | [Visual QA for AV](examples/reason1/visual-qa-for-AV/overview.md) |-->

## Cosmos Model Ecosystem

The Cosmos architecture consists of five core repositories, each targeting specific capabilities in the AI development workflow:

**[Cosmos Curate](https://github.com/nvidia-cosmos/cosmos-curate)** - A GPU-accelerated video curation pipeline built on Ray. Supports multi-model analysis, content filtering, annotation, and deduplication for both inference and training data preparation.

**[Cosmos Predict2](https://github.com/nvidia-cosmos/cosmos-predict2)** - A diffusion transformer for future state prediction. Provides text-to-image and video-to-world generation capabilities, with specialized variants for robotics and simulation. Supports custom training for domain-specific prediction tasks.

**[Cosmos Transfer1](https://github.com/nvidia-cosmos/cosmos-transfer1)** - A multi-control video generation system with ControlNet and MultiControlNet conditioning (including depth, segmentation, LiDAR, and HDMap). Includes 4K upscaling capabilities and supports training for custom control modalities and domain adaptation.

**[Cosmos Reason1](https://github.com/nvidia-cosmos/cosmos-reason1)** - A 7B vision-language model for physically grounded reasoning. Handles spatial/temporal understanding and chain-of-thought tasks, with fine-tuning support for embodied AI applications and domain-specific reasoning.

**[Cosmos RL](https://github.com/nvidia-cosmos/cosmos-rl)** - A distributed training framework supporting both supervised fine-tuning (SFT) and reinforcement learning approaches. Features elastic policy rollout, FP8/FP4 precision support, and optimization for large-scale VLM and LLM training.

All models include pre-trained checkpoints and support custom training for domain-specific adaptation. The diagram below illustrates component interactions across inference and training workflows.

![Cosmos Overview](assets/images/cosmos_overview.png)

## Cosmos Workflows

The cookbook is organized around key workflows spanning **inference** and **training** use cases:

**1. [Data Curation](core_concepts/data_curation/overview.md)** - Use Cosmos Curate to prepare your datasets with modular, scalable processing pipelines. This includes splitting, captioning, filtering, deduplication, task-specific sampling, and cloud-native or local execution.

**2. [Model Post-Training](core_concepts/post_training/overview.md)** - Fine-tune foundation models using your curated data. This covers domain adaptation for Predict2, Transfer1, and Reason1, setup for supervised fine-tuning, LoRA, or reinforcement learning, and use of Cosmos RL for large-scale distributed rollout.

**3. [Evaluation and Quality Control](core_concepts/evaluation/overview.md)** - Ensure your post-trained models are aligned and robust through metrics, visualization, and qualitative inspection. Leverage Cosmos Reason1 as a quality filter (e.g. for synthetic data rejection sampling).

**4. [Model Distillation](core_concepts/distillation/overview.md)** - Compress large foundation models into smaller, efficient variants while preserving performance. This includes knowledge distillation techniques for Cosmos models, teacher-student training setups, and deployment optimization for edge devices and resource-constrained environments.

## Quick Start Paths

This cookbook provides flexible entry points for both **inference** and **training** workflows. Each section contains runnable scripts, technical recipes, and complete examples.

### **Quick Start Paths**

- **Inference workflows:** [Getting Started](get_started.md) for setup and immediate model deployment
- **Data processing:** [Data Processing & Analysis](core_concepts/data_curation/overview.md) for content analysis workflows
- **Training workflows:** [Model Training & Fine-tuning](core_concepts/post_training/overview.md) for domain adaptation
- **Domain examples:** [End-to-End Examples](#end-to-end-examples) organized by application area
