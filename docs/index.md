# Cosmos Cookbook

## Overview

**[NVIDIA Cosmos™](https://www.nvidia.com/en-us/ai/cosmos/)** is a platform of state-of-the-art generative world foundation models (WFMs), guardrails, and an accelerated data processing and curation pipeline. This cookbook serves as a practical guide to the Cosmos open models — offering step-by-step workflows, technical recipes, and concrete examples for building, adapting, and deploying WFMs. It helps developers reproduce successful Cosmos model deployments and customize them for their specific domains.

The Cosmos ecosystem supports the complete Physical AI development lifecycle — from inference using pre-trained models to custom post-training for domain adaptation. Inside, you'll find:

- Quick-start inference examples to get up and running fast.
- Advanced post-training workflows for domain-specific fine-tuning.
- Proven recipes for scalable, production-ready deployments.

## Open Source Community Platform

The Cosmos Cookbook is an open-source resource where NVIDIA and the broader Physical AI community share practical workflows, proven techniques, and domain-specific adaptations.

**📂 Repository:** [https://github.com/nvidia-cosmos/cosmos-cookbook](https://github.com/nvidia-cosmos/cosmos-cookbook)

We welcome contributions—from new examples and workflow improvements to bug fixes and documentation updates. Together, we can evolve best practices and accelerate the adoption of Cosmos models across domains.

## Case Study Recipes

The cookbook includes comprehensive use cases demonstrating real-world applications across the Cosmos platform.

### **Cosmos Predict**

#### Future state prediction and generation

| **Workflow** | **Description** | **Link** |
|--------------|-----------------|----------|
| **Inference** | Text2Image synthetic data generation for intelligent transportation systems | [ITS Synthetic Data Generation](recipes/inference/predict2/inference-its/inference.md) |
| **Training** | Traffic anomaly generation with improved realism and prompt alignment | [Traffic Anomaly Generation](recipes/post_training/predict2/its-accident/post_training.md) |
| **Training** | Synthetic trajectory data generation for humanoid robot learning | [GR00T-Dreams](recipes/post_training/predict2/gr00t-dreams/post-training.md) |

### **Cosmos Transfer**

#### Multi-control video generation and augmentation

| **Workflow** | **Description** | **Link** |
|--------------|-----------------|----------|
| **Inference** | CARLA simulator-to-real augmentation for traffic anomaly scenarios | [CARLA Sim2Real](recipes/inference/transfer2_5/inference-carla-sdg-augmentation/inference.md) |
| **Inference** | Multi-control video editing for background replacement, lighting, and object transformation | [Real-World Video Manipulation](recipes/inference/transfer2_5/inference-real-augmentation/inference.md) |
| **Inference** | Weather augmentation pipeline for simulation data using multi-modal controls | [Weather Augmentation](recipes/inference/transfer1/inference-its-weather-augmentation/inference.md) |
| **Inference** | CG-to-real conversion for multi-view warehouse environments | [Warehouse Simulation](recipes/inference/transfer1/inference-warehouse-mv/inference.md) |
| **Inference** | Sim2Real data augmentation for robotics navigation tasks | [X-Mobility Navigation](recipes/inference/transfer1/inference-x-mobility/inference.md) |
| **Inference** | Synthetic manipulation motion generation for humanoid robots | [GR00T-Mimic](recipes/inference/transfer1/gr00t-mimic/inference.md) |

### **Cosmos Reason**

#### Vision-language reasoning and quality control

| **Workflow** | **Description** | **Link** |
|--------------|-----------------|----------|
| **Training** | Physical plausibility check for video quality assessment | [Video Rewards](recipes/post_training/reason1/physical-plausibility-check/post_training.md) |
| **Training** | Spatial AI understanding for warehouse environments | [Spatial AI Warehouse](recipes/post_training/reason1/spatial-ai-warehouse/post_training.md) |
| **Training** | Intelligent transportation scene understanding and analysis | [Intelligent Transportation](recipes/post_training/reason1/intelligent-transportation/post_training.md) |
| **Training** | AV video captioning and visual question answering for autonomous vehicles | [AV Video Caption VQA](recipes/post_training/reason1/av_video_caption_vqa/post_training.md) |

### **Cosmos Curator**

| **Workflow** | **Description** | **Link** |
|--------------|-----------------|----------|
| **Curation** | Curate video data for Cosmos Predict 2 post-training | [Predict 2 Data Curation](recipes/data_curation/predict2_data/data_curation.md) |

## Cosmos Models for Physical AI

The Cosmos family of open models consists of five core repositories, each targeting specific capabilities in the AI development workflow:

**[Cosmos Curator](https://github.com/nvidia-cosmos/cosmos-curate)** - A GPU-accelerated video curation pipeline built on Ray. Supports multi-model analysis, content filtering, annotation, and deduplication for both inference and training data preparation.

**[Cosmos Predict](https://github.com/nvidia-cosmos/cosmos-predict2.5)** - A diffusion transformer for future state prediction. Provides text-to-image and video-to-world generation capabilities, with specialized variants for robotics and simulation. Supports custom training for domain-specific prediction tasks.

**[Cosmos Transfer](https://github.com/nvidia-cosmos/cosmos-transfer2.5)** - A multi-control video generation system with ControlNet and MultiControlNet conditioning (including depth, segmentation, LiDAR, and HDMap). Includes 4K upscaling capabilities and supports training for custom control modalities and domain adaptation.

**[Cosmos Reason](https://github.com/nvidia-cosmos/cosmos-reason1)** - A 7B vision-language model for physically grounded reasoning. Handles spatial/temporal understanding and chain-of-thought tasks, with fine-tuning support for embodied AI applications and domain-specific reasoning.

**[Cosmos RL](https://github.com/nvidia-cosmos/cosmos-rl)** - A distributed training framework supporting both supervised fine-tuning (SFT) and reinforcement learning approaches. Features elastic policy rollout, FP8/FP4 precision support, and optimization for large-scale VLM and LLM training.

All models include pre-trained checkpoints and support custom training for domain-specific adaptation. The diagram below illustrates component interactions across inference and training workflows.

![Cosmos Overview](assets/images/cosmos_overview.png)

## ML/Gen AI Concepts

The cookbook is organized around key concepts spanning (controlled) **inference** and **training** use cases:

**1. [Control Modalities](core_concepts/control_modalities/overview.md)** - Master precise control over video generation with Cosmos Transfer 2.5 using Edge, Depth, Segmentation, and Vis modalities. This covers structural preservation, semantic replacement, lighting consistency, and multi-control approaches for achieving high-fidelity, controllable video transformations.

**2. [Data Curation](core_concepts/data_curation/overview.md)** - Use Cosmos Curator to prepare your datasets with modular, scalable processing pipelines. This includes splitting, captioning, filtering, deduplication, task-specific sampling, and cloud-native or local execution.

**3. [Model Post-Training](core_concepts/post_training/overview.md)** - Fine-tune foundation models using your curated data. This covers domain adaptation for Predict (2 and 2.5), Transfer (1 and 2.5), and Reason 1, setup for supervised fine-tuning, LoRA, or reinforcement learning, and use of Cosmos RL for large-scale distributed rollout.

**4. [Evaluation and Quality Control](core_concepts/evaluation/overview.md)** - Ensure your post-trained models are aligned and robust through metrics, visualization, and qualitative inspection. Leverage Cosmos Reason 1 as a quality filter (e.g. for synthetic data rejection sampling).

**5. [Model Distillation](core_concepts/distillation/overview.md)** - Compress large foundation models into smaller, efficient variants while preserving performance. This includes knowledge distillation techniques for Cosmos models, teacher-student training setups, and deployment optimization for edge devices and resource-constrained environments.

## Gallery

Visual examples of Cosmos Transfer results across Physical AI domains:

- **[Robotics Domain Adaptation](gallery/robotics_inference.md)** - Sim-to-real transfer for robotic manipulation with varied materials, lighting, and environments.
- **[Autonomous Vehicle Domain Adaptation](gallery/av_inference.md)** - Multi-control video generation for driving scenes across different weather, lighting, and time-of-day conditions.

## Quick Start Paths

This cookbook provides flexible entry points for both **inference** and **training** workflows. Each section contains runnable scripts, technical recipes, and complete examples.

- **Inference workflows:** [Getting Started](get_started.md) for setup and immediate model deployment
- **Data processing:** [Data Processing & Analysis](core_concepts/data_curation/overview.md) for content analysis workflows
- **Training workflows:** [Model Training & Fine-tuning](core_concepts/post_training/overview.md) for domain adaptation
- **Case study recipes:** [Case Study Recipes](#case-study-recipes) organized by application area
