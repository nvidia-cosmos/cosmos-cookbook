# Cosmos Cookbook

<div style="width: 100%; max-width: 969px; margin: 2rem 0; display: block;">
  <video autoplay loop muted playsinline style="width: 100%; max-width: 969px; height: auto; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); display: block;">
    <source src="assets/images/homepage_video.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>

## Overview

**[NVIDIA Cosmos™](https://www.nvidia.com/en-us/ai/cosmos/)** is a platform of state-of-the-art generative world foundation models (WFMs), guardrails, and an accelerated data processing and curation pipeline. This cookbook serves as a practical guide to the Cosmos open models--offering step-by-step workflows, technical recipes, and concrete examples for building, adapting, and deploying WFMs. It helps developers reproduce successful Cosmos model deployments and customize them for their specific domains.

The Cosmos ecosystem supports the complete Physical AI development lifecycle--from inference using pre-trained models to custom post-training for domain adaptation. Inside, you'll find the following:

- Quick-start inference examples to get up and running fast.
- Advanced post-training workflows for domain-specific fine-tuning.
- Proven recipes for scalable, production-ready deployments.

## Latest Updates

| **Date** | **Recipe** | **Model** |
|----------|------------|-----------|
| Feb 4 | [Worker Safety in a Classical Warehouse](recipes/inference/reason2/worker_safety/inference.md) | Cosmos Reason 2 |
| Jan 30 | [Prompt Guide](core_concepts/prompt_guide/reason_guide.md) | Cosmos Reason 2 |
| Jan 29 | [Video Search and Summarization with Cosmos Reason](recipes/inference/reason2/vss/inference.md) | Cosmos Reason 2 |
| Jan 28 | [Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning](recipes/post_training/predict2/cosmos_policy/post_training.md) | Cosmos Predict 2 |
| Jan 27 | [Physical Plausibility Prediction with Cosmos Reason 2](recipes/post_training/reason2/physical-plausibility-check/post_training.md) | Cosmos Reason 2 |
| Jan 26 | [Intelligent Transportation Post-Training with Cosmos Reason 2](recipes/post_training/reason2/intelligent-transportation/post_training.md) | Cosmos Reason 2 |
| Jan 6 | [Dataset Video Clustering with Time Series K-Means](recipes/data_curation/embedding_analysis/embedding_analysis.md) | Cosmos Curator |

## Upcoming Activities

### NVIDIA GTC 2026

Register for [NVIDIA GTC](https://www.nvidia.com/gtc/) happening **March 16–19, 2026**, and add the [Cosmos sessions](https://www.nvidia.com/gtc/session-catalog/?sessions=S81667,CWES81669,DLIT81644,DLIT81698,S81836,S81488,S81834,DLIT81774,CWES81733,CWES81568) to your calendar. Don't miss the must-see keynote from CEO Jensen Huang at SAP Center on Monday, March 16 at 11:00 a.m. PT.

### NVIDIA Cosmos Cookoff

Introducing the **[NVIDIA Cosmos Cookoff](https://luma.com/nvidia-cosmos-cookoff)** — a virtual, four-week physical AI challenge running **January 29 – February 26** for robotics, AV, and vision AI builders.

Build with NVIDIA Cosmos Reason and Cosmos Cookbook recipes—from egocentric robot reasoning to physical plausibility checks and traffic-aware models for a chance to win **$5,000**, an **NVIDIA DGX Spark**, and more!

**[Register Now →](https://luma.com/nvidia-cosmos-cookoff)**

Sponsored by Nebius and Milestone.

## Open Source Community Platform

The Cosmos Cookbook is an open-source resource where NVIDIA and the broader Physical AI community share practical workflows, proven techniques, and domain-specific adaptations.

**📂 Repository:** [https://github.com/nvidia-cosmos/cosmos-cookbook](https://github.com/nvidia-cosmos/cosmos-cookbook)

We welcome contributions—from new examples and workflow improvements to bug fixes and documentation updates. Together, we can evolve best practices and accelerate the adoption of Cosmos models across domains.

**📊 Physical AI Datasets:** Access curated datasets for autonomous vehicles, intelligent transportation systems, robotics, smart spaces, and warehouse environments on the [NVIDIA Physical AI Collection](https://huggingface.co/collections/nvidia/physical-ai) on Hugging Face.

## Case Study Recipes

The Cosmos Cookbook includes comprehensive use cases demonstrating real-world applications across the Cosmos platform.

### [**Cosmos Predict**](https://github.com/nvidia-cosmos/cosmos-predict2.5)

#### Future state prediction and generation

| **Workflow** | **Description** | **Link** |
|--------------|-----------------|----------|
| **Inference** | Text2Image synthetic data generation for intelligent transportation systems | [ITS Synthetic Data Generation](recipes/inference/predict2/inference-its/inference.md) |
| **Training** | Fine-tuning Cosmos Predict 2 for robotic manipulation through latent frame injection for visuomotor control | [Cosmos Policy](recipes/post_training/predict2/cosmos_policy/post_training.md) |
| **Training** | Traffic anomaly generation with improved realism and prompt alignment | [Traffic Anomaly Generation](recipes/post_training/predict2/its-accident/post_training.md) |
| **Training** | Synthetic trajectory data generation for humanoid robot learning | [GR00T-Dreams](recipes/post_training/predict2/gr00t-dreams/post-training.md) |
| **Training** | LoRA post-training for sports video generation with improved player dynamics and rule coherence | [Sports Video Generation](recipes/post_training/predict2_5/sports/post_training.md) |

> **Advanced Topics:** Refer to [Distilling Cosmos Predict 2.5](core_concepts/distillation/distilling_predict2.5.md) to learn how to distill the model into a 4-step student using DMD2.

### [**Cosmos Transfer**](https://github.com/nvidia-cosmos/cosmos-transfer2.5)

#### Multi-control video generation and augmentation

| **Workflow** | **Description** | **Link** |
|--------------|-----------------|----------|
| **Guide** | Master precise control over video generation using Edge, Depth, Segmentation, and Vis modalities for structural preservation and semantic replacement | [Control Modalities Guide](core_concepts/control_modalities/overview.md) |
| **Inference** | Style-guided video generation using image references with edge/depth/segmentation control | [Style-Guided Generation](recipes/inference/transfer2_5/inference-image-prompt/inference.md) |
| **Inference** | CARLA simulator-to-real augmentation for traffic anomaly scenarios | [CARLA Sim2Real](recipes/inference/transfer2_5/inference-carla-sdg-augmentation/inference.md) |
| **Inference** | Multi-control video editing for background replacement, lighting, and object transformation | [Real-World Video Manipulation](recipes/inference/transfer2_5/inference-real-augmentation/inference.md) |
| **Inference** | Domain transfer pipeline for scarce biological datasets using edge-based control and FiftyOne | [BioTrove Moths Augmentation](recipes/inference/transfer2_5/biotrove_augmentation/inference.md) |
| **Inference** | Weather augmentation pipeline for simulation data using multi-modal controls | [Weather Augmentation](recipes/inference/transfer1/inference-its-weather-augmentation/inference.md) |
| **Inference** | CG-to-real conversion for multi-view warehouse environments | [Warehouse Simulation](recipes/inference/transfer1/inference-warehouse-mv/inference.md) |
| **Inference** | Sim2Real data augmentation for robotics navigation tasks | [X-Mobility Navigation](recipes/inference/transfer1/inference-x-mobility/inference.md) |
| **Inference** | Synthetic manipulation motion generation for humanoid robots | [GR00T-Mimic](recipes/inference/transfer1/gr00t-mimic/inference.md) |
| **Training** | ControlNet post-training for spatially-conditioned multiview AV video generation with world scenario maps | [Multiview AV Generation](recipes/post_training/transfer2_5/av_world_scenario_maps/post_training.md) |

### [**Cosmos Reason**](https://github.com/nvidia-cosmos/cosmos-reason1)

#### Vision-language reasoning and quality control

| **Workflow** | **Description**                                                           | **Link**                                                                                                |
| ------------ | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| **Guide** | Comprehensive prompting guide for Cosmos Reason 2 covering message structure, sampling parameters, and domain-specific patterns | [Cosmos Reason 2 Prompt Guide](core_concepts/prompt_guide/reason_guide.md)                   |
| **Inference** | GPU-accelerated video analysis pipeline for large-scale video summarization, Q&A, and live stream alerts | [Video Search and Summarization](recipes/inference/reason2/vss/inference.md)                   |
| **Inference** | Zero-shot safety compliance and hazard detection in industrial warehouse environments | [Worker Safety in a Classical Warehouse](recipes/inference/reason2/worker_safety/inference.md) |
| **Inference** | Egocentric social and physical reasoning for social robotics             | [Egocentric Social Reasoning](recipes/inference/reason2/intbot_showcase/inference.md) |
| **Training** | Post-training Cosmos Reason 2 for AV video captioning and VQA with production data | [AV Video Caption VQA (Reason 2)](recipes/post_training/reason2/video_caption_vqa/post_training.md)     |
| **Training** | Post-training Cosmos Reason 2 for intelligent transportation scene understanding with WTS data | [Intelligent Transportation (Reason 2)](recipes/post_training/reason2/intelligent-transportation/post_training.md) |
| **Training** | Physical plausibility prediction for video quality assessment with Cosmos Reason 2 | [Physical Plausibility (Reason 2)](recipes/post_training/reason2/physical-plausibility-check/post_training.md) |
| **Training** | Physical plausibility check for video quality assessment                  | [Physical Plausibility (Reason 1)](recipes/post_training/reason1/physical-plausibility-check/post_training.md)             |
| **Training** | Spatial AI understanding for warehouse environments                       | [Spatial AI Warehouse](recipes/post_training/reason1/spatial-ai-warehouse/post_training.md)             |
| **Training** | Intelligent transportation scene understanding and analysis               | [Intelligent Transportation (Reason 1)](recipes/post_training/reason1/intelligent-transportation/post_training.md) |
| **Training** | AV video captioning and visual question answering for autonomous vehicles | [AV Video Caption VQA (Reason 1)](recipes/post_training/reason1/av_video_caption_vqa/post_training.md)  |
| **Training** | Temporal localization for MimicGen robot learning data generation         | [Temporal Localization](recipes/post_training/reason1/temporal_localization/post_training.md)           |
| **Training** | Wafer map anomaly classification with supervised fine-tuning on WM-811k  | [Wafer Map Classification](recipes/post_training/reason1/wafermap_classification/post_training.md)      |

### [**Cosmos Curator**](https://github.com/nvidia-cosmos/cosmos-curate)

| **Workflow** | **Description**                                      | **Link**                                                                        |
| ------------ | ---------------------------------------------------- | ------------------------------------------------------------------------------- |
| **Curation** | Curate video data for Cosmos Predict 2 post-training | [Predict 2 Data Curation](recipes/data_curation/predict2_data/data_curation.md) |
| **Analysis** | Advanced video clustering using Time Series K-Means on embedding trajectories | [Video Clustering with Time Series K-Means](recipes/data_curation/embedding_analysis/embedding_analysis.md) |

### **End-to-End Workflows**

| **Workflow** | **Description** | **Link** |
|--------------|-----------------|----------|
| **SDG Pipeline** | Complete synthetic data generation pipeline for traffic scenarios using CARLA, Cosmos Transfer 2.5, and Cosmos Reason 1 | [Smart City SDG](recipes/end2end/smart_city_sdg/workflow_e2e.md) |

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

**1. [Prompt Guide](core_concepts/prompt_guide/overview.md)** - Learn effective prompting strategies for Cosmos models. This covers message structure, media ordering, sampling parameters, and domain-specific patterns to help you get the best results from Cosmos Reason and other vision-language models.

**2. [Control Modalities](core_concepts/control_modalities/overview.md)** - Master precise control over video generation with Cosmos Transfer 2.5 using Edge, Depth, Segmentation, and Vis modalities. This covers structural preservation, semantic replacement, lighting consistency, and multi-control approaches for achieving high-fidelity, controllable video transformations.

**3. [Data Curation](core_concepts/data_curation/overview.md)** - Use Cosmos Curator to prepare your datasets with modular, scalable processing pipelines. This includes splitting, captioning, filtering, deduplication, task-specific sampling, and cloud-native or local execution.

**4. [Model Post-Training](core_concepts/post_training/overview.md)** - Fine-tune foundation models using your curated data. This covers domain adaptation for Predict (2 and 2.5), Transfer (1 and 2.5), and Reason 1, setup for supervised fine-tuning, LoRA, or reinforcement learning, and use of Cosmos RL for large-scale distributed rollout.

**5. [Evaluation and Quality Control](core_concepts/evaluation/overview.md)** - Ensure your post-trained models are aligned and robust through metrics, visualization, and qualitative inspection. Leverage Cosmos Reason 1 as a quality filter (e.g. for synthetic data rejection sampling).

**6. [Model Distillation](core_concepts/distillation/overview.md)** - Compress large foundation models into smaller, efficient variants while preserving performance. This includes knowledge distillation techniques for Cosmos models, teacher-student training setups, and deployment optimization for edge devices and resource-constrained environments.

## Gallery

Visual examples of Cosmos Transfer results across Physical AI domains:

- **[Robotics Domain Adaptation](gallery/robotics_inference.md)** - Sim-to-real transfer for robotic manipulation with varied materials, lighting, and environments.
- **[Autonomous Vehicle Domain Adaptation](gallery/av_inference.md)** - Multi-control video generation for driving scenes across different weather, lighting, and time-of-day conditions.

## Quick Start Paths

This cookbook provides flexible entry points for both **inference** and **training** workflows. Each section contains runnable scripts, technical recipes, and complete examples.

- **Inference workflows:** [Getting Started](getting_started/setup.md) for setup and immediate model deployment
- **Physical AI datasets:** [NVIDIA Physical AI Collection](https://huggingface.co/collections/nvidia/physical-ai) on Hugging Face for curated datasets across domains
- **Data processing:** [Data Processing & Analysis](core_concepts/data_curation/overview.md) for content analysis workflows
- **Training workflows:** [Model Training & Fine-tuning](core_concepts/post_training/overview.md) for domain adaptation
- **Case study recipes:** [Case Study Recipes](#case-study-recipes) organized by application area
