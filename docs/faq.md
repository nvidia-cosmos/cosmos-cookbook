# Frequently Asked Questions

This document contains comprehensive FAQ information compiled from multiple sources.

## General Questions

### What is NVIDIA Cosmos?

**[NVIDIA Cosmos™](https://www.nvidia.com/en-us/ai/cosmos/)** is a world foundation model (WFM) development platform to advance physical AI. At its core are Cosmos WFMs, openly available pretrained multimodal models that developers can use out-of-the-box for generating world states as videos and physical AI reasoning, or post-train to develop specialized physical AI models. NVIDIA Cosmos also includes advanced tokenizers, guardrails, accelerated data-processing pipeline, and post-training scripts.

### What are the main components of Cosmos?

#### Cosmos World Foundation Models (WFMs)

Cosmos world foundation models (WFMs) are pretrained generative AI models for virtual world generation to advance physical AI. The WFM family includes:

- **Cosmos Predict** for generating future world states as videos
- **Cosmos Transfer** for conditioned synthetic data
- **Cosmos Reason** for physical AI reasoning

These models are fully customizable to develop specialized physical AI models.

#### Cosmos Curator

GPU-accelerated video preprocessing and curation toolkit for preparing high-quality datasets.

#### Cosmos Tokenizer

To efficiently convert visual data into tokens.

### Who is Cosmos designed for?

Cosmos is designed for developers and ISVs working in the following domains:

- Robotics
- Autonomous vehicles
- Simulation
- Computer vision applications

### What are the technical capabilities of the Cosmos platform?

The Cosmos platform provides the following capabilities:

- Pre-trained world foundation models (WFMs) for immediate deployment
- GPU-accelerated data processing and curation tools
- Post-training frameworks for domain-specific adaptation
- CUDA-optimized inference and training pipelines
- Synthetic data generation for physical AI model training

## Models

### What are the main use cases for Cosmos?

**Data Curation:** Cosmos platform includes Cosmos Curator for video data and video data search to accelerate data curation for developers working with vast amounts of real or synthetic data to train physical AI models.

**Accelerate synthetic data generation (SDG):** Cosmos WFMs are purpose-built to accelerate SDG in many ways.

- With **Cosmos Predict**, developers can generate synthetic data from a text prompt or a pair of images. Outputs include predictive next frames or interpolated frames—ideal for edge cases or exploring multiple scenarios from a single input.
- **Omniverse** creates realistic 3D scenes that can be used as an input also referred to as 'ground truth' for **Cosmos Transfer**, which amplifies them across diverse environments and lighting. This process generates photorealistic, scalable, augmented data for robot and autonomous vehicle training as well as computer vision applications.
- **Cosmos Reason** acts as a critic for synthetic data. It scores video inputs based on how well they match a text prompt and can generate captions to help curate training data.
- Any combination of these models can accelerate the synthetic data generation processes. Combined with [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac/sim), [AV Simulation](https://www.nvidia.com/en-us/use-cases/autonomous-vehicle-simulation/), [Isaac GR00T](https://developer.nvidia.com/isaac/gr00t), these models can unlock a variety of SDGs.

**Post-training:** Cosmos WFMs are fully customizable to develop downstream vision, robotics or autonomous vehicle foundation models tailored for customer data. Post-training can be done to change output type, output quantity, output quality, output style or output point of view.

### How do Cosmos models differ from other video foundation models?

Cosmos world foundation models are designed specifically for physical AI applications. The models are openly available and customizable, with Cosmos Predict and Cosmos Reason supporting post-training for autonomous vehicle, robotics, and vision-action generation models.

### What is Policy Initialization?

Policy Initialization is the process of developing a policy model from the world foundation model (WFM) by modifying its output head. A Policy model maps observed states (e.g. video) to actions. It can be initialized by post-training a Cosmos world foundation model with a new output head tailored for action selection (from video head → action head).

### What is Policy Evaluation?

Policy Evaluation is the process of assessing a trained policy model. It can be done by conditioning with/seeding specific inputs (e.g. actions or instructions) and analyzing model output. This step ensures that the model correctly maps states to actions and performs as expected in real-world or simulated environments.

### Can Cosmos be used for creative content generation?

Cosmos models can generate video content under the NVIDIA Open Model License, but the platform is primarily designed for physical AI applications rather than creative content generation.

### What is multiverse simulation?

Multiverse simulation involves generating multiple future outcomes from a given state. Cosmos, integrated with NVIDIA Omniverse, enables simulation of multiple scenarios for tasks such as predictive maintenance and autonomous decision-making.

### Is Cosmos Reason a VLM, VLA, or MLLM?

Cosmos Reason 1 is a physical reasoning engine designed to analyze real-world scenarios through natural language explanations. It functions as a Vision Language Model (VLM) or a Multi-Modal Large Language Model (MLLM) with chain-of-thought reasoning built-in. Unlike Vision-Language-Action (VLA) models, which map sensory inputs to executable actions, Cosmos Reason 1 employs hierarchical ontologies for space, time, and physics to generate text-based reasoning traces about safety, causality, and object interactions.

### What does Cosmos Reason output?

While VLAs output motor commands, Cosmos Reason 1 produces insights in text like "The slope exceeds the vehicle's tilt tolerance" that require translation layers for robotic execution. The models can do high-level planning and explainable safety checks but cannot directly control actuators or navigate dynamic environments.

## Technical Details

### What is 3D consistency, and how is it tested?

3D consistency measures how well models maintain spatial alignment in 3D scenes. Cosmos testing methodology:

**Test Setup**: Static scenes from 500 curated videos using the following metrics:

- Geometric consistency (e.g. Sampson error, pose estimation)
- View synthesis consistency (PSNR, SSIM, LPIPS)

**Key Metrics**:

- **Sampson error**: Lower values indicate better geometric accuracy
- **Pose estimation success rate**: Higher percentages reflect better camera alignment
- **PSNR and SSIM**: Higher scores indicate higher quality in synthesized views
- **LPIPS**: Lower values indicate better perceptual similarity

### What is physics alignment, and how is it evaluated?

Physics alignment tests models' ability to simulate physical dynamics like gravity and collisions.

**Evaluation Method**: Controlled scenarios in virtual environments, assessed with multiple metrics

**Key Metrics**:

- **PSNR**: Higher values show less noise and better pixel accuracy
- **SSIM**: Higher values reflect better visual fidelity
- **DreamSim**: Evaluates semantic consistency of objects and motion
- **IoU**: Measures overlap between predicted and actual object regions for alignment with physical expectations

### What precision strategy do the Cosmos world foundation models use?

**Training Strategy**: Mixed-precision approach

- Maintains copies of weights in FP32 and BF16
- Gradients computed in BF16 only
- Final storage and inference in BF16

**Current Limitations**:

- The repositories do not currently support FP8
- FP8 and FP4 training capabilities are work in progress

### Infrastructure Requirements for Post-Training

#### Minimum Setup

**For Cosmos Reason 1-7B**:

- **SFT Training**: Minimum 2x 80GB GPUs
- **RL Training**: Minimum 4x 80GB GPUs

**General Requirements**:

- NVIDIA GPU with sufficient memory
- CUDA toolkit compatibility
- High-speed interconnects for distributed training

#### Distributed Training Requirements

**Networking**:

- **Recommended**: InfiniBand or RoCE for efficient communication
- **Supported**: AWS EFA
- **Essential**: High-bandwidth, low-latency connections for multi-GPU setups

### Optimization Strategies

**Pipeline Optimization**:

- Ray-based pipeline allows specification of GPU types.
- Dynamic hardware detection supported.
- Mixed GPU types can be leveraged for different pipeline stages.
- Telemetry ensures efficient resource utilization.

**Memory Management**:

- Requirements vary by model size and dataset characteristics.
- Prompt length and chain-of-thought length affect memory needs.
- Horizontal scaling supported for large deployments.

### Performance Characteristics

#### Compression and Quality

**Video Compression Impact**:

- Higher compression rates can affect generation quality.
- Temporal compression with reduced tokens may lower quality
- Optimal settings depend on specific application requirements.
- Testing recommended to find best trade-offs.

#### Processing Performance

**Cosmos Curator Performance**:

- GPU-accelerated processing compared to CPU-based pipelines.
- Optimized for large-scale video processing workloads.

**Tokenizer Performance**:

- Optimized compression and processing for video data
- Supports both training and inference workloads.

### Model Architecture Details

#### Training Configuration

**Layer Management**:

- No layers are frozen during SFT and RL training.
- Full model fine-tuning approach.
- No specific attention mechanism modifications required.

**Memory Recommendations**:

- Model size determines base memory requirements.
- Dataset characteristics (video length, resolution) affect memory needs.
- Multi-GPU training recommended for larger models.

#### Input Specifications

**Video Input Guidelines**:

- **Recommended FPS**: 4 frames per second
- **Token Budget**: Centered at 8k tokens, randomized within the [6k, 10k] range
- **Total Pixels**: Approximately 8k × 28 × 28 × 2
- **Generalization**: Model can handle inputs outside training bounds with potential quality degradation

### Scalability Considerations

#### Database Scaling

**Vector Database Performance**:

- Horizontal scaling for large datasets
- Optimized indexing for bulk data ingestion
- Resource-efficient scaling (spin down after ingestion)
- Fast search times maintained at scale

#### GPU Resource Management

**Dynamic Allocation**:

- Ray-based pipeline supports multiple GPU types.
- Dynamic hardware detection and optimization
- Efficient resource utilization through telemetry
- Mixed hardware configurations supported.

## Licensing & Availability

### What is the licensing model for Cosmos models?

Cosmos world foundation models are available under the **NVIDIA Open Model License Agreement**, which permits the following:

- Commercial use without payment requirements
- No company size restrictions
- Synthetic data generation
- Post-training and derivative model development
- Model distribution and modification

### Enterprise Options

#### Open Source vs Enterprise

- **Model weights and scripts**: Open source and free
- **Basic development tools**: Available under permissive licenses
- **Enterprise features**: Available through NVIDIA AI Enterprise (NVAIE)

#### NVIDIA AI Enterprise Features

- Optimized NIMs for enhanced inference performance
- Advanced NeMo features and maintenance
- Professional support and updates
- Production deployment tools

#### License Compatibility

Existing NVIDIA Omniverse Enterprise (NVOVE) licenses can be used for Cosmos entitlements.

### Getting Support

#### Community Resources

- **GitHub Issues**: Report bugs and request features in relevant repositories
- **Documentation**: Comprehensive guides in each repository
- **Examples**: Reference implementations and tutorials
- **Community Forums**: Engage with other developers

#### Official Channels

- **NVIDIA Developer Portal**: Latest updates and announcements
- **build.nvidia.com**: Try models and access NIMs
- **NVIDIA AI Catalog**: Enhanced text prompt tools and specialized models

#### Enterprise Support

For enterprise deployments:

- NVIDIA AI Enterprise subscriptions include professional support
- Dedicated technical assistance for production deployments
- Regular updates and optimizations
- Custom integration guidance

### Legal and Compliance

#### License Terms

The [NVIDIA Open Model License Agreement](https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf) covers:

- Commercial usage rights
- Distribution permissions
- Modification allowances
- Attribution requirements

### General Questions

#### Q: Do I need to pay to use Cosmos models?

**A**: No, the core models are freely available under the NVIDIA Open Model License.

#### Q: Can I use Cosmos for commercial applications?

**A**: Yes, commercial use is permitted without restrictions.

#### Q: What if I need enterprise-grade support?

**A**: NVIDIA AI Enterprise provides optimized tools and professional support.

#### Q: Are there any usage restrictions?

**A**: Use must comply with the NVIDIA Open Model License Agreement and applicable laws.

#### Q: Can I modify and redistribute the models?

**A**: Yes, modification and redistribution are permitted under the license terms.
