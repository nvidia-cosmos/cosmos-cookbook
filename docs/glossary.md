# Glossary

## C

**Chain of Thought (CoT)**
: A reasoning technique where models generate step-by-step explanations of their thought process before arriving at a final answer, improving transparency and accuracy.

**Checkpoint**
: A saved snapshot of a model's weights and training state at a specific point during training, allowing training to be resumed or models to be evaluated at different stages.

**Context Parallelism (CP)**
: A parallelization strategy that splits the sequence/context dimension across multiple devices to handle longer sequences.

**Cosmos Curator**
: A GPU-accelerated video curation pipeline built on Ray for multi-model analysis, content filtering, annotation, and deduplication of inference and training data.

**Cosmos Predict**
: A diffusion transformer model for future state prediction and video-to-world generation, with specialized variants for robotics and simulation.

**Cosmos Reason**
: A 7B vision-language model for physically grounded reasoning, handling spatial/temporal understanding and chain-of-thought tasks for embodied AI applications.

**Cosmos RL**
: A distributed training framework supporting supervised fine-tuning (SFT) and reinforcement learning approaches with elastic policy rollout and FP8/FP4 precision support.

**Cosmos Transfer**
: A multi-control video generation system with ControlNet and MultiControlNet conditioning (depth, segmentation, LiDAR, HDMap) including 4K upscaling capabilities.

**ControlNet**
: A neural network architecture that adds conditional control to diffusion models, allowing them to be guided by additional inputs like depth maps, edge maps, or segmentation masks.

## D

**Data Parallel (DP)**
: A training parallelization strategy where the model is replicated across multiple devices, and each device processes different data batches.

**Data Parallelism Shard Size**
: The number of devices across which gradients are synchronized in distributed training.

**Deduplication**
: The process of identifying and removing duplicate or near-duplicate samples from a dataset to improve data quality and training efficiency.

**Diffusion Model**
: A generative model that learns to create data by iteratively denoising samples, starting from pure noise and gradually refining them into coherent outputs.

## E

**Embodied AI**
: AI systems that interact with the physical world through sensors and actuators, such as robots and autonomous vehicles.

**Epoch**
: A complete pass through the entire training dataset during model training.

## F

**Fine-Tuning**
: The process of adapting a pre-trained model to a specific task or domain by training it on task-specific data.

**FP4/FP8**
: 4-bit and 8-bit floating-point number formats that reduce memory usage and increase training speed while maintaining acceptable model performance.

**FPS (Frames Per Second)**
: The number of video frames processed or generated per second.

**FSDP (Fully Sharded Data Parallel)**
: A memory-efficient distributed training strategy that shards model parameters, gradients, and optimizer states across multiple devices.

## G

**Gradient Checkpointing**
: A memory-saving technique that trades computation for memory by recomputing intermediate activations during backpropagation instead of storing them.

**Gradient Clipping**
: A technique to prevent exploding gradients by capping the gradient norm at a maximum value during training.

## H

**HDMap (High-Definition Map)**
: A detailed, lane-level map representation used in autonomous driving that includes precise road geometry, lane markings, and traffic rules.

## I

**Inference**
: The process of using a trained model to make predictions or generate outputs on new, unseen data.

**Interactive Meta-Action**
: In autonomous driving, a driving behavior that involves interaction with other traffic participants, such as yielding, following, or overtaking.

**ITS (Intelligent Transportation Systems)**
: Advanced applications that aim to provide innovative services relating to different modes of transport and traffic management.

## L

**LoRA (Low-Rank Adaptation)**
: An efficient fine-tuning method that adds trainable low-rank matrices to pre-trained model weights, reducing the number of trainable parameters.

**LLM (Large Language Model)**
: A neural network model trained on vast amounts of text data, capable of understanding and generating human-like text.

## M

**Max Pixels**
: The maximum number of pixels to process in an image or video frame, often used to control computational requirements.

**Model Checkpoint**
: See **Checkpoint**.

**Multi-Control**
: The ability to condition a generative model on multiple types of control signals simultaneously (e.g., depth, segmentation, and HDMap).

**MultiControlNet**
: An extension of ControlNet that combines multiple conditional control signals to guide video generation with greater precision.

## N

**Non-Interactive Meta-Action**
: In autonomous driving, a driving behavior that doesn't directly involve other traffic participants, such as lane merging or turning at an empty intersection.

## O

**Optimizer**
: An algorithm that adjusts model weights during training to minimize the loss function. Common optimizers include Adam, AdamW, and SGD.

## P

**Parallelism**
: The strategy of distributing computation across multiple devices to speed up training or inference. See also **Data Parallel**, **Tensor Parallel**, **Pipeline Parallel**.

**Physical Plausibility**
: The degree to which generated or predicted content adheres to real-world physics laws and constraints.

**Pipeline Parallel (PP)**
: A training strategy that splits a model into sequential stages across multiple devices, with different devices processing different layers.

**Post-Training**
: The process of further training or fine-tuning a pre-trained model on specific tasks or domains after initial training is complete.

## R

**Reinforcement Learning (RL)**
: A machine learning approach where models learn by receiving rewards or penalties based on their actions, optimizing for long-term cumulative reward.

**Reward Model**
: A model trained to score or evaluate outputs, used in reinforcement learning to provide feedback signals for training.

## S

**Scene Understanding**
: The ability of a model to interpret and comprehend the contents, context, and relationships within a visual scene.

**SFT (Supervised Fine-Tuning)**
: A training method that fine-tunes a pre-trained model using labeled examples with supervised learning objectives.

**Sim-to-Real (Sim2Real)**
: The process of transferring knowledge or models trained in simulation environments to real-world applications.

**Spatial AI**
: AI systems that understand and reason about spatial relationships, positions, and interactions in physical environments.

**System Prompt**
: Initial instructions or context provided to a language model that define its role, behavior, or constraints for a conversation or task.

## T

**Tensor Parallel (TP)**
: A parallelization strategy that splits individual tensors (model layers) across multiple devices.

**Traffic Participant**
: Any entity in a traffic environment, including vehicles, pedestrians, cyclists, and other road users.

**Training Configuration**
: A set of hyperparameters and settings that define how a model is trained, including learning rate, batch size, and optimization strategy.

**Transfer Learning**
: The technique of applying knowledge learned from one task or domain to improve performance on a different but related task or domain.

## U

**Upscaling**
: The process of increasing the resolution of an image or video while attempting to preserve or enhance quality.

## V

**Video Augmentation**
: Techniques for modifying or enhancing video data, such as changing weather conditions, lighting, or style, to increase dataset diversity.

**Video-Language Model (VLM)**
: A neural network model that processes both video and language inputs, capable of understanding visual content and generating or responding to text.

**Visual Question Answering (VQA)**
: A task where models answer questions about the content of images or videos.

**VRU (Vulnerable Road User)**
: Traffic participants who are not protected by a vehicle structure, including pedestrians, cyclists, and motorcyclists.

## W

**Warmup Steps**
: An initial training period where the learning rate gradually increases from a small value to the target learning rate, helping stabilize early training.

**Weight Decay**
: A regularization technique that adds a penalty proportional to the magnitude of model weights to the loss function, helping prevent overfitting.

**WFM (World Foundation Model)**
: Large-scale foundation models designed to understand and generate representations of the physical world, forming the basis of the Cosmos ecosystem.

## Z

**Zero-Shot**
: The ability of a model to perform a task without having been explicitly trained on examples of that specific task, relying only on pre-training knowledge.
