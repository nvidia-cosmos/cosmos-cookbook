# Overview

This example demonstrates a use case for Visual Question Answering (VQA) on Autonomous Vehicle (AV) data using the Cosmos-Reason1 model. VQA examples in the autonomous driving domain can encompass various types of problems including perception (object detection, scene understanding), action recognition (lane changes, acceleration patterns), and complex reasoning tasks (safety assessment, behavioral analysis).

## Model Background

The current Cosmos-Reason1 model has been densely trained on AV and robotics data, giving it improved accuracy in scene understanding compared to off-the-shelf models. However, there are still opportunities for improvement, particularly in domain-specific tasks.

## Objective

This use case demonstrates how the post-training process can improve the model's accuracy on AV-related VQA tasks. The goal is to track improvements of the Reason model in answering a set of pre-defined multiple-choice questions through the post-training process, ultimately increasing the model's accuracy on these specific tasks.

## Dataset

We use the [Nexar Collision Prediction dataset](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction) as an open-access proxy for demonstration. The Nexar collision prediction dataset comprises videos from Nexar dashcams with the following characteristics:

- **Resolution**: 1280x720 at 30 frames per second
- **Duration**: Typically about 40 seconds per video
- **Size**: 1,500 videos total
- **Annotations**: Time of event collision or near-miss events. We are not using the annotations for our VQA problem as the problem domain is not only on collision detection.
- **License**: Copyright (c) 2025 Nexar Inc. - Permission granted for use, copy, modify, and distribute with appropriate attribution

## Evaluation Questions

The model's performance is evaluated on the following multiple-choice questions:

**Q1: Did the ego vehicle change lanes in the video?**

- (A) Yes
- (B) No
- (C) Not sure
- (D) Not applicable

**Q2: What is the ego vehicle's acceleration?**

- (A) Speeding up
- (B) Slowing down
- (C) Constant speed
- (D) Not applicable

**Q3: Choose the most accurate description of the ego vehicle's behavior.**

- (A) Stopped
- (B) Driving forward
- (C) Driving backward
- (D) Turning left
- (E) Turning right
- (F) None

**Q4: Is there a pedestrian walking in front of the ego vehicle?**

- (A) Yes
- (B) No
- (C) Not sure

## Post-Training Process

Post-training is a two-stage process designed to enhance the model's performance on these specific AV VQA tasks:

1. **Supervised Fine Tuning (SFT)**: Distill AV video reasoning from an existing LLM (e.g. DeepSeek-R1).
2. **Reinforcement Learning (RL)**: Fine-tune AV video understanding on high quality QA pairs.
