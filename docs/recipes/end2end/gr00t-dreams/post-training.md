# Leveraging World Foundation Models for Synthetic Trajectory Generation in Robot Learning

> **Author:** [Rucha Apte](https://www.linkedin.com/in/ruchaa-apte/), [Saurav Nanda](https://www.linkedin.com/in/sauravnanda/), [Jingyi Jin](https://www.linkedin.com/in/jingyi-jin/)
> **Organization:** NVIDIA


| **Model**                                                                | **Workload**             | **Use Case**                                   |
| ------------------------------------------------------------------------ | ------------------------ | ---------------------------------------------- |
| [Cosmos Predict 2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) | Post-training, Inference | Synthetic Trajectory Generation                |
| [Cosmos Reason 2](https://github.com/nvidia-cosmos/cosmos-reason2)       | Inference                | Reasoning and filtering synthetic trajectories |


This guide walks you through post-training the Cosmos Predict 2.5 model on the [PhysicalAI-Robotics-GR00T-GR1](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-GR1) open dataset to generate synthetic robot trajectories for robot learning applications. After post-training, we'll use the fine-tuned model to generate trajectory predictions on the [PhysicalAI-Robotics-GR00T-Eval](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Eval) dataset. Finally, Cosmos Reason 2 is leveraged to evaluate these generated trajectories by assessing their physical plausibility, helping to quantify and filter for valid, realistic, and successful robot motions.

## Motivation

Generalist robotics is emerging, driven by advances in mechatronics and robot foundation models, but scaling skill learning remains limited by the need for massive training data. [NVIDIA Isaac GR00T-Dreams](https://github.com/nvidia/gr00t-dreams), built on [NVIDIA Cosmos](https://www.nvidia.com/en-us/ai/cosmos/?sortBy=developer_learning_library%2Fsort%2Ffeatured_in.cosmos%3Adesc%2Ctitle%3Aasc&hitsPerPage=6), addresses this by generating large-scale synthetic trajectory data from a single image and language prompt. This enables efficient training of models such as [NVIDIA Isaac GR00T N1.5](https://developer.nvidia.com/isaac/gr00t) for reasoning and skill learning.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Preparing Data](#1-preparing-data)
- [Post-training Cosmos Predict 2.5](#2-post-training-cosmos-predict)
  - [Configuration](#21-configuration)
  - [Training](#22-training)
- [Inference with Post-trained Predict 2.5](#3-inference-with-post-trained-cosmos-predict)
  - [Converting DCP Checkpoint to Consolidated PyTorch Format](#31-converting-dcp-checkpoint-to-consolidated-pytorch-format)
  - [Running Inference](#32-running-inference)
- [Inference with Reason 2 for plausibilty check and filtering](#4-inference-with-Reason2-for-plausibility-check-and-filtering)

## Prerequisites

Follow the [Setup guide](./setup.md) for general environment setup instructions, including installing dependencies for Cosmos Predict 2.5 and Cosmos Reason 2.

## 1. Preparing Data

First, we will download the [GR1 training dataset](https://huggingface.co/datasets/nvidia/GR1-100) and then preprocess it to create text prompt txt files for each video.

Download DreamGen Bench Training Dataset 

```bash
cd cosmos-predict2.5 
```

```bash
hf download nvidia/GR1-100 --repo-type dataset --local-dir datasets/benchmark_train/hf_gr1/ && \
mkdir -p datasets/benchmark_train/gr1/videos && \
mv datasets/benchmark_train/hf_gr1/gr1/*mp4 datasets/benchmark_train/gr1/videos && \
mv datasets/benchmark_train/hf_gr1/metadata.csv datasets/benchmark_train/gr1/
```

Preprocess DreamGen Bench Training Dataset

```bash
python -m scripts.create_prompts_for_gr1_dataset --dataset_path datasets/benchmark_train/gr1
```

Upon running the above preprocessing, the dataset folder format should look like this:

```bash
datasets/benchmark_train/gr1/
├── metas/
│   ├── *.txt
├── videos/
│   ├── *.mp4
├── metadata.csv
```

Preview of the Training Dataset


| Input Prompt File | Video File |
| ----------------- | ---------- |
| The robot arm is performing a task. Use the right hand to pick up green bok choy from tan table right side to bottom level of wire basket. | <video src="assets/1.mp4" controls width="320"></video> |
| The robot arm is performing a task. Use the right hand to pick up rubik's cube from top level of the shelf to bottom level of the shelf. | <video src="assets/2.mp4" controls width="320"></video> |
| The robot arm is performing a task. Use the right hand to pick up banana from teal plate to wooden table. | <video src="assets/3.mp4" controls width="320"></video> |

## 2. Post Training Cosmos Predict

## 3. Inference with Post Trained Cosmos Predict

