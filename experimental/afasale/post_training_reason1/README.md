# Cosmos Reason1 Post-Training Pipeline

This repository contains scripts and configurations for post-training the Cosmos Reason1 model using supervised fine-tuning (SFT) on automotive video datasets.

## Overview

The pipeline consists of three main stages:
1. **Caption Generation**: Generate detailed captions for video datasets using Vision Language Models
2. **Q&A Dataset Creation**: Convert captions into structured question-answer pairs
3. **Supervised Fine-Tuning**: Fine-tune the Cosmos Reason1 model on the generated Q&A dataset

## Prerequisites

Ensure you have access to the Cosmos post-training environment:

```bash
cd examples/post_training
just install
source .venv/bin/activate
```

## Nexar Dataset and Ground Truth
-  Train: ```s3://lha-datasets/uber/nexar/DESCR_caption/v2/train/```
-  Eval: ```s3://lha-datasets/uber/nexar/DESCR_caption/v2/eval/```
-  Ground Truth of Eval(Human Generated): ```s3://lha-datasets/uber/nexar/DESCR_caption/v2/ground_truth/nexar_eval_ground_truth.json```

## Usage

### Step 1: Generate Video Captions

Generate captions from video datasets using the Qwen2-VL-7B-Instruct model:

```bash
./cosmos-curate/scripts/generate_captions.py \
    ./datasets/nexar_itr5/train \
    ./datasets/nexar_itr5/train_captions_qwen_itr1 \
    --model Qwen/Qwen2-VL-7B-Instruct
```

**Configuration**: 
- **`cosmos-curate/configs/prompt_0.yaml`**: Prompt template for caption generation focusing on ego vehicle behavior analysis
- **`cosmos-curate/configs/vision_config.json`**: Video processing parameters (fps, max_pixels)
- **`cosmos-curate/configs/generation_config.json`**: Language model generation settings (temperature, top_p, etc.)

### Step 2: Create Q&A Dataset

Extract structured question-answer pairs from the generated captions:

```bash
# Extract question-answer pairs from the generated captions using the provided script.
# This script reads the HuggingFace dataset directory containing model-generated captions,
# parses the JSON-formatted answers, and outputs a new HuggingFace dataset with structured QA pairs.
./cosmos-curate/scripts/generate_qa_dataset.py \
    ./datasets/nexar_itr5/train_captions_qwen_itr1 \
    ./datasets/nexar_itr5/train_captions_qwen_qa_itr1
```
> **TODO:** Extend the pipeline to support multiple QA pairs per single video (currently assumes one QA set per video). This may require updating the caption generation prompt, the QA extraction script, and the dataset structure to handle multiple question-answer pairs for each video.


### Step 3: Launch Post-Training

Run supervised fine-tuning on the Cosmos Reason1 model:

```bash
cosmos-rl --config config/sft_nexar.toml ./script/sft_nexar.py
```

## Configuration Files

- **`config/sft_nexar.toml`**: Main training configuration including model settings, batch sizes, and output directories
- **`script/sft_nexar.py`**: Training script for supervised fine-tuning (SFT) using HuggingFace datasets.  

