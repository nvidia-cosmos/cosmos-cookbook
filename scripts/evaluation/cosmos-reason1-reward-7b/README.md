# Cosmos-Reason1-7B-Reward Video Anomaly Detection

This repository contains inference scripts and tools for the **Cosmos-Reason1-7B-Reward** model, a video reward model designed to detect physical anomalies and artifacts in videos.

## Getting Started in 5 Minutes

**Want to analyze a video right away? Here's the fastest path:**

```bash
# 1. Download model (replace with your HF token)
./download_checkpoints.sh your_hf_token_here

# 2. Install dependencies
pip install torch torchvision transformers mediapy numpy pillow qwen-vl-utils

# 3. Analyze a single video
python inference.py --video your_video.mp4 --checkpoint ./checkpoints

# 4. Or process multiple videos at once
python batch_inference.py --checkpoint ./checkpoints --video-dir ./your_videos --output-dir ./results
```

That's it! Your videos will be analyzed for physical anomalies and artifacts.

## Overview

The Cosmos-Reason1-7B-Reward model analyzes videos to identify physical inconsistencies, anomalies, and artifacts by focusing on:

- **Physics violations**: Gravity, collision, fluid dynamics
- **Object behavior**: Permanence, interaction, cause-and-effect
- **Human motion**: Realistic body movement and joint constraints
- **Common sense**: Functional and logical object behavior

The model provides both binary classification (Yes/No for anomalies) and continuous scoring with optional explanations.

## Directory Structure

```
cosmos-reason1-reward-7b/
├── README.md                 # This file
├── inference.py              # Single video inference script
├── batch_inference.py        # Batch processing script
├── download_checkpoints.sh   # Model download script
├── checkpoints/              # Model files and weights
│   ├── README.md            # Model-specific documentation
│   ├── config.json          # Model configuration
│   ├── model-*.safetensors  # Model weights (4 parts)
│   ├── tokenizer*           # Tokenizer files
│   └── inference_video_reward.py  # Alternative inference script
└── output/                   # Generated results
    ├── good/                # Videos classified as anomaly-free
    └── bad/                 # Videos classified as containing anomalies
```

## Quick Start Workflow

Follow these steps to get up and running with the Cosmos-Reason1-7B-Reward model:

### Step 1: Download Model Checkpoints

First, download the pre-trained model weights using one of these methods:

#### Method A: Using the download script (Recommended)

```bash
# Make the script executable
chmod +x download_checkpoints.sh

# Download with HuggingFace token
./download_checkpoints.sh your_hf_token_here

# Or set token as environment variable
export HF_TOKEN=your_hf_token_here
./download_checkpoints.sh
```

#### Method B: Manual download with HuggingFace CLI

```bash
# Install HuggingFace CLI
pip install huggingface-hub

# Download model
huggingface-cli download nvidia/Cosmos-Reason1-7B-Reward --local-dir ./checkpoints --token your_hf_token_here
```

> **Note**: You need a HuggingFace account and token with access to the nvidia/Cosmos-Reason1-7B-Reward model.

### Step 2: Install Dependencies

```bash
# Install core dependencies
pip install torch torchvision transformers
pip install mediapy numpy pillow qwen-vl-utils

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 3: Single Video Inference

Analyze a single video file:

```bash
# Basic inference
python inference.py --video path/to/your/video.mp4 --checkpoint ./checkpoints

# Example with sample video
python inference.py --video sample_anomaly_video.mp4 --checkpoint ./checkpoints
```

**Expected Output:**

```
Loading model from: ./checkpoints
Processing video: sample_anomaly_video.mp4
Video: sample_anomaly_video.mp4
Physical accuracy: No
Score (high is good): 0.2341
```

### Step 4: Batch Processing Multiple Videos

Process all videos in a directory:

```bash
# Process all .mp4 files in a directory
python batch_inference.py --checkpoint ./checkpoints --video-dir ./test_videos --output-dir ./results

# Process with output in same directory as videos
python batch_inference.py --checkpoint ./checkpoints --video-dir ./test_videos
```

**What happens during batch processing:**

1. Scans directory for all `.mp4` files
2. Processes each video sequentially
3. Generates individual `.txt` result files
4. Copies videos to output directory (if specified)
5. Provides processing summary

**Example batch output:**

```
Found 15 videos in directory: ./test_videos
=== Processing videos ===
[1/15] Processing video1.mp4
  Prediction: No, Score: 0.8542
  Text output saved to: ./results/video1.txt
[2/15] Processing video2.mp4
  Prediction: Yes, Score: 0.1234
  Text output saved to: ./results/video2.txt
...
==================================================
PROCESSING SUMMARY
==================================================
Total videos found: 15
Successfully processed: 15
Errors encountered: 0
Output directory: ./results
```

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for faster inference)
- HuggingFace account with access to nvidia/Cosmos-Reason1-7B-Reward
- ~15GB free disk space for model checkpoints
- ~15GB VRAM for optimal GPU performance

## Usage

### Single Video Inference

Analyze a single video file:

```bash
python inference.py --video path/to/video.mp4 --checkpoint ./checkpoints
```

**Example:**

```bash
python inference.py --video sample_video.mp4 --checkpoint ./checkpoints
```

**Output:**

```
Video: sample_video.mp4
Physical accuracy: No
Score (high is good): 0.2341
```

### Batch Processing

Process multiple videos in a directory:

```bash
python batch_inference.py --checkpoint ./checkpoints --video-dir /path/to/videos --output-dir ./output
```

**Features:**

- Processes all `.mp4` files in the specified directory
- Generates individual `.txt` files with results for each video
- Organizes output into `good/` and `bad/` subdirectories
- Provides processing summary

**Example:**

```bash
python batch_inference.py --checkpoint ./checkpoints --video-dir ./test_videos --output-dir ./results
```

## Command Line Options

### inference.py

- `--video`: Path to input video file (required)
- `--checkpoint`: Path to model checkpoint directory (required)

### batch_inference.py

- `--checkpoint`: Path to trained model checkpoint (required)
- `--video-dir`: Directory containing video files (required)
- `--output-dir`: Output directory for results (optional, defaults to video directory)

### download_checkpoints.sh

- Accepts HuggingFace token as command line argument
- Can also use `HF_TOKEN` environment variable
- Interactive prompt if no token provided

## Model Details

The Cosmos-Reason1-7B-Reward model uses a sophisticated prompt-based approach to analyze videos:

### Analysis Framework

The model evaluates videos based on:

- **Gravity**: Objects following realistic gravitational behavior
- **Collision**: Proper object interaction and collision physics
- **Object Interaction**: Logical cause-and-effect relationships
- **Fluid Dynamics**: Realistic liquid and gas behavior
- **Object Permanence**: Consistent object existence and properties
- **Human Motion**: Natural body movement and joint constraints

### What the Model Ignores

- Animation style (cartoons are not automatically anomalous)
- Audio content (no sound-based analysis)
- Lighting, shadows, and camera effects
- Artistic style or background elements
- Overall visual impression

## Output Format

### Text Files

Each processed video generates a `.txt` file containing:

```
Video: filename.mp4
Physical accuracy: Yes/No
Score (high is good): 0.xxxx
Checkpoint: /path/to/checkpoint
```

### Scoring

- **Score Range**: 0.0 to 1.0
- **High Score**: Indicates good physical accuracy (fewer anomalies)
- **Low Score**: Indicates potential anomalies or artifacts
- **Binary Classification**: "Yes" = anomalies detected, "No" = no anomalies

## Performance Notes

- **GPU Recommended**: Model inference is optimized for CUDA-enabled GPUs
- **Memory Requirements**: ~15GB VRAM recommended for optimal performance
- **Processing Time**: Varies by video length and hardware (typically 10-30 seconds per video)
