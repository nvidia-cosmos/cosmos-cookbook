# LoRA Post-training for Sports Video Generation

> **Author:** [Arslan Ali](https://www.linkedin.com/in/arslan-ali-ph-d-5b314239/)
> **Organization:** NVIDIA

| **Model** | **Workload** | **Use Case** |
|-----------|--------------|--------------|
| [Cosmos Predict 2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) | Post-training | Sports Video Generation |

This guide provides instructions on running LoRA (Low-Rank Adaptation) post-training with the Cosmos Predict 2.5 models for sports video generation tasks, supporting Text2World, Image2World, and Video2World generation modes.

## Motivation

While the base Cosmos Predict 2.5 model excels at general video generation, sports content demands specialized understanding of athletic dynamics and game rules. Post-training addresses critical gaps in **player kinematic realism and physics**, ensuring natural body movements and accurate ball trajectories. The adapted model achieves higher **rule-coherence scores** by respecting sport-specific constraints like offside lines, field boundaries, and valid player positions. Additionally, post-training significantly improves **identity consistency**, maintaining stable player appearances, jersey numbers, and team colors throughout generated sequences—essential for realistic sports simulation and analysis applications.

## Table of Contents

- [Prerequisites](#prerequisites)
- [What is LoRA?](#what-is-lora)
- [Preparing Data](#1-preparing-data)
- [LoRA Post-training](#2-lora-post-training)
  - [Configuration](#21-configuration)
  - [Training](#22-training)
- [Inference with LoRA Post-trained checkpoint](#3-inference-with-lora-post-trained-checkpoint)
  - [Converting DCP Checkpoint to Consolidated PyTorch Format](#31-converting-dcp-checkpoint-to-consolidated-pytorch-format)
  - [Running Inference](#32-running-inference)

## Prerequisites

### 1. Environment Setup

Follow the [Setup guide](./setup.md) for general environment setup instructions, including installing dependencies.

### 2. Hugging Face Configuration

Model checkpoints are automatically downloaded during post-training if they are not present. Configure Hugging Face as follows:

```bash
# Login with your Hugging Face token (required for downloading models)
hf auth login

# Set custom cache directory for HF models
# Default: ~/.cache/huggingface
export HF_HOME=/path/to/your/hf/cache
```

> **💡 Tip**: Ensure you have sufficient disk space in `HF_HOME`.

### 3. Training Output Directory

Configure where training checkpoints and artifacts will be saved:

```bash
# Set output directory for training checkpoints and artifacts
# Default: /tmp/imaginaire4-output
export IMAGINAIRE_OUTPUT_ROOT=/path/to/your/output/directory
```

> **💡 Tip**: Ensure you have sufficient disk space in `IMAGINAIRE_OUTPUT_ROOT`.

## Weights & Biases (W&B) Logging

By default, training will attempt to log metrics to Weights & Biases. You have several options:

### Option 1: Enable W&B

To enable full experiment tracking with W&B:

1. Create a free account at [wandb.ai](https://wandb.ai)
2. Get your API key from [https://wandb.ai/authorize](https://wandb.ai/authorize)
3. Set the environment variable:

    ```bash
    export WANDB_API_KEY=your_api_key_here
    ```

> ⚠️ **Security Warning:** Store API keys in environment variables or secure vaults. Never commit API keys to source control.

### Option 2: Disable W&B

Add `job.wandb_mode=disabled` to your training command to disable wandb logging.

## What is LoRA?

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that allows you to adapt large pre-trained models to specific domains or tasks by training only a small number of additional parameters.

### Key Benefits of LoRA Post-Training

- **Memory Efficiency**: Only trains ~1-2% of total model parameters
- **Faster Training**: Significantly reduced training time per iteration
- **Storage Efficiency**: LoRA checkpoints are much smaller than full model checkpoints
- **Flexibility**: Can maintain multiple LoRA adapters for different domains
- **Preserved Base Capabilities**: Retains the original model's capabilities while adding domain-specific improvements

### When to Use LoRA vs Full Fine-tuning

**Use LoRA when:**

- You have limited compute resources
- You want to create domain-specific adapters
- You need to preserve the base model's general capabilities
- You're working with smaller datasets

**Use full fine-tuning when:**

- You need maximum model adaptation
- You have sufficient compute and storage
- You're making fundamental changes to model behavior

## 1. Preparing Data

### 1.1 Understanding Training Data Requirements

The training approach uses the same video dataset to train all three generation modes:

- **Text2World (0 frames)**: Uses only text prompts, videos serve as ground truth for reconstruction
- **Image2World (1 frame)**: Uses first frame as condition, generates remaining frames
- **Video2World (2+ frames)**: Uses initial frames as condition, continues the video generation

### 1.2 Dataset Location

The sports dataset should be organized in a directory structure that you'll specify in the configuration. Set your dataset path to point to your video collection:

```
/path/to/sports/videos
```

Replace this path with the actual location of your sports video dataset. The dataset should contain sports video clips in **MP4 format** at 720p resolution (704x1280). For the current training, we prepared approximately **4,350 training clips** and **50 clips for validation/inference** to support both Image2World and Video2World generation tasks.

### 1.3 Dataset Formats

The system supports two caption formats:

#### Text Format (.txt files)

- Simple text files containing one caption per file
- Files should be placed in a `metas/` directory
- Filename should match the video filename (e.g., `video1.mp4` → `video1.txt`)

#### JSON Format (.json files)

- More flexible format supporting multiple prompt variations
- Files should be placed in a `captions/` directory
- Supports long, short, and medium prompt types

**JSON Caption File Format:**

```json
{
  "model_name": {
    "long": "Detailed description of the sports action...",
    "short": "Brief summary of the sports play...",
    "medium": "Moderate length description of the sports scene..."
  }
}
```

## 2. LoRA Post-training

### 2.1 Configuration

Two configurations for sports are provided:

- `predict2_lora_training_2b_cosmos_sports_assets_txt` - For text caption format
- `predict2_lora_training_2b_cosmos_sports_assets_json_rank32` - For JSON caption format with long prompts

The configurations can be found in the comsmos predict-2.5 github.

#### Complete Sports Configuration

```python
from imaginaire.lazy_config import LazyCall as L
from projects.cosmos.predict2.datasets.local_datasets.dataset_video import (
    VideoDataset,
    get_generic_dataloader,
    get_sampler,
)

# Dataset configuration for sports videos (Text format)
example_dataset_cosmos_sports_assets_lora_txt = L(VideoDataset)(
    dataset_dir="/path/to/sports/videos",
    num_frames=93,
    video_size=(704, 1280),
)

# Dataset configuration for sports videos (JSON format with long prompts)
example_dataset_cosmos_sports_assets_lora_json = L(VideoDataset)(
    dataset_dir="/path/to/sports/videos",
    num_frames=93,
    video_size=(704, 1280),
    caption_format="json",
    prompt_type="long",
)

# Dataloader configuration
dataloader_train_cosmos_sports_assets_lora_txt = L(get_generic_dataloader)(
    dataset=example_dataset_cosmos_sports_assets_lora_txt,
    sampler=L(get_sampler)(dataset=example_dataset_cosmos_sports_assets_lora_txt),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)

dataloader_train_cosmos_sports_assets_lora_json = L(get_generic_dataloader)(
    dataset=example_dataset_cosmos_sports_assets_lora_json,
    sampler=L(get_sampler)(dataset=example_dataset_cosmos_sports_assets_lora_json),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)

# Model configuration with LoRA
_lora_model_config = dict(
    config=dict(
        # Enable LoRA training
        use_lora=True,
        # LoRA configuration parameters
        lora_rank=32,              # Rank of LoRA adaptation matrices (higher for sports complexity)
        lora_alpha=32,             # LoRA scaling parameter
        lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
        init_lora_weights=True,    # Properly initialize LoRA weights

        # Training configuration for all three modes
        # The model will randomly sample between 0, 1, and 2 conditional frames during training
        min_num_conditional_frames=0,  # Allow text2world (0 frames)
        max_num_conditional_frames=2,  # Allow up to video2world (2 frames)

        # Probability distribution for sampling number of conditional frames
        # This controls how often each mode is trained:
        # - 0 frames: text2world (33.3%)
        # - 1 frame: image2world (33.3%)
        # - 2 frames: video2world (33.4%)
        conditional_frames_probs={0: 0.333, 1: 0.333, 2: 0.334},

        # Optional: set conditional_frame_timestep for better control
        conditional_frame_timestep=-1.0,  # Default -1 means not effective
        # Keep the default conditioning strategy
        conditioning_strategy="frame_replace",
        denoise_replace_gt_frames=True,
    ),
)

# Training configuration
_lora_trainer = dict(
    logging_iter=100,
    max_iter=10000,
    callbacks=dict(
        heart_beat=dict(save_s3=False),
        iter_speed=dict(hit_thres=1000, save_s3=False),
        device_monitor=dict(save_s3=False),
        every_n_sample_reg=dict(every_n=1000, save_s3=False),
        every_n_sample_ema=dict(every_n=1000, save_s3=False),
        wandb=dict(save_s3=False),
        wandb_10x=dict(save_s3=False),
        dataloader_speed=dict(save_s3=False),
    ),
)

# Optimizer configuration
_lora_optimizer = dict(
    lr=0.0001,
    weight_decay=0.001,
)

# Scheduler configuration
_lora_scheduler = dict(
    f_max=[0.5],
    f_min=[0.2],
    warm_up_steps=[2_000],
    cycle_lengths=[100000],
)

# Checkpoint configuration
_lora_checkpoint_base = dict(
    load_path=get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri),
    load_from_object_store=dict(enabled=False),
    save_to_object_store=dict(enabled=False),
    save_iter=1000,  # Save checkpoint every 1000 iterations
)

# Model parallel configuration
_lora_model_parallel = dict(
    context_parallel_size=1,
)

# Complete experiment configurations
predict2_lora_training_2b_cosmos_sports_assets_txt = dict(
    defaults=_lora_defaults,
    job=dict(
        project="cosmos_predict_v2p5",
        group="lora",
        name="2b_cosmos_sports_assets_lora",
    ),
    dataloader_train=dataloader_train_cosmos_sports_assets_lora_txt,
    checkpoint=_lora_checkpoint_base,
    optimizer=_lora_optimizer,
    scheduler=_lora_scheduler,
    trainer=_lora_trainer,
    model=_lora_model_config,
    model_parallel=_lora_model_parallel,
)

predict2_lora_training_2b_cosmos_sports_assets_json_rank32 = dict(
    defaults=_lora_defaults,
    job=dict(
        project="cosmos_predict_v2p5",
        group="lora",
        name="2b_cosmos_sports_assets_json_lora",
    ),
    dataloader_train=dataloader_train_cosmos_sports_assets_lora_json,
    checkpoint=_lora_checkpoint_base,
    optimizer=_lora_optimizer,
    scheduler=_lora_scheduler,
    trainer=_lora_trainer,
    model=_lora_model_config,
    model_parallel=_lora_model_parallel,
)
```

### 2.2 Training

Run the LoRA post-training using one of the following configurations:

#### Using Text Caption Format

```bash
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
  --config=projects/cosmos/predict2/configs/video2world/config.py -- \
  experiment=predict2_lora_training_2b_cosmos_sports_assets_txt
```

#### Using JSON Caption Format with Long Prompts

```bash
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
  --config=projects/cosmos/predict2/configs/video2world/config.py -- \
  experiment=predict2_lora_training_2b_cosmos_sports_assets_json_rank32
```

#### Disabling W&B Logging

Add `job.wandb_mode=disabled` to disable wandb:

```bash
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
  --config=projects/cosmos/predict2/configs/video2world/config.py -- \
  experiment=predict2_lora_training_2b_cosmos_sports_assets_txt \
  job.wandb_mode=disabled
```

Checkpoints are saved to:

- Text format: `${IMAGINAIRE_OUTPUT_ROOT}/cosmos_predict_v2p5/lora/2b_cosmos_sports_assets_lora/checkpoints`
- JSON format: `${IMAGINAIRE_OUTPUT_ROOT}/cosmos_predict_v2p5/lora/2b_cosmos_sports_assets_json_lora/checkpoints`

**Note**: By default, `IMAGINAIRE_OUTPUT_ROOT` is `/tmp/imaginaire4-output`. We strongly recommend setting `IMAGINAIRE_OUTPUT_ROOT` to a location with sufficient storage space for your checkpoints.

## Checkpointing

Training uses two checkpoint formats:

### 1. Distributed Checkpoint (DCP) Format

**Primary format for training checkpoints.**

- **Structure**: Multi-file directory with sharded model weights
- **Used for**: Saving checkpoints during training, resuming training
- **Advantages**:
  - Efficient parallel I/O for multi-GPU training
  - Supports FSDP (Fully Sharded Data Parallel)
  - Optimized for distributed workloads

**Example directory structure:**

```
checkpoints/
├── iter_{NUMBER}/
│   ├── model/
│   │   ├── .metadata
│   │   └── __0_0.distcp
│   ├── optim/
│   ├── scheduler/
│   └── trainer/
└── latest_checkpoint.txt
```

### 2. Consolidated PyTorch (.pt) Format

**Single-file format for inference and distribution.**

- **Structure**: Single `.pt` file containing the complete model state
- **Used for**: Inference, model sharing, initial post-training
- **Advantages**:
  - Easy to distribute and version control
  - Standard PyTorch format
  - Simpler for single-GPU workflows

## 3. Inference with LoRA Post-trained checkpoint

### 3.1 Converting DCP Checkpoint to Consolidated PyTorch Format

Since the checkpoints are saved in DCP format during training, you need to convert them to consolidated PyTorch format (.pt) for inference. Use the `convert_distcp_to_pt.py` script:

```bash
# For text format checkpoint
CHECKPOINTS_DIR=${IMAGINAIRE_OUTPUT_ROOT:-/tmp/imaginaire4-output}/cosmos_predict_v2p5/lora/2b_cosmos_sports_assets_lora/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)
CHECKPOINT_DIR=$CHECKPOINTS_DIR/$CHECKPOINT_ITER

# Convert DCP checkpoint to PyTorch format
python scripts/convert_distcp_to_pt.py $CHECKPOINT_DIR/model $CHECKPOINT_DIR
```

This conversion will create three files:

- `model.pt`: Full checkpoint containing both regular and EMA weights
- `model_ema_fp32.pt`: EMA weights only in float32 precision
- `model_ema_bf16.pt`: EMA weights only in bfloat16 precision (recommended for inference)

### 3.2 Running Inference

After converting the checkpoint, you can run inference with your post-trained model using the command-line interface.

The model can be used for any generation mode. Simply use the appropriate JSON configuration with the corresponding experiment:

#### Text2World Generation

```bash
# Using Text format checkpoint
torchrun --nproc_per_node=8 examples/inference.py \
  -i assets/sports_text2world_prompts.json \
  -o outputs/sports_text2world \
  --checkpoint-path $CHECKPOINT_DIR/model_ema_bf16.pt \
  --experiment predict2_lora_training_2b_cosmos_sports_assets_txt

# Using JSON format checkpoint
torchrun --nproc_per_node=8 examples/inference.py \
  -i assets/sports_text2world_prompts.json \
  -o outputs/sports_text2world \
  --checkpoint-path $JSON_CHECKPOINT_DIR/model_ema_bf16.pt \
  --experiment predict2_lora_training_2b_cosmos_sports_assets_json_rank32
```

#### Image2World Generation

```bash
torchrun --nproc_per_node=8 examples/inference.py \
  -i assets/sports_image2world_inputs.json \
  -o outputs/sports_image2world \
  --checkpoint-path $CHECKPOINT_DIR/model_ema_bf16.pt \
  --experiment predict2_lora_training_2b_cosmos_sports_assets_txt
```

#### Video2World Generation

```bash
torchrun --nproc_per_node=8 examples/inference.py \
  -i assets/sports_video2world_inputs.json \
  -o outputs/sports_video2world \
  --checkpoint-path $CHECKPOINT_DIR/model_ema_bf16.pt \
  --experiment predict2_lora_training_2b_cosmos_sports_assets_txt
```

The model automatically detects the generation mode based on the input:

- Provide text only → Text2World generation
- Provide 1 image frame → Image2World generation
- Provide 2+ video frames → Video2World generation

Generated videos will be saved to the output directory.

### Example Prompts for Soccer Video Generation

#### Image2World/Video2World Generation Example

```json
{
  "inference_type": "image2world",
  "name": "soccer_action_sequence",
  "prompt": "A soccer player in a red and black uniform dribbles the ball past an opponent in an orange and white uniform. The player in red sprints towards the goal, evading the defender. As he approaches the goalpost, the goalkeeper dives to make a save but fails to stop the ball from entering the net. The camera follows the ball as it flies into the goal, capturing the excitement of the moment.",
  "input_path": "first_frame.mp4",
  "seed": 0,
  "guidance": 3,
  "num_output_frames": 93
}
```

## Evaluation and Results

### Comparison: Base Model vs Post-Trained Model

The LoRA post-training significantly improves the quality and realism of generated soccer videos. Below is a comparison of videos generated by the base model versus the post-trained model:

| Sample | Base Model | Post-Trained Model |
|--------|------------|-------------------|
| **Sample 1** | <video width="320" controls autoplay loop muted><source src="assets/base/0.mp4" type="video/mp4"></video> | <video width="320" controls autoplay loop muted><source src="assets/post_trained/0.mp4" type="video/mp4"></video> |
| **Sample 2** | <video width="320" controls autoplay loop muted><source src="assets/base/12.mp4" type="video/mp4"></video> | <video width="320" controls autoplay loop muted><source src="assets/post_trained/12.mp4" type="video/mp4"></video> |
| **Sample 3** | <video width="320" controls autoplay loop muted><source src="assets/base/5.mp4" type="video/mp4"></video> | <video width="320" controls autoplay loop muted><source src="assets/post_trained/5.mp4" type="video/mp4"></video> |

### Key Improvements After Post-Training

**The post-training experiment demonstrated model improvements with limited data, despite not solving all physics artifacts.**

The post-trained model demonstrates substantial enhancements in several critical areas:

#### 1. **Rule Coherence and Field Semantics**

The post-trained model shows significantly better understanding of soccer rules and field layout:

- Players stay within field boundaries and respect offside lines
- Goal areas and penalty boxes are rendered with proper dimensions
- Ball physics follow realistic trajectories during passes and shots

#### 2. **Identity and Team Preservation**

Team identities and player characteristics are better maintained throughout the generated sequences:

- Consistent jersey colors and numbers across all frames
- Individual player features remain stable during complex movements
- Team formations and tactical positioning are more realistic

#### 3. **Reduced Broadcast Artifacts**

The post-trained model produces cleaner, broadcast-quality videos:

- Minimized motion blur during fast-paced action
- Reduced ghosting effects around players and the ball
- Cleaner rendering of stadium elements and crowd backgrounds
- Improved temporal consistency across frame sequences

#### 4. **Sport-Specific Motion Dynamics**

The model better captures soccer-specific movements:

- Realistic dribbling patterns and ball control
- Natural goalkeeper diving and saving motions
- Accurate representation of tackles, passes, and shots
- Proper player acceleration and deceleration patterns

These improvements make the post-trained model particularly suitable for:

- Training computer vision systems for sports analytics
- Generating synthetic data for referee training
- Creating realistic game simulations for tactical analysis
- Producing content for sports broadcasting and entertainment

For more inference options and advanced usage, see the Cosmos Predict 2 [inference documentation](https://github.com/nvidia-cosmos/cosmos-predict2/blob/main/docs/inference.md).

---

## Document Information

**Publication Date:** December 1, 2025

### Citation

If you use this recipe or reference this work, please cite it as:

```bibtex
@misc{cosmos_cookbook_lora_sports_2025,
  title={LoRA Post-training for Sports Video Generation},
  author={Ali, Arslan},
  organization={NVIDIA},
  year={2025},
  month={December},
  howpublished={\url{https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/predict2_5/sports/post_training.html}},
  note={NVIDIA Cosmos Cookbook}
}
```

**Suggested text citation:**

> Arslan Ali (2025). LoRA Post-training for Sports Video Generation. In *NVIDIA Cosmos Cookbook*. NVIDIA. Accessible at <https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/predict2_5/sports/post_training.html>
