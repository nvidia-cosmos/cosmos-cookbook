# 3D Autonomous Vehicle Grounding Post-Training with Cosmos Reason 1 & 2

> **Authors:** [Amol Fasale](https://www.linkedin.com/in/amolfasale/) • [Jingyi Jin](https://www.linkedin.com/in/jingyi-jin/)
> **Organization:** NVIDIA

| **Model** | **Workload** | **Use Case** |
| --- | --- | --- |
| [Cosmos Reason 1](https://github.com/nvidia-cosmos/cosmos-reason1) | Post-training | 3D vehicle grounding in autonomous driving scenarios |
| [Cosmos Reason 2](https://github.com/nvidia-cosmos/cosmos-reason2) | Post-training | 3D vehicle grounding in autonomous driving scenarios |

## Overview

3D vehicle grounding is a computer vision task that enables autonomous vehicles to detect and precisely localize surrounding vehicles in three-dimensional space from camera images. Unlike traditional 2D object detection, which only identifies objects within the image plane, 3D grounding provides complete spatial information including each vehicle's position (x, y, z coordinates), dimensions (length, width, height), and orientation (roll, pitch, yaw angles) in the real world. This comprehensive 3D understanding is essential for autonomous driving systems, enabling accurate path planning, collision avoidance, and safe navigation by allowing vehicles to reason about spatial relationships and predict future trajectories of surrounding objects.


### Expected Input and Output After Post-Training

The following example demonstrates the expected input and output of the 3D vehicle grounding task after completing the post-training exercise. This shows what a fine-tuned model can achieve: taking an autonomous vehicle camera image as input and producing accurate 3D bounding box coordinates as output.

<p align="center">
  <img src="assets/expected_input_output.png" alt="Expected Input and Output Flow for 3D AV Grounding" width="700"/>
</p>
<p align="center">
  <em>Figure: Input-output workflow for 3D vehicle grounding. The model takes a text prompt and camera image as input and predicts 3D bounding boxes for detected vehicles and it also shows the visualization of predicted 3d coordinates.</em>
</p>


The expected output includes:
- **annotations**: Array of detected vehicles, each with:
  - **label**: Vehicle category (car, truck, bus, etc.)
  - **bbox_3d**: Array of 9 values representing [x, y, z, x_size, y_size, z_size, roll, pitch, yaw]

**Visualization: 3D Coordinates Projected on Image**

The raw 3D bounding box coordinates from the model output are then projected back onto the 2D image plane for visualization. This projection uses the camera parameters to transform the 3D coordinates into 2D image coordinates, showing the detected vehicles with their 3D bounding box projections:

The output image demonstrates:
- **Vehicle Detection**: Multiple vehicles detected in the scene
- **3D Bounding Box Projections**: 3D bounding boxes projected onto the 2D image plane using camera intrinsics
- **Spatial Accuracy**: Accurate localization of vehicles in 3D space, visible through the projected bounding boxes

In this recipe, we demonstrate how to fine-tune both [Cosmos Reason 1-7B](https://build.nvidia.com/nvidia/cosmos-reason1-7b) and [Cosmos Reason 2-8B](https://build.nvidia.com/nvidia/cosmos-reason2-8b) models for 3D vehicle grounding tasks using supervised fine-tuning (SFT). Both models can be trained using two different frameworks:

- **Cosmos-RL**: An async post-training framework specialized for SFT and RLHF, supporting both Cosmos Reason 1 and Cosmos Reason 2
- **Qwen-Finetune**: A fine-tuning framework based on Qwen-VL, supporting Cosmos Reason 2

Both models learn to predict precise 3D bounding box parameters including:

- **Position**: x, y, z coordinates
- **Dimensions**: x_size, y_size, z_size
- **Orientation**: roll, pitch, yaw angles
- **Category**: Vehicle label (car, truck, bus, etc.)

This fine-tuning process adapts the general-purpose reasoning model to the specific requirements of autonomous vehicle perception, improving its accuracy in predicting 3D spatial relationships from 2D visual inputs.

## Data Curation

The training dataset is curated from the **MADS (RDS-HQ) dataset**, which contains **5,500+ autonomous vehicle video driving sequences**, each with **5 camera views** (front, front-left, front-right, rear-left, rear-right). This rich dataset provides comprehensive multi-view coverage of driving scenarios, enabling robust 3D perception training.

The MADS dataset is a comprehensive autonomous driving dataset that includes multiple data modalities and attributes:

- **Video Data**: Multi-camera video sequences capturing driving scenarios from different viewpoints
- **3D Annotations**: Detailed 3D bounding box annotations for vehicles and other objects
- **LiDAR Data**: Point cloud data providing precise 3D spatial information
- **Labels**: Object labels and semantic segmentation data
- **Camera Calibration**: Intrinsic and extrinsic camera parameters for accurate projection
- **Ego Vehicle Poses**: Vehicle position and orientation data over time
- **Scene Metadata**: Additional contextual information about driving scenarios

For this 3D vehicle grounding task, we focus on extracting frames from the video sequences along with their corresponding 3D bounding box annotations, while leveraging the camera calibration and pose data for accurate coordinate transformations.

### Data Curation Pipeline

The Cosmos Data Curation Pipeline involves 5 main steps that transform raw video sequences and annotations into a structured training dataset:

<p align="center">
  <img src="assets/data_curation_pipeline.png" alt="Data Curation Pipeline Overview" width="700"/>
</p>
<p align="center">
  <em>Figure: Cosmos Data Curation Pipeline showing the five main steps from video extraction to dataset validation.</em>
</p>

**Pipeline Overview:**

1. **Extract Frames from Videos**: Extract frames at specified intervals (typically 1 FPS or every 30 frames) to reduce duplication and manage dataset size.
2. **Extract 3D Text Annotations**: Extract corresponding 3D annotations (positions, dimensions, orientations, labels, camera parameters) for each extracted frame.
3. **Transformation to Camera Coordinates and Coordinate System Conversion**: Transform from world to camera coordinates and convert coordinate convention from FLU to RDF.
4. **Filter Objects**: Apply filtering criteria including distance (>100m), field of view, occlusion, and depth to remove problematic annotations.
5. **Validate the Extracted Dataset**: Project 3D bounding boxes onto 2D image plane for visual verification and quality control.

#### 1. Extract Frames from Videos

The script extracts individual frames from video sequences at specified intervals (typically 1 FPS or every 30 frames). Each extracted frame is saved as a high-resolution image file, preserving the visual information needed for 3D grounding tasks.

#### 2. Extract 3D Text Annotations

For each extracted frame, the script extracts corresponding 3D annotations from the original dataset. These annotations include:
- Vehicle positions in 3D space
- Bounding box dimensions
- Orientation angles (roll, pitch, yaw)
- Vehicle category labels
- Camera parameters (intrinsic and extrinsic): focal lengths (fx, fy), principal point (cx, cy), camera pose, and field of view information

#### 3. Transformation to Camera Coordinates and Coordinate System Conversion

The script performs a two-step transformation process to convert 3D bounding boxes from **FLU world coordinates** to **RDF camera coordinates**:

**Step 3a: Transformation from World to Camera Coordinates**

The script first transforms 3D bounding boxes from **world coordinates** to **camera coordinates** (both in FLU convention). This transformation is essential because:

- **World Coordinates**: Represent the vehicle's position relative to a fixed world reference frame (typically the ego vehicle's initial position)
- **Camera Coordinates**: Represent the vehicle's position relative to the camera's coordinate frame, which is necessary for accurate 2D projection and vision-based processing

**Transformation Process**:

1. **Extract Camera Extrinsics**: The script uses the camera pose information (rotation and translation) from the dataset metadata to construct the transformation matrix from world to camera coordinates

2. **Apply Rigid Transformation**: For each 3D bounding box center point, the transformation is applied:
   ```
   P_camera_FLU = R * P_world_FLU + t
   ```
   Where:
   - `P_world_FLU`: 3D point in world coordinates (FLU convention)
   - `R`: Rotation matrix from world to camera frame
   - `t`: Translation vector from world to camera frame
   - `P_camera_FLU`: 3D point in camera coordinates (FLU convention)

3. **Preserve Bounding Box Dimensions**: The bounding box dimensions (x_size, y_size, z_size) remain unchanged during this transformation, as they represent the object's physical size

4. **Update Orientation**: The orientation angles (roll, pitch, yaw) are adjusted to account for the camera's orientation relative to the world frame

**Step 3b: Coordinate Convention Conversion (FLU → RDF)**

After transforming to camera coordinates, the script converts coordinates from **FLU (Front-Left-Up)** convention to **RDF (Right-Down-Forward)** convention, which is the standard coordinate system used in computer vision and OpenCV. This conversion ensures compatibility with standard vision processing pipelines.

The transformation matrix for FLU to RDF conversion:

```
FLU Coordinate System (Forward-Left-Up)        RDF Coordinate System (Right-Down-Forward)
         Z (Up)                                       Y (Right)
         |                                            |
         |                                            |
         |                                            |
         +----> X (Forward)                           +----> Z (Forward)
        /                                            /
       /                                            /
      Y (Left)                                    X (Down)
      
Transformation Mapping:
┌─────────┬──────────────┬──────────────┐
│ FLU Axis│ Direction    │ RDF Axis     │
├─────────┼──────────────┼──────────────┤
│ +X (→)  │ Forward      │ +Y (→)       │
│ +Y (←)  │ Left         │ -Z (↓)       │
│ +Z (↑)  │ Up           │ +X (→)       │
└─────────┴──────────────┴──────────────┘

Example: A point at (1, 2, 3) in FLU camera coordinates becomes (3, 1, -2) in RDF camera coordinates
  FLU Camera: (x=1, y=2, z=3)  →  RDF Camera: (x=3, y=1, z=-2)
```

**Why This Combined Transformation Matters**:
- Camera coordinates enable direct projection to 2D image plane using camera intrinsics
- Simplifies distance calculations and FOV filtering
- Provides a consistent reference frame for multi-camera scenarios
- Essential for accurate 3D to 2D projection validation
- Ensures compatibility with standard computer vision pipelines (RDF convention)

#### 4. Filter Objects

To ensure high-quality training data, the script applies several filtering criteria to remove problematic annotations:

- **Distance Filtering**: Removes objects that are too far away (at distances greater than 100 m (>100 m)), as these are difficult to accurately annotate and provide limited training value
- **Field of View (FOV) Filtering**: Removes objects outside the camera's field of view, ensuring all annotations correspond to visible objects
- **Occlusion Filtering**: Removes objects that are fully overlapped by other objects, as these cannot be reliably annotated
- **Depth Filtering**: Removes objects that are behind closer objects, prioritizing annotations for objects that are clearly visible

These filters ensure that the training dataset contains only high-quality, reliable annotations that contribute to effective model learning.

#### 5. Validate the Extracted Dataset by Projecting Annotations on Image Plane

After extracting frames and 3D annotations, it's crucial to validate that the extracted data is accurate and correctly formatted. This validation step projects the 3D bounding box annotations onto the 2D image plane, allowing visual verification that the annotations align correctly with the objects visible in the images.

**Purpose of Validation:**
- **Verify Annotation Accuracy**: Ensures that 3D annotations correctly correspond to objects visible in the images
- **Check Coordinate Transformations**: Validates that the coordinate system conversions (FLU to RDF, world to camera) were performed correctly
- **Visual Quality Control**: Enables visual inspection to identify any misaligned or incorrect annotations
- **Compare Ground Truth**: Allows comparison between ground truth annotations and model predictions

**Validation Process:**
1. **Load Scene Data**: The script loads camera models, camera poses, ego poses, and scene metadata from the original dataset
2. **Load Extracted Annotations**: Reads the 3D bounding box annotations from the extracted JSON files
3. **Project 3D to 2D**: For each 3D bounding box:
   - Extracts the 8 corners of the 3D bounding box
   - Uses camera intrinsics (fx, fy, cx, cy) to project each 3D corner point onto the 2D image plane
   - Converts 3D camera coordinates to 2D pixel coordinates
4. **Draw Projections**: Draws the projected bounding boxes on the images by connecting the projected corner points, creating annotated visualization images
5. **Visual Inspection**: The annotated images allow you to verify that:
   - Bounding boxes align correctly with vehicles in the images
   - Projections are geometrically accurate
   - No annotations are misaligned or incorrectly positioned

This validation step ensures data quality before proceeding to training, helping identify and correct any issues in the data curation pipeline.

### Curation Script Usage

**Script:** `local_extract_frames.py`

This script extracts frames from video sequences and corresponding 3D annotations from the MADS dataset. It performs all data curation steps including coordinate system conversion (FLU to RDF), transformation from world to camera coordinates, and applies filtering criteria (distance, FOV, occlusion, depth).

```bash
python imaginaire/auxiliary/world_scenario/scripts/local_extract_frames.py \
    --sequence-ids-file <sequence_ids_file> \
    --s3-input-base-path <original_av_video_dataset_directory> \
    --s3-input-profile <s3_profile> \
    --extract-frames \
    --skip-frames 30 \
    --output-dir <output_extracted_images_directory>
```

**Key Parameters:**
- `--sequence-ids-file`: Text file containing sequence IDs to process (one per line)
- `--s3-input-base-path`: S3 path to the MADS dataset video sequences
- `--s3-input-profile`: AWS profile for S3 access
- `--extract-frames`: Flag to enable frame extraction
- `--skip-frames`: Number of frames to skip between extractions (30 = extract every 30th frame, ~1 FPS)
- `--output-dir`: Output directory for extracted frames and annotations

### Validation Script Usage

**Script:** `local_project_annotations.py`

This script projects 3D bounding box annotations onto the 2D image plane for visual verification. It generates annotated images showing projected 3D boxes overlaid on original images, enabling validation of annotation accuracy and coordinate transformations.

```bash
python imaginaire/auxiliary/world_scenario/scripts/local_project_annotations.py \
    --sequence-ids-file <sequence_ids_file> \
    --images-dir <gt_images_directory> \
    --text-dir <inferred_annotations_directory> \
    --input-dir <original_av_video_dataset_directory> \
    --output-dir <output_annotated_images_directory>
```

**Key Parameters:**
- `--sequence-ids-file`: Text file containing sequence IDs to validate
- `--images-dir`: Directory containing ground truth images
- `--text-dir`: Directory containing inferred 3D annotation JSON files
- `--input-dir`: Original s3 or local dataset directory with camera models and metadata
- `--output-dir`: Output directory for annotated visualization images

For the first iteration, we selected **~700 unique sequences** from the MADS dataset. After curation, this selection yields approximately **~80k frames/annotations**, providing a substantial training set for 3D vehicle grounding tasks.

## Dataset

The training dataset consists of autonomous vehicle camera images with corresponding 3D vehicle bounding box annotations. The dataset format uses a structured directory layout with separate metadata, conversation, and media files.

<p align="left">
  The images below illustrate the curated autonomous vehicle camera frames (left) and their corresponding 3D text annotations (right) as used in the benchmarking and training dataset. These paired examples highlight the link between real-world visual input and structured annotation data for 3D vehicle grounding.
</p>
<p align="center">
  <img src="assets/training_images_overview.png" alt="Training Images Overview" width="450" style="display:inline-block; vertical-align:middle; margin-right:16px;"/>
  <img src="assets/training_annotations_overview.png" alt="Training Text Annotations Overview" width="450" style="display:inline-block; vertical-align:middle; margin-right:16px;"/>
  <img src="assets/training_images_overlay_overview.png" alt="Training Images Overlay Overview" width="450" style="display:inline-block; vertical-align:middle;"/>
</p>
<p align="center">
  <em>Figure: Example (left) of curated AV camera frames, (center) their corresponding 3D text annotations, and (right) overlay visualization extracted for benchmarking and training.</em>
</p>


### Dataset Splits

The curated dataset is divided into training and evaluation splits to enable supervised fine-tuning and performance assessment. The training split contains the majority of sequences for model learning, while the evaluation split provides a held-out set for unbiased performance evaluation.

| **Split** | **Sequences** | **Frames/Annotations** | **Purpose** |
|-----------|---------------|------------------------|-------------|
| **Evaluation** | 10 AV Multi-view Sequences | ~1.3k | For model performance assessment and validation |
| **Training** | 700 AV Multi-view Sequences | ~80k | For supervised fine-tuning of models |

### Dataset Structure

```
dataset/
├── meta.json                    # Index file linking conversations to media
├── images/                      # Directory containing all image files
│   ├── frame_000000.jpg
│   ├── frame_000030.jpg
│   └── ...
└── text/                        # Directory containing annotation files
    ├── frame_000000.json
    ├── frame_000030.json
    └── ...
```

### meta.json Format

The `meta.json` file contains a list of entries, each linking a conversation to its media:

```json
[
  {
    "id": "frame_000000",
    "media": "images/frame_000000.jpg",
    "conversations": "text/frame_000000.json"
  }
]
```

### Annotation Format

Each annotation file contains 3D bounding box data for vehicles in the corresponding image, along with camera parameters:

```json
{
  "frame_id": 0,
  "camera": "camera_cross_left_120fov",
  "camera_params": {
    "fx": 358.99285463551877,
    "fy": 540.7505167122987,
    "cx": 636.54638671875,
    "cy": 365.6412048339844
  },
  "annotations": [
    {
      "label": "car",
      "bbox_3d": [
        -8.962420703794697,    // x
        -0.15224466668225012,  // y
        18.004403863124224,    // z
        5.301581382751465,      // x_size
        2.187917470932007,     // y_size
        2.0336594581604004,    // z_size
        0.005443232293758353,  // roll
        0.6511561857044312,    // pitch
        -0.009069517922754287  // yaw
      ]
    }
  ]
}
```

The `camera_params` section contains:
- **fx, fy**: Focal lengths in pixels (horizontal and vertical)
- **cx, cy**: Principal point coordinates (optical center) in pixels

### Conversation Format

The conversation files are automatically generated from annotations and follow this structure:

```json
[
  [
    {
      "role": "system",
      "content": [{"type": "text", "text": "You are an expert AI assistant for 3D grounding. Your task is to accurately generate the 3D vehicle coordinates in the image, based on the user's input."}]
    },
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "image_0"},
        {"type": "text", "text": "Find all vehicles in this image. For each vehicle, provide its 3D bounding box coordinates including x, y, z, x_size, y_size, z_size, roll, pitch, yaw and the label of the vehicle. The output format required is JSON: `[{\"bbox_3d\":[x, y, z, x_size, y_size, z_size, roll, pitch, yaw],\"label\":\"category\"}]`."}
      ]
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "text": "[{\"bbox_3d\":[-8.962420703794697, -0.15224466668225012, 18.004403863124224, 5.301581382751465, 2.187917470932007, 2.0336594581604004, 0.005443232293758353, 0.6511561857044312, -0.009069517922754287], \"label\": \"car\"}]"}]
    }
  ]
]
```

## Zero-Shot Evaluation

Before fine-tuning, you can evaluate the base model's zero-shot performance on 3D grounding tasks. This provides a baseline to compare against post-training improvements.

The evaluation uses a prompt that asks the model to identify vehicles and provide their 3D bounding box coordinates:

???+ code "Prompt for 3D AV Grounding"
    ```yaml
    --8<-- "docs/recipes/post_training/reason1/av_3d_grounding/assets/3d_av_grounding.yaml"
    ```

To run zero-shot evaluation:

**Cosmos Reason 1:**
```bash
# From cosmos-reason1 root directory
python scripts/inference_local.py \
    --images <eval_images_dir> \
    --output <predictions_dir> \
    --prompt prompts/3d_av_grounding.yaml \
    --model nvidia/Cosmos-Reason1-7B
```

**Cosmos Reason 2:**
```bash
# From cosmos-reason2 root directory
python scripts/inference_local.py \
    --images <eval_images_dir> \
    --output <predictions_dir> \
    --prompt prompts/3d_av_grounding.yaml \
    --model nvidia/Cosmos-Reason2-8B
```

## Post-Training Frameworks

This recipe supports two different training frameworks, each with its own advantages:

### Cosmos-RL Framework

**Cosmos-RL** is an async post-training framework specialized for Supervised Fine-Tuning (SFT) and Reinforcement Learning with Human Feedback (RLHF). It prioritizes performance, scalability, and fault tolerance, making it ideal for large-scale training.

**Supported Models:**
- Cosmos Reason 1-7B
- Cosmos Reason 2-8B

**Advantages:**
- High-performance async training pipeline
- Built-in fault tolerance and checkpointing
- Optimized for multi-node distributed training
- Native support for Cosmos Reason models

### Qwen-Finetune Framework

**Qwen-Finetune** is a fine-tuning framework based on Qwen-VL, providing a flexible and easy-to-use interface for training vision-language models.

**Supported Models:**
- Cosmos Reason 2-8B

**Advantages:**
- Simple command-line interface
- DeepSpeed ZeRO optimization support
- Flexible dataset configuration
- Easy integration with HuggingFace models

## Post-Training Setup

### Prerequisites

1. **Repository Setup**: 
   - For Cosmos-RL: Clone and set up the [Cosmos Reason 1](https://github.com/nvidia-cosmos/cosmos-reason1) or [Cosmos Reason 2](https://github.com/nvidia-cosmos/cosmos-reason2) repository
   - For Qwen-Finetune: Clone and set up the [Qwen-VL-Finetune](https://github.com/QwenLM/Qwen3-VL) repository

2. **Environment Setup**: 
   - For Cosmos-RL: Follow the [main post-training guide](https://github.com/nvidia-cosmos/cosmos-reason1/tree/main/examples/post_training) for environment setup
   - For Qwen-Finetune: Follow the Qwen-VL-Finetune setup instructions

3. **Dataset Preparation**: Prepare your dataset in the format described above. **Cosmos-RL** uses the standard format with `meta.json`, `images/`, and `text/` directories. **Qwen-Finetune** uses a simplified, single-file format called `annotations.json` (see Qwen-Finetune Dataset Format below).

## Training with Cosmos-RL Framework

### Quick Start - Cosmos Reason 1

Navigate to the post-training example directory:

```bash
cd examples/post_training_3d_grounding
source ../post_training/.venv/bin/activate
```

Run supervised fine-tuning:

```bash
cosmos-rl --config configs/av_grounding.sft.toml scripts/av_grounding_dataloader.py
```

The default configuration uses `dataset/train_07` for training. Checkpoint path is shown in logs: `./outputs/sft-07/TIMESTAMP/safetensors/final`

### Quick Start - Cosmos Reason 2

Navigate to the cosmos-rl example directory:

```bash
cd examples/cosmos_rl
```

Run supervised fine-tuning:

```bash
cosmos-rl --config configs/av_grounding.sft.toml scripts/av_grounding_dataloader.py
```

### Cosmos-RL Configuration

The training configuration is specified in `configs/av_grounding.sft.toml`. Here are the key parameters:

???+ code "Cosmos Reason 1 Configuration"

    ```toml    
    --8<-- "assets/cr1_av_grounding.sft.toml"
    ```

???+ code "Cosmos Reason 2 Configuration"

    ```toml
    --8<-- "assets/cr2_av_grounding.sft.toml"
    ```

### Key Cosmos-RL Configuration Parameters

**Cosmos Reason 1:**
- **Model**: `nvidia/Cosmos-Reason1-7B` (7B parameter model)
- **Epochs**: 5 epochs
- **Batch Size**: 32 per replica
- **Learning Rate**: Default from config
- **Memory Optimization**: FSDP offloading and gradient checkpointing enabled

**Cosmos Reason 2:**
- **Model**: `nvidia/Cosmos-Reason2-8B` (8B parameter model)
- **Epochs**: 2 epochs
- **Batch Size**: 16 per replica
- **Learning Rate**: 2e-7
- **Memory Optimization**: Gradient checkpointing enabled

**Common Settings:**
- **Training Type**: Supervised Fine-Tuning (SFT)
- **Parallelism**: Data parallelism with shard size of 8
- **Dataset**: Points to `dataset/train_07` with `meta.json` metadata file
- **Vision Settings**: 
  - FPS: 1 (for video inputs)
  - Max pixels: 40,960 per frame
- **Logging**: Console and Weights & Biases (wandb) integration
- **Checkpointing**: Async checkpointing every 50 steps

#### Monitoring Cosmos-RL Training

Training progress is logged to:
- **Console**: Real-time training metrics
- **Weights & Biases**: 
  - Cosmos Reason 1: Project `cosmos_reason1`, experiment `post_training_capabilities/sft_v7_cr1`
  - Cosmos Reason 2: Project `cosmos_reason2`, experiment `post_training_capabilities/sft_v7_cr2_8b`

Key metrics to monitor:
- Training loss
- Learning rate schedule
- GPU memory usage
- Checkpoint save status

**Training Progress Visualization:**

The following graphs show the training progress for both Cosmos Reason 1 and Cosmos Reason 2 models:

**Cosmos Reason 1 Training:**
![Cosmos Reason 1 Training Progress](assets/post_training_cr1.png)

**Cosmos Reason 2 Training:**
![Cosmos Reason 2 Training Progress](assets/post_training_cr2.png)

## Training with Qwen-Finetune Framework

### Quick Start - Cosmos Reason 2

Navigate to your Qwen-VL-Finetune repository directory (the clone root, often named `qwen-vl-finetune` or `Qwen3-VL`):

```bash
cd qwen-vl-finetune   # or your clone path
```

Copy the training script from this recipe into the repo: save the script shown below as `scripts/sft_av_3d_grounding_cr2_8b.sh` in your clone (or copy `assets/qwen_finetune_script.sh` from this recipe into that path). Then run:

```bash
bash scripts/sft_av_3d_grounding_cr2_8b.sh
```

### Qwen-Finetune Configuration

The training script content is provided below. Save it as `scripts/sft_av_3d_grounding_cr2_8b.sh` in your Qwen-VL-Finetune repository:

???+ code "Qwen-Finetune Training Script"

    ```bash
    --8<-- "recipes/post_training/reason1/av_3d_grounding/assets/qwen_finetune_script.sh"    
    ```

### Key Qwen-Finetune Configuration Parameters

- **Model**: `nvidia/Cosmos-Reason2-8B` (8B parameter model)
- **Training Type**: Supervised Fine-Tuning (SFT)
- **Epochs**: 2 epochs
- **Batch Size**: 2 per device with gradient accumulation of 8 (effective batch size: 16)
- **Learning Rate**: 2e-7
- **Optimization**: DeepSpeed ZeRO-3 for memory efficiency
- **Dataset**: Uses `av3dgrounding_train` and `av3dgrounding_eval` datasets
- **Tuning**: Tunes vision, MLP, and LLM components
- **Precision**: BF16 mixed precision training
- **Logging**: Weights & Biases integration

**Training Progress Visualization:**

The following graph shows the training progress for Cosmos Reason 2 using the Qwen-Finetune framework:

**Qwen-Finetune Training:**
![Qwen-Finetune Training Progress](assets/post_training_qwen.png)

### Qwen-Finetune Dataset Format

Qwen-Finetune uses the `annotations.json` format, a simplified single-file format specific to that framework. The standard format for Cosmos-RL (described earlier in this recipe) uses `meta.json` with separate `images/` and `text/` directories. The `annotations.json` format is shown below:

```json
[
  {
    "image": "images/frame_000000.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nFind all vehicles in this image. For each vehicle, provide its 3D bounding box coordinates including x, y, z, x_size, y_size, z_size, roll, pitch, yaw and the label of the vehicle. The output format required is JSON: `[{\"bbox_3d\":[x, y, z, x_size, y_size, z_size, roll, pitch, yaw],\"label\":\"category\"}]`."
      },
      {
        "from": "gpt",
        "value": "[{\"bbox_3d\":[-8.962420703794697, -0.15224466668225012, 18.004403863124224, 5.301581382751465, 2.187917470932007, 2.0336594581604004, 0.005443232293758353, 0.6511561857044312, -0.009069517922754287], \"label\": \"car\"}]"
      }
    ]
  }
]
```

### Monitoring Qwen-Finetune Training

Training progress is logged to:
- **Console**: Real-time training metrics
- **Weights & Biases**: Project `qwen3-vl_cosmos_reason2`, run name `av_3d_grounding_sft_qwen3_vl_cr2_8b`

Key metrics to monitor:
- Training loss
- Evaluation metrics (if eval dataset is provided)
- Learning rate schedule
- GPU memory usage
- Checkpoint save status

## Evaluation

After training, evaluate the fine-tuned model on the evaluation dataset:

### Cosmos Reason 1 Evaluation

```bash
# From cosmos-reason1 root directory
python scripts/inference_local.py \
    --images <eval_images_dir> \
    --output <predictions_dir> \
    --prompt prompts/3d_av_grounding.yaml \
    --model <path_to_finetuned_checkpoint>
```

### Cosmos Reason 2 Evaluation

```bash
# From cosmos-reason2 root directory
python scripts/inference_local.py \
    --images <eval_images_dir> \
    --output <predictions_dir> \
    --prompt prompts/3d_av_grounding.yaml \
    --model <path_to_finetuned_checkpoint>
```

The evaluation script generates JSON files containing predicted 3D bounding boxes for each image. Compare these predictions against ground truth annotations using the following evaluation metrics:

### Evaluation Metrics

#### Average Precision (AP) Metrics

**AP2D (Average Precision in 2D)**: Measures detection accuracy in the 2D image plane using axis-aligned IoU.

- **2D Axis-Aligned IoU**: Projects 3D bounding boxes onto the 2D image plane and computes the Intersection over Union of the axis-aligned 2D bounding boxes
- **Calculation**: For each predicted vehicle, project its 3D bounding box corners to 2D image coordinates, compute the axis-aligned bounding rectangle, and compare with ground truth 2D projections
- **AP2D Score**: Computes Average Precision across different IoU thresholds (typically 0.5, 0.75) using the standard COCO evaluation protocol
- **Formula**: 
  ```
  IoU_2D = Area(Intersection) / Area(Union)
  AP2D = Average Precision over all IoU thresholds
  ```

**AP3D (Average Precision in 3D)**: Measures detection accuracy in 3D space using axis-aligned IoU.

- **3D Axis-Aligned IoU**: Computes the Intersection over Union of axis-aligned 3D bounding boxes (ignoring orientation)
- **Calculation**: For each predicted vehicle, compute the axis-aligned 3D bounding box (aligned with world coordinate axes) and compare with ground truth axis-aligned boxes
- **AP3D Score**: Computes Average Precision across different IoU thresholds (typically 0.25, 0.5, 0.75) in 3D space
- **Formula**:
  ```
  IoU_3D = Volume(Intersection) / Volume(Union)
  AP3D = Average Precision over all IoU thresholds
  ```

**Key Differences**:
- **AP2D** evaluates how well vehicles are detected and localized in the 2D image plane
- **AP3D** evaluates how well vehicles are localized in 3D world space
- **Axis-Aligned IoU** simplifies the computation by ignoring rotation, focusing on position and size accuracy

#### Additional Metrics

- **Position Accuracy**: Mean error in x, y, z coordinates (in meters)
- **Orientation Accuracy**: Mean error in roll, pitch, yaw angles (in degrees)
- **Detection Rate**: Percentage of vehicles correctly detected (true positives / total ground truth)
- **Precision**: Ratio of true positives to all positive predictions
- **Recall**: Ratio of true positives to all ground truth objects

## Results

The following tables compare **zero-shot** (base model) and **post-trained** performance on the held-out evaluation set (~1.3k frames). Post-training consistently improves 3D vehicle grounding over the zero-shot baseline.

### Cosmos Reason 1: Zero-Shot vs Post-Training (by steps)

| **Metric** | **Zero-Shot** | **200** | **400** | **600** | **800** | **1000** | **1390** |
|------------|---------------|--------|--------|--------|--------|---------|----------|
| **Mean IoU** | 0 | 0.030 | 0.040 | 0.048 | 0.080 | 0.091 | **0.102** |
| **IoU Accuracy %** | 0 | 0.497 | 1.249 | 1.126 | 2.779 | 3.308 | **4.104** |
| **Label Accuracy %** | 0 | 47.93 | 55.33 | 56.42 | 60.13 | 60.63 | **64.91** |
| **Average Precision 2D** | 0 | 0.046 | 0.079 | 0.114 | 0.151 | 0.178 | **0.198** |
| **Average Precision 3D** | 0 | 0.005 | 0.013 | 0.013 | 0.025 | 0.034 | **0.035** |

### Result Comparison Graphs

The following graphs show zero-shot vs post-training comparison across training steps for Cosmos Reason 1.

<p align="center">
  <img src="assets/mean_iou.png" alt="Mean IoU vs training steps" width="450" style="display:inline-block; vertical-align:top; margin:8px;"/>
  <img src="assets/iou_accuracy.png" alt="IoU Accuracy % vs training steps" width="450" style="display:inline-block; vertical-align:top; margin:8px;"/>
</p>

<p align="center">
  <img src="assets/label_accuracy.png" alt="Label Accuracy % vs training steps" width="450" style="display:inline-block; vertical-align:top; margin:8px;"/>
  <img src="assets/ap2d.png" alt="Average Precision 2D vs training steps" width="450" style="display:inline-block; vertical-align:top; margin:8px;"/>
</p>

<p align="center">
  <img src="assets/ap3d.png" alt="Average Precision 3D vs training steps" width="450" style="display:inline-block; vertical-align:top; margin:8px;"/>
</p>

### Cosmos Reason 2: Zero-Shot vs Post-Training

Cosmos Reason 2 results can be added in the same format after evaluation.

**Summary:** Post-training with the curated MADS (RDS-HQ) data substantially improves 3D vehicle grounding over zero-shot for Cosmos Reason 1: Mean IoU rises from 0 to 0.102, AP2D from 0 to 0.198, AP3D from 0 to 0.035, IoU Accuracy to 4.10%, and Label Accuracy to 64.91% at 1390 training steps. Metrics improve monotonically with more training steps.

## Framework Comparison

| **Aspect** | **Cosmos-RL** | **Qwen-Finetune** |
|------------|---------------|-------------------|
| **Supported Models** | Cosmos Reason 1 & 2 | Cosmos Reason 2 |
| **Training Type** | SFT, RLHF | SFT |
| **Distributed Training** | Built-in multi-node support | PyTorch Distributed + DeepSpeed |
| **Memory Optimization** | FSDP, gradient checkpointing | DeepSpeed ZeRO-3, gradient checkpointing |
| **Checkpointing** | Async checkpointing | Standard checkpointing |
| **Dataset Format** | `meta.json` + structured directories (standard) | `annotations.json` (Qwen-Finetune simplified) |
| **Best For** | Large-scale production training | Quick experimentation and prototyping |

## Additional Resources

- [Cosmos Reason 1 Post-Training Guide](https://github.com/nvidia-cosmos/cosmos-reason1/tree/main/examples/post_training)
- [Cosmos Reason 2 Post-Training Guide](https://github.com/nvidia-cosmos/cosmos-reason2/tree/main/examples/cosmos_rl)
- [Cosmos-RL Documentation](https://github.com/nvidia-cosmos/cosmos-rl)
- [Qwen-VL-Finetune Repository](https://github.com/QwenLM/Qwen3-VL)
- [3D AV Grounding Example - Cosmos Reason 1](https://github.com/nvidia-cosmos/cosmos-reason1/tree/main/examples/post_training_3d_grounding)
- [3D AV Grounding Example - Cosmos Reason 2](https://github.com/nvidia-cosmos/cosmos-reason2/tree/main/examples/cosmos_rl)
