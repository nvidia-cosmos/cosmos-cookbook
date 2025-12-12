# Transfer2 Multiview Generation with World Scenario Map Control with Cosmos Predict2 Multiview

> **Authors:** [Tiffany Cai](https://www.linkedin.com/in/tiffany-cai-57681211a/) • [Francesco Ferroni](https://www.linkedin.com/in/francesco-ferroni-44708137/)
> **Organization:** [NVIDIA]

## Overview

| **Model** | **Workload** | **Use Case** |
|-----------|--------------|--------------|
| Cosmos Transfer2 | Post-training | Spatially-conditioned multiview AV video generation with world scenario map control |

Generating realistic multi-camera autonomous vehicle videos requires precise spatial control to ensure consistency across viewpoints and adherence to road geometry. While Cosmos Predict 2.5 Multiview can generate plausible multiview videos from text prompts, it lacks the ability to control specific spatial layouts, lane configurations, and object placements needed for AV simulation and testing scenarios.​

This recipe demonstrates how to add spatial conditioning to Cosmos Predict 2.5 Multiview through ControlNet-based post-training. The resulting Cosmos Transfer 2.5 Multiview model accepts world scenario map representations as additional inputs, enabling precise control over road geometry, vehicle positions, and scene layouts while maintaining temporal and multi-view consistency.​

## Setup and System Requirements

Hardware:

- 8 GPU setup required for training
- Training performed at 720p resolution, 10fps

The number of GPUs (context parallel size) must be greater than or equal to the number of active views in your spec. An active view is any camera entry that supplies a `control_path` (e.g., `front_wide`, `rear_left`, etc.). The default sample spec enables seven views, so it runs on seven GPUs. If you reduce the views in your JSON spec, you can run on fewer GPUs. Adjust `--nproc_per_node` (or total world size) accordingly before running the post-training/inference commands below.

Software Dependencies:

- Cosmos framework
- Reason 1.1 7B text encoder​

Pre-trained Models:

- Cosmos Predict 2.5 Multiview (2B parameters)​
- Supports 7-view multi-camera configuration
- 29-frame context with 8 latent frame state

## Problem

Autonomous vehicle development requires generating diverse driving scenarios with precise spatial control for testing and simulation. Key challenges include:​

- **Spatial precision**: Videos must accurately reflect specific road layouts, lane configurations, and object positions defined in world scenario maps
- **Multi-view consistency**: Generated videos across 7 camera views must maintain geometric consistency
- **Scenario reproducibility**: Ability to generate specific scenarios (e.g., particular intersection layouts, vehicle trajectories) repeatedly

Standard text-to-video models cannot provide this level of spatial control, making them unsuitable for AV simulation workflows that require deterministic scene geometry.

### Zero-Shot Evaluation
Before post-training, we evaluated Cosmos Predict 2.5 Multiview's ability to follow spatial instructions through text prompts alone. The base model:​

- Generated visually plausible multiview AV videos with reasonable temporal consistency
- Struggled to accurately reproduce specific lane geometries from textual descriptions
- Could not guarantee precise object placements or maintain strict spatial relationships across views
- Showed inconsistent adherence to complex spatial constraints described in prompts

#### Evaluation Metrics:
Lane detection accuracy on generated videos showed significant deviation from ground truth layouts​
3D cuboid evaluation revealed geometric inconsistencies in object placement across viewpoints
These limitations confirmed the need for explicit spatial conditioning through visual control inputs rather than relying solely on text descriptions.​

## Data Curation

The Cosmos Transfer 2.5 Multiview checkpoint available for download was trained using a two-stage approach with distinct datasets:

#### Base Model Training: Cosmos Predict 2.5 Multiview
For Cosmos-Predict2.5-2B/auto/multiview, we curated a multi-view captioned dataset of 1.5 million clips, each containing 20-second-long scenarios with 7 synchronized cameras recording at 30FPS (front-wide, front-tele, front-left, front-right, rear-left, rear-right, rear-tele). To facilitate training with text conditioning, we generated captions at 150-frame intervals using Qwen2.5-7B-Instruct with three different lengths (short, medium, and long).​

#### ControlNet Post-Training: Cosmos Transfer 2.5 Multiview
For Cosmos-Transfer2.5-2B/auto/multiview, we used the RDS-HQ dataset (Ren et al., 2025), which consists of 140,000 20-second multi-view driving scenes and HD map metadata covering a diverse set of traffic scenarios. The post-training process introduces spatial conditioning through "world scenario maps" that provide comprehensive control over the generated videos.

#### World Scenario Map Control Signal
World scenario maps project HD maps and dynamic objects in the scene onto the seven camera views as the control input (see the below image). The world scenario map includes:​

- **Map elements**: Lane lines (with fine-grained types such as dashed line, dotted line, double yellow line), poles, road boundaries, traffic lights with state, all directly rendered with appropriate colors and geometry patterns​
- **Dynamic 3D bounding boxes**: Indicate positions of vehicles and pedestrians with occlusion-aware and heading-aware representations​
- **Color coding**: Each object type is color-coded, with bounding boxes shaded according to the direction of motion, providing both semantic and motion cues​
- **Complex road topologies**: Can represent overpasses and intricate road structures​

<img width="818" height="957" alt="Screenshot 2025-12-05 at 4 40 45 PM" src="https://github.com/user-attachments/assets/d83a1cb3-1e9c-4ea2-bd03-3d51adb9dca4" />

Compared to previous control video approaches, the world scenario map improves control signal quality through fine-grained lane line type representation and more accurate dynamic object positioning.

- Resolution: 720p per camera view
- Frame rate: 10fps (optimized for LiDAR alignment and computational efficiency)​
- Camera configuration: Up to 7 synchronized views per clip
- Duration: 20 length clips

Each clip includes:​

- **Multi-camera text captions**: Describing scene content, weather, lighting, and activities
- **Raw RGB videos**
- **World scenario map videos**: combined spatial representation of lanes and objects

| Camera View | Raw RGB Videos | World Scenario Map Videos |
|-------------|----------------|---------------------------|
| **Front Wide** | <video width="320" controls autoplay loop muted><source src="https://github.com/user-attachments/assets/998c4e06-a077-4834-a19e-a427cac7e4cb" type="video/mp4"></video> | <video width="320" controls autoplay loop muted><source src="https://github.com/user-attachments/assets/0a973854-4a48-4803-a911-5f040dbaa520" type="video/mp4"></video> |
| **Cross Left** | <video width="320" controls autoplay loop muted><source src="https://github.com/user-attachments/assets/21930d1f-b98f-48ac-8564-f3235294d0d7" type="video/mp4"></video> | <video width="320" controls autoplay loop muted><source src="https://github.com/user-attachments/assets/a3831026-2a2c-4a41-bea3-f118d21de78e" type="video/mp4"></video> |
| **Cross Right** | <video width="320" controls autoplay loop muted><source src="https://github.com/user-attachments/assets/1ce0f192-913b-41f4-b677-aa5da43a1786" type="video/mp4"></video> | <video width="320" controls autoplay loop muted><source src="https://github.com/user-attachments/assets/69e3aa5a-072f-4603-b9cc-1a6d4669ee4d" type="video/mp4"></video> |

### Data Processing

To prepare the RDS-HQ dataset for multiview training, we apply several strategic processing decisions that balance model capability with computational efficiency:

- Training with variable view ranges to improve model robustness across different camera configurations​
- Uniform time weighting applied during training to handle AV data quality variations
- Shorter temporal window (8 latent frames vs 24 in other models) makes multiview task more tractable​

The key difference from single-view Cosmos models is the use of a specialized `ExtractFramesAndCaptions` augmentor that adds multiview-specific keys to the data loader:

- Synchronized multi-camera extraction: Processes frames from multiple camera views simultaneously while ensuring temporal alignment
- View index tracking: Maintains geometric relationships between cameras via `view_indices` and `view_indices_selection` tensors
- Camera-specific mappings:
    - `camera_video_key_mapping`: Maps camera names to video data sources
    - `camera_caption_key_mapping`: Maps camera names to caption sources
    - `camera_control_key_mapping`: Maps camera names to control inputs (HD maps, bounding boxes)​
- Caption conditioning flexibility:
    - Single caption mode: Uses one caption (e.g., from front camera) for all views
    - Per-view captions: Supports unique descriptions for each camera
    - Optional view prefixes: Can add camera-specific prefixes to captions for explicit view identification
- Control input extraction: For Transfer models, extracts spatially-aligned HD map and bounding box frames in the `control_input_hdmap_bbox` key​
- Consistency validation: Verifies frame indices and FPS match across all synchronized cameras
- Output keys specific to multiview:
    - `sample_n_views`: Number of cameras in this sample
    - `camera_keys_selection`: Which cameras were selected
    - `num_video_frames_per_view`: Frames per camera view
    - `front_cam_view_idx_sample_position`: Reference camera index for caption conditioning

Unlike single-view augmentors that process one video stream, `ExtractFramesAndCaptions` coordinates extraction across multiple synchronized cameras while maintaining strict temporal and spatial alignment requirements for multiview consistency.

## Post-Training Methodology

### Architecture: ControlNet Integration

Cosmos Transfer 2.5 Multiview extends the base Predict 2.5 Multiview model through a ControlNet architecture, which adds spatial conditioning capabilities while preserving the model's temporal and multi-view consistency.

The architecture consists of two main components:

- Control Branch:
    - A parallel neural network branch that processes spatial conditioning inputs
    - Accepts world scenario map visualization videos as inputs
    - Injects control features into the base model at multiple layers to guide generation
- Base Model (Frozen/Fine-tuned):
    - Initialized from the Cosmos Predict 2.5 Multiview checkpoint (2B parameters)
    - Uses a diffusion transformer architecture
    - Preserves learned multiview video generation capabilities

### Training Configuration

Control Inputs:

The model receives world scenario maps that combine lane geometry and bounding box information, providing comprehensive spatial context for each camera view.

Training Strategy:

Post-training builds on the Predict 2.5 Multiview checkpoint using the following approach:

- Train the control branch to encode spatial conditioning signals
- Leverage existing attention mechanisms to maintain temporal and multi-view consistency


Key Hyperparameters:

- Resolution: 720p per view
- Frame rate: 10fps​
- Context window: 29 frames with 8 latent frame state​
- Variable view sampling: views per training example​

### Design Decisions

- 10fps vs 30fps: The choice of 10fps aligns with LiDAR sensor frequency in AV systems and provides 3x computational efficiency for 10-second clips without sacrificing quality for target use cases.​
- Reduced Temporal Window: Using 8 latent frames instead of 24 makes the challenging multiview generation task more manageable while still capturing necessary temporal dynamics.​
- Variable View Training: Training with variable views in the range of 4 to 7, rather than fixed 7-view inputs improves model flexibility and robustness when deployed with different camera configurations.​
- Uniform Time Weighting: This training choice helps address quality variations inherent in autonomous vehicle datasets.​


## Post-Training with Your Own Data

This section provides instructions for post-training Cosmos Transfer 2.5 Multiview on your own dataset using the released checkpoint and [Cosmos-Transfer2.5 repository](https://github.com/nvidia-cosmos/cosmos-transfer2.5).

### 1. Preparing Data

#### 1.1 Prepare Transfer Multiview Training Dataset

The first step is preparing a dataset with videos.

You must provide a folder containing a collection of videos in **MP4 format**, preferably 720p, as well as a corresponding folder containing a collection of the hdmap control input videos in  **MP4 format**. The views for each samples should be further stratified by subdirectories with the camera name. We have an example dataset that can be used at `assets/multiview_hdmap_posttrain_dataset`

#### 1.2 Verify the dataset folder format

Dataset folder format:

```
assets/multiview_hdmap_posttrain_dataset/
├── captions/
│   └── ftheta_camera_front_wide_120fov/
│       └── *.json
├── control_input_hdmap_bbox/
│   ├── ftheta_camera_cross_left_120fov/
│   │   └── *.mp4
│   ├── ftheta_camera_cross_right_120fov/
│   │   └── *.mp4
│   ├── ftheta_camera_front_wide_120fov/
│   │   └── *.mp4
│   ├── ftheta_camera_front_tele_30fov/
│   │   └── *.mp4
│   ├── ftheta_camera_rear_left_70fov/
│   │   └── *.mp4
│   ├── ftheta_camera_rear_right_70fov/
│   │   └── *.mp4
│   ├── ftheta_camera_rear_tele_30fov/
│   │   └── *.mp4
├── videos/
│   ├── ftheta_camera_cross_left_120fov/
│   │   └── *.mp4
│   ├── ftheta_camera_cross_right_120fov/
│   │   └── *.mp4
│   ├── ftheta_camera_front_wide_120fov/
│   │   └── *.mp4
│   ├── ftheta_camera_front_tele_30fov/
│   │   └── *.mp4
│   ├── ftheta_camera_rear_left_70fov/
│   │   └── *.mp4
│   ├── ftheta_camera_rear_right_70fov/
│   │   └── *.mp4
│   ├── ftheta_camera_rear_tele_30fov/
│   │   └── *.mp4
```

### 2. Post-training Execution

Run the following command to execute an example post-training job with multiview data.

```bash
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py -- experiment=transfer2_auto_multiview_post_train_example job.wandb_mode=disabled
```

The model will be post-trained using the multiview dataset. 

Checkpoints are saved to `${IMAGINAIRE_OUTPUT_ROOT}/PROJECT/GROUP/NAME/checkpoints`.

Here is the example training experiment configuration:

```python
transfer2_auto_multiview_post_train_example = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT.experiment}",
        {"override /data_train": "example_multiview_train_data_control_input_hdmap"},
    ],
    job=dict(project="cosmos_transfer_v2p5", group="auto_multiview", name="2b_cosmos_multiview_post_train_example"),
    checkpoint=dict(
        save_iter=200,
        # pyrefly: ignore  # missing-attribute
        load_path=get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri),
        load_training_state=False,
        strict_resume=False,
        load_from_object_store=dict(
            enabled=False,  # Loading from local filesystem, not S3
        ),
        save_to_object_store=dict(
            enabled=False,
        ),
    ),
    model=dict(
        config=dict(
            base_load_from=None,
        ),
    ),
    trainer=dict(
        logging_iter=100,
        max_iter=5_000,
        callbacks=dict(
            heart_beat=dict(
                save_s3=False,
            ),
            iter_speed=dict(
                hit_thres=200,
                save_s3=False,
            ),
            device_monitor=dict(
                save_s3=False,
            ),
            every_n_sample_reg=dict(
                every_n=200,
                save_s3=False,
            ),
            every_n_sample_ema=dict(
                every_n=200,
                save_s3=False,
            ),
            wandb=dict(
                save_s3=False,
            ),
            wandb_10x=dict(
                save_s3=False,
            ),
            dataloader_speed=dict(
                save_s3=False,
            ),
            frame_loss_log=dict(
                save_s3=False,
            ),
        ),
    ),
    model_parallel=dict(
        context_parallel_size=int(os.environ.get("WORLD_SIZE", "1")),
    ),
)
```

### 3. Inference with the Post-trained checkpoint

#### 3.1 Converting DCP Checkpoint to Consolidated PyTorch Format

Since the checkpoints are saved in DCP format during training, you need to convert them to consolidated PyTorch format (.pt) for inference. Use the `convert_distcp_to_pt.py` script:

```bash
# Get path to the latest checkpoint
CHECKPOINTS_DIR=${IMAGINAIRE_OUTPUT_ROOT:-/tmp/imaginaire4-output}/cosmos_transfer_v2p5/auto_multiview/2b_cosmos_multiview_post_train_example/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)
CHECKPOINT_DIR=$CHECKPOINTS_DIR/$CHECKPOINT_ITER

# Convert DCP checkpoint to PyTorch format
python scripts/convert_distcp_to_pt.py $CHECKPOINT_DIR/model $CHECKPOINT_DIR
```

This conversion will create three files:

- `model.pt`: Full checkpoint containing both regular and EMA weights
- `model_ema_fp32.pt`: EMA weights only in float32 precision
- `model_ema_bf16.pt`: EMA weights only in bfloat16 precision (recommended for inference)

#### 3.2 Running Inference

After converting the checkpoint, you can run inference with your post-trained model using a JSON configuration file that specifies the inference parameters (see below for an example).

```json
{
    "name": "auto_multiview",
    "prompt_path": "prompt.txt",
    "fps": 10,
    "front_wide":{
        "input_path": "input_videos/night_front_wide_120fov.mp4",
        "control_path": "world_scenario_videos/ws_front_wide_120fov.mp4"
    },
    "cross_left":{
        "input_path": "input_videos/night_cross_left_120fov.mp4",
        "control_path": "world_scenario_videos/ws_cross_left_120fov.mp4"
    },
    "cross_right":{
        "input_path": "input_videos/night_cross_right_120fov.mp4",
        "control_path": "world_scenario_videos/ws_cross_right_120fov.mp4"
    },
    "rear_left":{
        "input_path": "input_videos/night_rear_left_70fov.mp4",
        "control_path": "world_scenario_videos/ws_rear_left_70fov.mp4"
    },
    "rear_right":{
        "input_path": "input_videos/night_rear_right_70fov.mp4",
        "control_path": "world_scenario_videos/ws_rear_right_70fov.mp4"
    },
    "rear":{
        "input_path": "input_videos/night_rear_30fov.mp4",
        "control_path": "world_scenario_videos/ws_rear_30fov.mp4"
    },
    "front_tele":{
        "input_path": "input_videos/night_front_tele_30fov.mp4",
        "control_path": "world_scenario_videos/ws_front_tele_30fov.mp4"
    }
}
```


```bash
export NUM_GPUS=8
torchrun --nproc_per_node=$NUM_GPUS --master_port=12341 -m examples.multiview -i assets/multiview_example/multiview_spec.json -o outputs/postrained-auto-mv --checkpoint_path $CHECKPOINT_DIR/model_ema_bf16.pt --experiment transfer2_auto_multiview_post_train_example
```

Generated videos will be saved to the output directory (e.g., `outputs/postrained-auto-mv/`).

For an explanation of all the available parameters run:

```bash
python examples/multiview.py --help
python examples/multiview.py control:view-config --help # for information specific to view configuration
```

Run autoregressive multiview (for generating longer videos):

```bash
torchrun --nproc_per_node=8 --master_port=12341 -m examples.multiview -i assets/multiview_example/multiview_autoregressive_spec.json -o outputs/multiview_autoregressive
```
 
## Results

### Evaluation Metrics

For evaluation, we used a 1000 multi-view clip dataset from RQS-HQ (Ren et al., 2025), which includes HD maps as well as human-labeled lanes and cuboids. These clips are disjoint from the datasets used in training, ensuring unbiased evaluation.​

Our evaluation framework consists of two complementary approaches:

#### Standard Video Quality Metrics
We measure perceptual quality and temporal consistency using:

- **FVD (Fréchet Video Distance)**: Evaluates visual quality and temporal coherence
- **FID (Fréchet Inception Distance)**: Measures frame-level visual quality
- **Temporal Sampson Error**: Assesses temporal consistency within each camera view
- **Cross-Camera Sampson Error**: Measures geometric consistency across multiple camera views

#### Spatial Accuracy Metrics
To test adherence to the control signals, we measure the detection performance of 3D-cuboid and lane detection models on generated videos and compare these with ground truth labels. Following the protocol described in Ren et al. (2025), we use:​

- **LATR (Luo et al., 2023)**: A monocular 3D lane detector for evaluating 3D lane detection tasks​
- **BEVFormer (Li et al., 2022)**: A temporal 3D object detector for evaluating 3D cuboid detection tasks​

This dual evaluation approach ensures that generated videos maintain both high visual quality and accurate spatial alignment with the control inputs.

#### Quantitative Results

The Cosmos Transfer 2.5 Multiview model demonstrates substantial improvements over the previous generation Transfer1-7B-Sample-AV baseline:

**Video Quality Improvements**:

<img width="675" height="166" alt="Screenshot 2025-12-05 at 4 46 04 PM" src="https://github.com/user-attachments/assets/f3f8d209-dd1a-49c0-a92f-0260f49eed34" />

We observe a significant boost (up to 2.3x improvement) in FVD/FID scores while remaining competitive in temporal and cross-camera Sampson error. This indicates that the model generates higher-quality, more realistic videos without sacrificing temporal consistency or multi-view geometric coherence.

**Spatial Accuracy Improvements**:

<img width="789" height="145" alt="Screenshot 2025-12-05 at 4 47 15 PM" src="https://github.com/user-attachments/assets/ea4829cf-61e1-4cdf-aeb9-0446564783aa" />

We observe a substantial improvement (up to 60%) in detection metrics compared to Transfer1-7B-Sample-AV. The enhanced performance on both lane detection and 3D cuboid detection tasks demonstrates that the model generates videos with significantly better adherence to the spatial control signals provided by world scenario maps.

#### Qualitative Analysis

Visual comparisons between Cosmos-Transfer1-7B-Sample-AV and Cosmos-Transfer2.5-2B/auto/multiview reveal significant quality improvements.

<img width="828" height="634" alt="Screenshot 2025-12-05 at 4 48 24 PM" src="https://github.com/user-attachments/assets/c0332162-a38a-4bd0-bebf-5ab18ea25166" />

Example (1) - Hallucination Reduction:

- Previous model Cosmos-Transfer1-7B-Sample-AV hallucinates a distorted black car behind the silver vehicle, which is described neither in the text prompt nor in the control video​
- Lack of alignment to the control signal when generating parked vehicles behind grassy mounds​
- Current model Cosmos-Transfer2.5-2B/auto/multiview correctly adheres to both text and spatial control inputs​

Example (2) - Spatial Consistency Improvements:

- Previous model renders the vehicle in the central lane driving on the wrong side of the street with incorrect orientation​
- Generates a truck instead of a pedestrian close to the sidewalk, contradicting the control input​
- Current model resolves all these inconsistencies, accurately representing vehicle positions, orientations, and object types​

#### Key Findings

The evaluation demonstrates that Cosmos Transfer 2.5 Multiview achieves:

- **Superior visual quality**: 2.3x improvement in perceptual metrics​
- **Enhanced spatial control**: 60% better detection performance on controlled elements​
- **Reduced hallucinations**: Fewer spurious objects not specified in prompts or control signals​
- **Improved consistency**: Better adherence to both semantic (text) and geometric (world scenario map) conditioning​
- **Maintained geometric coherence**: Competitive temporal and cross-camera consistency

## Conclusion
Cosmos Transfer 2.5 Multiview successfully addresses the spatial control requirements of autonomous vehicle development by extending Cosmos Predict 2.5 Multiview with ControlNet-based conditioning. The resulting system generates photorealistic multi-camera driving scenarios that accurately reflect specific HD map layouts and object configurations, combining visual quality with the deterministic spatial properties necessary for systematic testing and simulation workflows.

The key innovation lies in maintaining multiview consistency and temporal dynamics while respecting spatial constraints encoded in world scenario maps. Lane detection and 3D cuboid evaluation demonstrate high spatial accuracy, transforming video generation from a creative tool into a precision instrument suitable for engineering applications where spatial fidelity directly impacts testing validity. Critical design decisions—operating at 10 fps for LiDAR alignment, using 8 latent frames for tractability, and training with variable view counts for robustness—reflect pragmatic choices that balance real-world constraints with quality requirements.

Looking forward, Cosmos Transfer 2.5 Multiview enables systematic exploration of perception system behavior across diverse conditions while holding spatial layout constant, supporting comprehensive testing of autonomous vehicle algorithms. This work demonstrates that post-training with spatial conditioning represents a viable path toward production-ready video generation, combining photorealism with the spatial determinism essential for reliable autonomous vehicle development workflows.
