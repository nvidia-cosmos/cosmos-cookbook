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
The post-training dataset, MADS (Multiview Autonomous Driving Scenarios) consists of 400K multi-camera clips specifically curated for spatially-conditioned generation:​

- Resolution: 720p per camera view
- Frame rate: 10fps (optimized for LiDAR alignment and computational efficiency)​
- Camera configuration: Up to 7 synchronized views per clip
- Duration: Variable length clips

Each clip includes:​

- **Multi-camera text captions**: Describing scene content, weather, lighting, and activities
- **Raw RGB videos**
- **World scenario map videos**: combined spatial representation of lanes and objects

### Data Processing

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
Cosmos Transfer 2.5 Multiview extends the base Predict 2.5 model through a ControlNet architecture:​

- Control Branch:
    - Parallel neural network branch processing spatial conditioning inputs
    - Accepts world scenario map visualization videos as inputs
    - Control features injected into the base model at multiple layers
- Base Model (Frozen/Fine-tuned):
    - Cosmos Predict 2.5 Multiview checkpoint serves as initialization​
    - 2B parameter diffusion transformer architecture
    - Preserves learned multiview video generation capabilities

### Training Configuration

Control Inputs:

- World scenario maps combining lanes and bounding boxes​

Training Strategy:

- Post-training from Predict 2.5 Multiview checkpoint​
- Train control branch while optionally fine-tuning base model layers
- Maintain temporal and multi-view consistency through existing attention mechanisms

Key Hyperparameters:

- Resolution: 720p per view
- Frame rate: 10fps​
- Context window: 29 frames with 8 latent frame state​
- Variable view sampling: views per training example​

#### Training Configuration Setup

```python
def xiaomi_transfer2p5_2b_mv_7views_res720p_fps10_t8_fromfinetuned12knofpsuniform_mads720pmulticaps29frames_world_scenario_nofps_uniform() -> (
   dict
):
   text_encoder_ckpt_path = "s3://checkpoints-us-east-1/cosmos_reasoning1/sft_exp700/sft_exp721-1_qwen7b_tl_721_5vs5_s3_balanced_n32_resume_16k/checkpoints/iter_000016000/model/"
   base_load_path = "checkpoints-us-east-1/cosmos_predict2_multiview/cosmos2_mv/xiaomi_predict2p5_2b_7views_res720p_fps30_t8_joint_alpamayo1capviewprefix_allcapsviewprefix_29frames_nofps_uniform_dropoutt0-0/checkpoints/iter_000012000/"
   base_load_credentials = "credentials/s3_checkpoint.secret"


   return dict(
       defaults=[
           {"override /data_train": "video_control_mads_multiview_0823_s3_720p_10fps_29frames_7views"},
           {"override /model": "fsdp_rectified_flow_multiview_control"},
           {"override /net": "cosmos_v1_2B_multiview_control"},
           {"override /conditioner": "video_prediction_multiview_control_conditioner"},
           {"override /ckpt_type": "dcp"},
           {"override /optimizer": "fusedadamw"},
           {
               "override /callbacks": [
                   "basic",
                   "wandb",
                   "cluster_speed",
                   "load_base_model_callbacks",
               ]
           },
           {"override /checkpoint": "s3"},
           {"override /tokenizer": "wan2pt1_tokenizer"},
           "_self_",
       ],
       job=dict(
           group="cosmos2_mv",
           name="xiaomi_transfer2p5_2b_mv_7views_res720p_fps10_t8_fromfinetuned12knofpsuniform_mads720pmulticaps29frames_world_scenario_nofps_uniform",
       ),
       checkpoint=dict(
           save_iter=500,
           load_path="",
           load_from_object_store=dict(
               enabled=True,
           ),
           save_to_object_store=dict(
               enabled=True,
           ),
           strict_resume=False,
       ),
       optimizer=dict(
           lr=8.63e-5,  # 2**(-14.5) = 3.0517578125e-05
           weight_decay=1e-3,
           betas=[0.9, 0.999],
       ),
       scheduler=dict(
           f_max=[0.5],
           f_min=[0.2],
           warm_up_steps=[1000],
           cycle_lengths=[100_000],
       ),
       model_parallel=dict(
           context_parallel_size=8,
       ),
       model=dict(
           config=dict(
               hint_keys="hdmap_bbox",
               min_num_conditional_frames_per_view=0,  # t2w
               max_num_conditional_frames_per_view=2,  # i2w or v2v
               condition_locations=["first_random_n"],
               train_sample_views_range=[7, 7],
               conditional_frames_probs={0: 0.5, 1: 0.25, 2: 0.25},
               state_t=8,
               online_text_embeddings_as_dict=False,
               fsdp_shard_size=8,
               resolution="720p",
               shift=5,
               use_dynamic_shift=False,
               train_time_weight="uniform",
               train_time_distribution="logitnormal",
               net=dict(
                   timestep_scale=0.001,
                   use_wan_fp32_strategy=True,
                   concat_view_embedding=True,
                   view_condition_dim=7,
                   state_t=8,
                   n_cameras_emb=7,
                   vace_has_mask=False,
                   use_input_hint_block=True,
                   condition_strategy="spaced",
                   vace_block_every_n=7,  # 4 layers
                   rope_enable_fps_modulation=False,
                   rope_h_extrapolation_ratio=3.0,
                   rope_w_extrapolation_ratio=3.0,
                   rope_t_extrapolation_ratio=8.0 / 24.0,
                   use_crossattn_projection=True,
                   crossattn_proj_in_channels=100352,
                   crossattn_emb_channels=1024,
                   sac_config=dict(
                       mode="predict2_2b_720_aggressive",
                   ),
               ),
               conditioner=dict(
                   use_video_condition=dict(
                       dropout_rate=0.0,
                   ),
                   text=dict(
                       dropout_rate=0.2,
                       use_empty_string=False,
                   ),
               ),
               tokenizer=dict(
                   temporal_window=16,
               ),
               text_encoder_class="reason1p1_7B",
               text_encoder_config=dict(
                   embedding_concat_strategy=str(EmbeddingConcatStrategy.FULL_CONCAT),
                   compute_online=True,
                   ckpt_path=text_encoder_ckpt_path,
               ),
               base_load_from=dict(
                   load_path=base_load_path,
                   credentials=base_load_credentials,
               ),
           )
       ),
       trainer=dict(
           max_iter=100_000,
           logging_iter=100,
           callbacks=dict(
               compile_tokenizer=dict(
                   enabled=False,
               ),
               iter_speed=dict(
                   hit_thres=50,
                   every_n=100,
               ),
               grad_clip=dict(
                   clip_norm=0.1,
               ),
               every_n_sample_reg=L(EveryNDrawSampleMultiviewVideo)(
                   every_n=2_000,
                   is_x0=False,
                   is_ema=False,
                   num_sampling_step=35,
                   guidance=[7],
                   fps=10,
                   ctrl_hint_keys=["control_input_hdmap_bbox"],
                   control_weights=[0.0, 1.0],
               ),
               every_n_sample_ema=L(EveryNDrawSampleMultiviewVideo)(
                   every_n=2_000,
                   is_x0=False,
                   is_ema=True,
                   num_sampling_step=35,
                   guidance=[7],
                   fps=10,
                   ctrl_hint_keys=["control_input_hdmap_bbox"],
                   control_weights=[0.0, 1.0],
               ),
           ),
           straggler_detection=dict(
               enabled=False,
           ),
       ),
       dataloader_train=dict(
           augmentation_config=dict(
               single_caption_camera_name="camera_front_wide_120fov",
               add_view_prefix_to_caption=True,
           ),
       ),
       upload_reproducible_setup=True,
   )
```

#### Training Execution
```bash
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=projects/cosmos/transfer2_multiview/configs/vid2vid_transfer/config.py -- experiment=transfer2_auto_multiview_post_train_example job.wandb_mode=disabled
```

## Inference with Transfer2.5 Multiview

Run multiview2world:

```bash
torchrun --nproc_per_node=8 --master_port=12341 examples/multiview.py -i assets/multiview_example/multiview_spec.json -o outputs/multiview/
```

For an explanation of all the available parameters run:

```bash
python examples/multiview.py --help
python examples/multiview.py control:view-config --help # for information specific to view configuration
```

Run autoregressive multiview (for generating longer videos):

```bash
torchrun --nproc_per_node=8 --master_port=12341 -m examples.multiview -i assets/multiview_example/multiview_autoregressive_spec.json -o outputs/multiview_autoregressive
```

### Design Decisions

- 10fps vs 30fps: The choice of 10fps aligns with LiDAR sensor frequency in AV systems and provides 3x computational efficiency for 10-second clips without sacrificing quality for target use cases.​
- Reduced Temporal Window: Using 8 latent frames instead of 24 makes the challenging multiview generation task more manageable while still capturing necessary temporal dynamics.​
- Variable View Training: Training with variable views in the range of 4 to 7, rather than fixed 7-view inputs improves model flexibility and robustness when deployed with different camera configurations.​
- Uniform Time Weighting: This training choice helps address quality variations inherent in autonomous vehicle datasets.​

 
## Results

### Evaluation Metrics
Unlike standard video generation models evaluated with just FID/FVD, Transfer models require spatial accuracy metrics:​

- Lane Detection Accuracy:

    - Run lane detection models on generated videos

    - Compare detected lanes against input HD map ground truth

    - Measure geometric accuracy and consistency across views​

- 3D Cuboid Evaluation:

    - Assess object placement accuracy relative to input bounding boxes

    - Evaluate geometric consistency of 3D objects across camera views

    - Measure adherence to specified object trajectories​

#### Quantitative Results

The trained Cosmos Transfer 2.5 Multiview model demonstrates:

- Significant improvement in lane geometry accuracy compared to text-only baseline​
- High spatial fidelity to input HD maps and scenario specifications
- Maintained temporal consistency and video quality from base Predict model
- Robust performance across variable camera configurations (4-7 views)

Detailed evaluation results and methodology are available in the [paper](https://arxiv.org/abs/2511.00062).

#### Qualitative Observations

- Generated videos accurately reflect input HD map lane layouts
- Objects appear at specified positions with correct 3D geometry across views
- Natural appearance and dynamics consistent with real AV footage
- Successful generation of diverse scenarios while maintaining spatial control

## Conclusion
Cosmos Transfer 2.5 Multiview successfully addresses the spatial control requirements of autonomous vehicle development by extending Cosmos Predict 2.5 Multiview with ControlNet-based conditioning. The resulting system generates photorealistic multi-camera driving scenarios that accurately reflect specific HD map layouts and object configurations, combining visual quality with the deterministic spatial properties necessary for systematic testing and simulation workflows.

The key innovation lies in maintaining multiview consistency and temporal dynamics while respecting spatial constraints encoded in world scenario maps. Lane detection and 3D cuboid evaluation demonstrate high spatial accuracy, transforming video generation from a creative tool into a precision instrument suitable for engineering applications where spatial fidelity directly impacts testing validity. Critical design decisions—operating at 10 fps for LiDAR alignment, using 8 latent frames for tractability, and training with variable view counts for robustness—reflect pragmatic choices that balance real-world constraints with quality requirements.

Looking forward, Cosmos Transfer 2.5 Multiview enables systematic exploration of perception system behavior across diverse conditions while holding spatial layout constant, supporting comprehensive testing of autonomous vehicle algorithms. This work demonstrates that post-training with spatial conditioning represents a viable path toward production-ready video generation, combining photorealism with the spatial determinism essential for reliable autonomous vehicle development workflows.
