# Additional Examples from Cosmos Model Repos

This page provides links to inference and post-training examples from the official Cosmos model repositories. These examples complement the end-to-end tutorials in the cookbook with comprehensive guides for model usage and customization.

## Cosmos Predict

### Cosmos Predict 2.5 *(Latest)*

For the latest Cosmos Predict 2.5 model documentation, visit the [Cosmos Predict 2.5 Repository](https://github.com/nvidia-cosmos/cosmos-predict2.5).

#### Inference with Pre-Trained Cosmos Predict 2.5 Models

- **[Inference Guide](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/docs/inference.md)**: Video generation with Text2World, Image2World, and Video2World capabilities
- **[Auto Multiview Inference Guide](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/docs/inference_auto_multiview.md)**: Multi-camera view generation for autonomous vehicle applications

#### Post-Training with Cosmos Predict 2.5 Models

- **[Video2World Post-Training for DreamGen Bench](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/docs/post-training_video2world_gr00t.md)**: Humanoid robot trajectory generation using the DreamGen benchmark

### Cosmos Predict 2

#### Inference with Pre-Trained Cosmos Predict 2 Models

- **[Text2Image Inference](https://github.com/nvidia-cosmos/cosmos-predict2/tree/main/documentations/inference_text2image.md)**: Generating high-quality images from text prompts
- **[Video2World Inference](https://github.com/nvidia-cosmos/cosmos-predict2/tree/main/documentations/inference_video2world.md)**: Generating videos from images/videos with text prompts (single/batch processing, multi-frame conditioning, multi-GPU inference, prompt refiner, rejection sampling)
- **[Text2World Inference](https://github.com/nvidia-cosmos/cosmos-predict2/tree/main/documentations/inference_text2world.md)**: Generating videos directly from text prompts (single/batch processing, multi-GPU inference)

#### Post-Training with Cosmos Predict 2 Models

- **[Video2World Post-Training Guide](https://github.com/nvidia-cosmos/cosmos-predict2/tree/main/documentations/post-training_video2world.md)**: General guide to the Video2World training system
- **[Video2World Post-Training on Cosmos-NeMo-Assets](https://github.com/nvidia-cosmos/cosmos-predict2/tree/main/documentations/post-training_video2world_cosmos_nemo_assets.md)**: Post-training on Cosmos-NeMo-Assets data
- **[Video2World Post-Training on Fisheye-View AgiBotWorld-Alpha Dataset](https://github.com/nvidia-cosmos/cosmos-predict2/tree/main/documentations/post-training_video2world_agibot_fisheye.md)**: Post-training on fisheye-view robot videos from the AgiBotWorld-Alpha dataset
- **[Video2World Post-Training on GR00T Dreams GR1 and DROID Datasets](https://github.com/nvidia-cosmos/cosmos-predict2/tree/main/documentations/post-training_video2world_gr00t.md)**: Post-training on GR00T Dreams GR1 and DROID datasets
- **[Video2World Action-Conditioned Post-Training on Bridge Dataset](https://github.com/nvidia-cosmos/cosmos-predict2/tree/main/documentations/post-training_video2world_action.md)**: Action-conditioned post-training on Bridge dataset
- **[Text2Image Post-Training Guide](https://github.com/nvidia-cosmos/cosmos-predict2/tree/main/documentations/post-training_text2image.md)**: General guide to the Text2Image training system
- **[Text2Image Post-Training on Cosmos-NeMo-Assets](https://github.com/nvidia-cosmos/cosmos-predict2/tree/main/documentations/post-training_text2image_cosmos_nemo_assets.md)**: Post-training on Cosmos-NeMo-Assets image data

## Cosmos Transfer

### Cosmos Transfer 2.5 *(Latest)*

For the latest Cosmos Transfer 2.5 model documentation, visit the [Cosmos Transfer 2.5 Repository](https://github.com/nvidia-cosmos/cosmos-transfer2.5).

#### Inference with Pre-Trained Cosmos Transfer 2.5 Models

- **[Inference Guide](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/docs/inference.md)**: Multi-control video generation with depth, segmentation, LiDAR, and HDMap conditioning
- **[Auto Multiview Inference Guide](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/docs/inference_auto_multiview.md)**: Multi-camera view generation for autonomous vehicle applications

#### Post-Training with Cosmos Transfer 2.5 Models

- **[Post-Training Guide](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/docs/post-training.md)**: General guide for custom control modalities and domain adaptation
- **[Auto Multiview Post-Training for HDMap](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/docs/post-training_auto_multiview.md)**: Multi-view autonomous driving scenarios with HDMap control

### Cosmos Transfer 1

#### Inference with Pre-Trained Cosmos Transfer 1 Models

- **[Cosmos-Transfer1-7B Inference](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/inference_cosmos_transfer1_7b.md)**: Multi-GPU support
- **[Cosmos-Transfer1-7B-Sample-AV Inference](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/inference_cosmos_transfer1_7b_sample_av.md)**: Multi-GPU support
- **[Cosmos-Transfer1-7B-4KUpscaler Inference](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/inference_cosmos_transfer1_7b_4kupscaler.md)**: 4K upscaling with multi-GPU support
- **[Cosmos-Transfer1-7B Inference (Depth)](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/inference_cosmos_transfer1_7b_depth.md)**: Depth-based control
- **[Cosmos-Transfer1-7B Inference (Segmentation)](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/inference_cosmos_transfer1_7b_seg.md)**: Segmentation-based control
- **[Cosmos-Transfer1-7B Inference (Edge)](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/inference_cosmos_transfer1_7b.md#example-1-single-control-edge)**: Edge-based control
- **[Cosmos-Transfer1-7B Inference (Vis)](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/inference_cosmos_transfer1_7b_vis.md)**: Visual-based control
- **[Cosmos-Transfer1pt1-7B Inference (Keypoint)](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/inference_cosmos_transfer1pt1_7b_keypoint.md)**: Keypoint-based control
- **[Cosmos-Transfer1-7B-Sample-AV-Multiview Inference](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/inference_cosmos_transfer1_7b_sample_av_single2multiview.md)**: Multi-view generation

#### Post-Training with Cosmos Transfer 1 Models

- **[Cosmos-Transfer1-7B Post-Training](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/training_cosmos_transfer_7b.md)**: Depth, Edge, Keypoint, Segmentation, and Vis controls with multi-GPU support
- **[Cosmos-Transfer1-7B-Sample-AV Post-Training](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/training_cosmos_transfer_7B_sample_AV.md)**: LiDAR and HDMap controls with multi-GPU support
- **[Cosmos-Transfer1-7B-Sample-AV-Multiview Post-Training](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/training_cosmos_transfer_7B_sample_AV.md)**: Multi-view LiDAR and HDMap controls with multi-GPU support

#### Post-Training Cosmos Transfer 1 Models from Scratch

- **[Cosmos-Transfer1-7B Post-Training](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/training_cosmos_transfer_7b.md)**: Depth, Edge, Keypoint, Segmentation, and Vis controls with multi-GPU support
- **[Cosmos-Transfer1-7B-Sample-AV Post-Training](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/training_cosmos_transfer_7B_sample_AV.md)**: LiDAR and HDMap controls with multi-GPU support
- **[Cosmos-Transfer1-7B-Sample-AV-Multiview Post-Training](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/training_cosmos_transfer_7B_sample_AV.md)**: Multi-view LiDAR and HDMap controls with multi-GPU support

## Cosmos Reason 1

For the latest Cosmos Reason 1 model documentation, visit the [Cosmos Reason 1 Repository](https://github.com/nvidia-cosmos/cosmos-reason1).

### Post-Training with Cosmos Reason 1 Models

- **[Cosmos Reason 1 Post-Training Example](https://github.com/nvidia-cosmos/cosmos-reason1/blob/main/examples/post_training/README.md)**: Complete post-training guide for vision-language reasoning tasks
