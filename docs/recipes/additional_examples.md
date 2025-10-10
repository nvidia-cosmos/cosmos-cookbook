# Additional Examples from Cosmos Model Repos

This page provides links to inference and post-training examples from the official Cosmos model repositories. These examples complement the end-to-end tutorials in the cookbook with comprehensive guides for model usage and customization.

## Cosmos Predict

### Cosmos Predict 2.5 *(Latest)*

For the latest Cosmos Predict 2.5 model documentation, visit the [Cosmos Predict 2.5 Repository](https://github.com/nvidia-cosmos/cosmos-predict2.5).

#### Inference with Pre-Trained Cosmos Predict 2.5 Models

- **[Inference Guide](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/docs/inference.md)**: Comprehensive guide for generating videos with Cosmos Predict 2.5, including Text2World, Image2World, and Video2World capabilities
- **[Auto Multiview Inference Guide](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/docs/inference_auto_multiview.md)**: Guide for multi-camera view generation for autonomous vehicle applications

#### Post-train Pre-Trained Cosmos Predict 2.5 Models

- **[Video2World Post-training for DreamGen Bench](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/docs/post-training_video2world_gr00t.md)**: Post-training guide for humanoid robot trajectory generation using the DreamGen benchmark

### Cosmos Predict 2

#### Inference with Pre-Trained Cosmos Predict 2 Models

- **[Text2Image Inference](https://github.com/nvidia-cosmos/cosmos-predict2/tree/main/documentations/inference_text2image.md)**: A guide for generating high-quality images from text prompts
- **[Video2World Inference](https://github.com/nvidia-cosmos/cosmos-predict2/tree/main/documentations/inference_video2world.md)**: A guide for generating videos from images/videos with text prompts, including the following:
  - Single and batch processing
  - Multi-frame conditioning
  - Multi-GPU inference for faster generation
  - Using the prompt refiner
  - Rejection sampling for quality improvement
- **[Text2World Inference](https://github.com/nvidia-cosmos/cosmos-predict2/tree/main/documentations/inference_text2world.md)**: A guide for generating videos directly from text prompts, including the following:
  - Single and batch processing
  - Multi-GPU inference for faster generation

#### Post-train Pre-Trained Cosmos Predict 2 Models

- **[Video2World Post-training guide](https://github.com/nvidia-cosmos/cosmos-predict2/tree/main/documentations/post-training_video2world.md)**: A general guide to the Video2World training system in the codebase
- **[Video2World Post-training on Cosmos-NeMo-Assets](https://github.com/nvidia-cosmos/cosmos-predict2/tree/main/documentations/post-training_video2world_cosmos_nemo_assets.md)**: A case study for post-training on Cosmos-NeMo-Assets data
- **[Video2World Post-training on fisheye-view AgiBotWorld-Alpha dataset](https://github.com/nvidia-cosmos/cosmos-predict2/tree/main/documentations/post-training_video2world_agibot_fisheye.md)**: A case study for post-training on fisheye-view robot videos from the AgiBotWorld-Alpha dataset.
- **[Video2World Post-training on GR00T Dreams GR1 and DROID datasets](https://github.com/nvidia-cosmos/cosmos-predict2/tree/main/documentations/post-training_video2world_gr00t.md)**: A case study for post-training on GR00T Dreams GR1 and DROID datasets.
- **[Video2World Action-conditioned Post-training on Bridge dataset](https://github.com/nvidia-cosmos/cosmos-predict2/tree/main/documentations/post-training_video2world_action.md)**: A case study for action-conditioned post-training on Bridge dataset.
- **[Text2Image Post-training guide](https://github.com/nvidia-cosmos/cosmos-predict2/tree/main/documentations/post-training_text2image.md)**: A general guide to the Text2Image training system in the codebase.
- **[Text2Image Post-training on Cosmos-NeMo-Assets](https://github.com/nvidia-cosmos/cosmos-predict2/tree/main/documentations/post-training_text2image_cosmos_nemo_assets.md)**: A case study for post-training on Cosmos-NeMo-Assets image data.

## Cosmos Transfer

### Cosmos Transfer 2.5 *(Latest)*

For the latest Cosmos Transfer 2.5 model documentation, visit the [Cosmos Transfer 2.5 Repository](https://github.com/nvidia-cosmos/cosmos-transfer2.5).

#### Inference with Pre-Trained Cosmos Transfer 2.5 Models

- **[Inference Guide](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/docs/inference.md)**: Comprehensive guide for multi-control video generation with Cosmos Transfer 2.5, including depth, segmentation, LiDAR, and HDMap conditioning
- **[Auto Multiview Inference Guide](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/docs/inference_auto_multiview.md)**: Guide for multi-camera view generation for autonomous vehicle applications

#### Post-Train Pre-Trained Cosmos Transfer 2.5 Models

- **[Post-Training Guide](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/docs/post-training.md)**: General guide to post-training Cosmos Transfer 2.5 models for custom control modalities and domain adaptation
- **[Auto Multiview Post-training for HDMap](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/docs/post-training_auto_multiview.md)**: Post-training guide for multi-view autonomous driving scenarios with HDMap control

### Cosmos Transfer 1

#### Inference with Pre-Trained Cosmos Transfer 1 Models

- [Inference with pre-trained Cosmos-Transfer1-7B](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/inference_cosmos_transfer1_7b.md) **[with multi-GPU support]**
- [Inference with pre-trained Cosmos-Transfer1-7B-Sample-AV](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/inference_cosmos_transfer1_7b_sample_av.md) **[with multi-GPU support]**
- [Inference with pre-trained Cosmos-Transfer1-7B-4KUpscaler](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/inference_cosmos_transfer1_7b_4kupscaler.md) **[with multi-GPU support]**
- [Inference with pre-trained Cosmos-Transfer1-7B (Depth)](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/inference_cosmos_transfer1_7b_depth.md)
- [Inference with pre-trained Cosmos-Transfer1-7B (Segmentation)](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/inference_cosmos_transfer1_7b_seg.md)
- [Inference with pre-trained Cosmos-Transfer1-7B (Edge)](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/inference_cosmos_transfer1_7b.md#example-1-single-control-edge)
- [Inference with pre-trained Cosmos-Transfer1-7B (Vis)](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/inference_cosmos_transfer1_7b_vis.md)
- [Inference with pre-trained Cosmos-Transfer1pt1-7B [Keypoint]](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/inference_cosmos_transfer1pt1_7b_keypoint.md)
- [Inference with pre-trained Cosmos-Transfer1-7B-Sample-AV-Multiview](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/inference_cosmos_transfer1_7b_sample_av_single2multiview.md)

#### Post-Train Pre-Trained Cosmos Transfer 1 Models

- [Post-train pre-trained Cosmos-Transfer1-7B [Depth | Edge | Keypoint | Segmentation | Vis]](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/training_cosmos_transfer_7b.md) **[with multi-GPU support]**
- [Post-train pre-trained Cosmos-Transfer1-7B-Sample-AV [LiDAR|HDMap]](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/training_cosmos_transfer_7B_sample_AV.md) **[with multi-GPU support]**
- [Post-train pre-trained Cosmos-Transfer1-7B-Sample-AV-Multiview[LiDAR|HDMap]](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/training_cosmos_transfer_7B_sample_AV.md) **[with multi-GPU support]**

#### Build your own Cosmos Transfer 1 Models from Scratch

- [Pre-train Cosmos-Transfer1-7B [Depth | Edge | Keypoint | Segmentation | Vis]](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/training_cosmos_transfer_7b.md) **[with multi-GPU support]**
- [Pre-train Cosmos-Transfer1-7B-Sample-AV [LiDAR|HDMap]](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/training_cosmos_transfer_7B_sample_AV.md) **[with multi-GPU support]**
- [Pre-train Cosmos-Transfer1-7B-Sample-AV-Multiview[LiDAR|HDMap]](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/training_cosmos_transfer_7B_sample_AV.md) **[with multi-GPU support]**

## Cosmos Reason 1

[Cosmos Reason 1 Post-Training Example](https://github.com/nvidia-cosmos/cosmos-reason1/blob/main/examples/post_training/README.md)
