# Post-Training examples from Cosmos Model repos

## Cosmos Predict2

### Inference with pre-trained Cosmos-Predict2 models

- **[Text2Image Inference](https://github.com/nvidia-cosmos/cosmos-predict2/tree/main/documentations/inference_text2image.md)**: Guide for generating high-quality images from text prompts
- **[Video2World Inference](https://github.com/nvidia-cosmos/cosmos-predict2/tree/main/documentations/inference_video2world.md)**: Guide for generating videos from images/videos with text prompts, including:
  - Single and batch processing
  - Multi-frame conditioning
  - Multi-GPU inference for faster generation
  - Using the prompt refiner
  - Rejection sampling for quality improvement
- **[Text2World Inference](https://github.com/nvidia-cosmos/cosmos-predict2/tree/main/documentations/inference_text2world.md)**: Guide for generating videos directly from text prompts, including:
  - Single and batch processing
  - Multi-GPU inference for faster generation

### Post-train pre-trained Cosmos-Predict2 models

- **[Video2World Post-training guide](https://github.com/nvidia-cosmos/cosmos-predict2/tree/main/documentations/post-training_video2world.md)**: General guide to the video2world training system in the codebase
- **[Video2World Post-training on Cosmos-NeMo-Assets](https://github.com/nvidia-cosmos/cosmos-predict2/tree/main/documentations/post-training_video2world_cosmos_nemo_assets.md)**: Case study for post-training on Cosmos-NeMo-Assets data
- **[Video2World Post-training on fisheye-view AgiBotWorld-Alpha dataset](https://github.com/nvidia-cosmos/cosmos-predict2/tree/main/documentations/post-training_video2world_agibot_fisheye.md)**: Case study for post-training on fisheye-view robot videos from AgiBotWorld-Alpha dataset
- **[Video2World Post-training on GR00T Dreams GR1 and DROID datasets](https://github.com/nvidia-cosmos/cosmos-predict2/tree/main/documentations/post-training_video2world_gr00t.md)**: Case study for post-training on GR00T Dreams GR1 and DROID datasets
- **[Video2World Action-conditioned Post-training on Bridge dataset](https://github.com/nvidia-cosmos/cosmos-predict2/tree/main/documentations/post-training_video2world_action.md)**: Case study for action-conditioned post-training on Bridge dataset
- **[Text2Image Post-training guide](https://github.com/nvidia-cosmos/cosmos-predict2/tree/main/documentations/post-training_text2image.md)**: General guide to the text2image training system in the codebase
- **[Text2Image Post-training on Cosmos-NeMo-Assets](https://github.com/nvidia-cosmos/cosmos-predict2/tree/main/documentations/post-training_text2image_cosmos_nemo_assets.md)**: Case study for post-training on Cosmos-NeMo-Assets image data

## Cosmos Transfer1

### Inference with pre-trained Cosmos-Transfer1 models

- [Inference with pre-trained Cosmos-Transfer1-7B](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/inference_cosmos_transfer1_7b.md) **[with multi-GPU support]**
- [Inference with pre-trained Cosmos-Transfer1-7B-Sample-AV](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/inference_cosmos_transfer1_7b_sample_av.md) **[with multi-GPU support]**
- [Inference with pre-trained Cosmos-Transfer1-7B-4KUpscaler](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/inference_cosmos_transfer1_7b_4kupscaler.md) **[with multi-GPU support]**
- [Inference with pre-trained Cosmos-Transfer1-7B (Depth)](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/inference_cosmos_transfer1_7b_depth.md)
- [Inference with pre-trained Cosmos-Transfer1-7B (Segmentation)](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/inference_cosmos_transfer1_7b_seg.md)
- [Inference with pre-trained Cosmos-Transfer1-7B (Edge)](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/inference_cosmos_transfer1_7b.md#example-1-single-control-edge)
- [Inference with pre-trained Cosmos-Transfer1-7B (Vis)](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/inference_cosmos_transfer1_7b_vis.md)
- [Inference with pre-trained Cosmos-Transfer1pt1-7B [Keypoint]](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/inference_cosmos_transfer1pt1_7b_keypoint.md)
- [Inference with pre-trained Cosmos-Transfer1-7B-Sample-AV-Multiview](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/inference_cosmos_transfer1_7b_sample_av_single2multiview.md)

### Post-train pre-trained Cosmos-Transfer1 models

- [Post-train pre-trained Cosmos-Transfer1-7B [Depth | Edge | Keypoint | Segmentation | Vis]](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/training_cosmos_transfer_7b.md) **[with multi-GPU support]**
- [Post-train pre-trained Cosmos-Transfer1-7B-Sample-AV [LiDAR|HDMap]](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/training_cosmos_transfer_7B_sample_AV.md) **[with multi-GPU support]**
- [Post-train pre-trained Cosmos-Transfer1-7B-Sample-AV-Multiview[LiDAR|HDMap]](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/training_cosmos_transfer_7B_sample_AV.md) **[with multi-GPU support]**

### Build your own Cosmos-Transfer1 models from scratch

- [Pre-train Cosmos-Transfer1-7B [Depth | Edge | Keypoint | Segmentation | Vis]](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/training_cosmos_transfer_7b.md) **[with multi-GPU support]**
- [Pre-train Cosmos-Transfer1-7B-Sample-AV [LiDAR|HDMap]](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/training_cosmos_transfer_7B_sample_AV.md) **[with multi-GPU support]**
- [Pre-train Cosmos-Transfer1-7B-Sample-AV-Multiview[LiDAR|HDMap]](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/examples/training_cosmos_transfer_7B_sample_AV.md) **[with multi-GPU support]**

## Cosmos Reason1

[Cosmos-Reason1 Post-Training Example](https://github.com/nvidia-cosmos/cosmos-reason1/blob/main/examples/post_training/README.md)
