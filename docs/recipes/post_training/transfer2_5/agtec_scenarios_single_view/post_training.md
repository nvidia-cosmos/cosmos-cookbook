# Sim2Real for Agriculture via Cosmos Transfer Post-Training

> **Authors:** *[Author Name](https://www.linkedin.com/) • [Author Name](https://www.linkedin.com/)*  
> **Organization:** [Aigen](https://www.aigen.io/)

> **⚠️ DRAFT — Under Review**  

## Overview

| **Model** | **Workload** | **Use Case** |
|-----------|--------------|--------------|
| Cosmos Transfer 2.5 | Post-training + Inference | Generating photorealistic synthetic agricultural imagery for robot perception training |

> **TODO for authors:** Add a hero image/gif/video showing the pipeline in action.
![3D Render](assets/image3drender.webp) ![Deep Control](assets/deepcontrol.webp) ![Diseased](assets/Diseased.webp)

Training robust perception models for agricultural robotics requires diverse, high-quality labeled data spanning crops, growth stages, weather conditions, and disease states. Agriculture presents a particularly challenging long-tail distribution problem — the number of variations across plant species, environmental conditions, and field states is enormous, and real-world data collection is slow, seasonal, and expensive.

This recipe demonstrates how to post-train Cosmos Transfer (depth-conditioned) on agricultural fleet video to generate photorealistic synthetic training images for downstream robot perception. **Key result: autonomous field weeding achieved using a perception model trained with only 1% real data**, with the remainder generated synthetically using this pipeline.

Cosmos Transfer is well suited to this domain because it enforces spatiotemporal structure, allowing visual variation while preserving the spatial relationships critical for centimeter-level precision agriculture. Post-training bridges the domain gap between the foundation model's visual priors and the appearance of row-crop fields — even a modest training run yields strong results on in-distribution crops and promising generalization to unseen ones.

---

## Setup and System Requirements

Follow the [Setup guide](https://nvidia-cosmos.github.io/cosmos-cookbook/tbd.html) for general environment setup instructions, including installing dependencies.

**Hardware:**
- 8× A100 GPUs (e.g., AWS `p4de.24xlarge`)

**Software Dependencies:**
- [Cosmos Transfer 2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5) model weights and training framework
- Depth estimation pipeline (synchronized RGB+D from fleet cameras, or an off-the-shelf monocular depth model such as [DepthAnything V2](https://github.com/DepthAnything/Depth-Anything-V2))

---

## Motivation: The Agricultural Sim2Real Gap

Agricultural robotics operates in one of the most visually diverse and uncontrolled environments in physical AI. A single soybean field changes dramatically across a growing season — from bare soil to canopy closure — and conditions shift with weather, time of day, weed pressure, and disease. Collecting and labeling real data across this full combinatorial space is prohibitively expensive, and much of it is only available during narrow seasonal windows.

Synthetic data generation offers a path forward, but generic world models lack the domain-specific realism needed for precision agriculture. Leaf textures, soil appearance, shadow patterns from row-crop geometry, and the visual signatures of disease are all details that matter for downstream perception.

Post-training Cosmos Transfer on fleet-captured agricultural video closes this domain gap.

```
┌─────────────────────────────────────────────────────────────────┐
│               AGRICULTURAL DOMAIN GAP                           │
│                                                                 │
│  Foundation model priors         Agricultural reality           │
│  ┌──────────────────────┐        ┌──────────────────────┐       │
│  │  Generic scenes      │        │  Row-crop fields     │       │
│  │  Controlled lighting │   ──►  │  Variable sunlight   │       │
│  │  Common objects      │  ???   │  Leaf morphology     │       │
│  │  Limited seasons     │        │  Disease symptoms    │       │
│  └──────────────────────┘        └──────────────────────┘       │
│                                                                 │
│  Post-training on ~3,000 fleet clips bridges this gap.          │
│  Result: 1% real data sufficient for production weeding.        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Zero-Shot Evaluation

> **TODO for authors:** Add before/after visual comparisons showing base Cosmos Transfer 2.5 outputs on agricultural inputs prior to post-training. Required by all cookbook recipes to establish baseline.
>
> Suggested evaluation criteria:
> - Leaf morphology accuracy (species-appropriate shapes)
> - Soil texture realism
> - Shadow consistency with plant geometry
> - Weed diversity and natural placement

---

## Data Preparation

### Source Data

The training data for this recipe consists of fleet-captured video clips (RGB + depth) collected across row-crop agriculture fields. The dataset includes approximately 3,000 clips of roughly 10 seconds each, covering soybean, cotton, and tomato fields with varying weed density, growth stages, and lighting conditions.

| Property | Value |
|---|---|
| Source | Fleet-captured RGB+D video |
| Clips | ~3,000 |
| Duration | ~10 seconds per clip |
| Original resolution | 720p |
| Training resolution | 480p |
| Frame rate | 10 FPS |
| Crops | Soybean, cotton, tomato |
| Depth | Synchronized per-frame from fleet cameras |

> **TODO for authors:** Add input images.

<table style="border: none; border-collapse: collapse;">
<tr style="border: none;">
<td style="border: none; text-align: center;"><img src="assets/image3drender.webp" alt="3D Render" /><br><em>3D Render</em></td>
<td style="border: none; text-align: center;"><img src="assets/deepcontrol.webp" alt="Depth Control Signal" /><br><em>Depth Control Signal</em></td>
</tr>
</table>

### Depth Map Extraction

Depth maps serve as the control signal for Cosmos Transfer. This recipe uses synchronized depth from fleet cameras, but monocular depth estimation can substitute when hardware depth is unavailable.

```python
# Monocular depth with DepthAnything V2 (when fleet depth unavailable)
from transformers import pipeline as hf_pipeline
import numpy as np
import cv2

estimator = hf_pipeline(
    "depth-estimation",
    model="depth-anything/Depth-Anything-V2-Base-hf",
    device="cuda"
)

result = estimator(pil_frame)
depth_np = np.array(result["depth"])

# Normalize to [0, 255] — 0=near, 255=far
depth_u8 = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
```

**Critical:** Normalize depth distributions to match the conventions present in the Cosmos Transfer training data. If your inference-time depth maps use a different normalization than training, outputs will degrade. Apply the same normalization pipeline consistently across training and inference.

### Dataset Formatting

### Dataset Schema

To make this recipe easier to reproduce and adapt, define each training or inference sample using a consistent schema. Even if the source data remains private, documenting the schema allows others to build compatible datasets and reuse the pipeline.

#### Required sample components

| Field | Type | Required | Description |
|---|---|---:|---|
| `sample_id` | string | Yes | Unique identifier shared across RGB video, depth video, and caption JSON |
| `video_path` | string | Yes | Path to the RGB clip used as the primary visual input |
| `depth_path` | string | Yes | Path to the depth-control clip aligned frame-by-frame with the RGB clip |
| `prompt` | string | Yes | Positive text prompt used during training or inference |
| `negative_prompt` | string | Yes | Negative prompt used to suppress common artifacts |
| `crop_type` | string | Recommended | Crop class, for example `soybean`, `cotton`, or `tomato` |
| `lighting` | string | Recommended | Scene lighting descriptor, for example `bright_direct_sunlight` or `overcast` |
| `field_state` | string | Recommended | Field condition descriptor, such as weed density, canopy stage, or soil exposure |
| `camera_view` | string | Recommended | Camera angle or mounting context, for example `angled_row_view` or `front_bumper` |
| `weather` | string | Optional | Weather metadata if available from field logs |
| `split` | string | Recommended | Dataset split: `train`, `val`, or `test` |

#### Example metadata record

```json
{
  "sample_id": "clip_001",
  "video_path": "videos/clip_001.mp4",
  "depth_path": "control_depth/clip_001.mp4",
  "caption_path": "captions/clip_001.json",
  "prompt": "A field view of soybean plants growing in rows under bright direct sunlight. Dense weed growth visible between crop rows. An angled view showing green oval-shaped leaves on slender stems in dry brown soil, with strong sunlight and clear shadows.",
  "negative_prompt": "blurry, overexposed, motion blur, synthetic render",
  "crop_type": "soybean",
  "lighting": "bright_direct_sunlight",
  "field_state": "dense_weeds_between_rows",
  "camera_view": "angled_row_view",
  "weather": "clear",
  "split": "train"
}
```

> **TODO for authors:** did you modify the default "negative_prompt"? Is this json file similar to your experiments?


#### Recommended manifest

In addition to the folder structure below, consider including a top-level manifest such as`dataset_root/metadata.jsonl` with one JSON object per clip. This makes it easier to validate the dataset, generate captions programmatically, and subset by crop type, lighting, or field condition.

```
dataset_root/
├── metadata.jsonl        # One JSON record per sample
├── videos/
├── control_depth/
└── captions/
```

> **TODO for authors:** Is this correct? Confirm which metadata fields Aigen can reliably export from the fleet pipeline. At minimum, please document the fields you already use to generate prompts or organize experiments.

Organize clips into the directory structure expected by the Cosmos Transfer training pipeline:

```
dataset_root/
├── videos/             # 480p, 10 FPS, clips trimmed to ×93 frames
│   ├── clip_001.mp4
│   └── clip_002.mp4
├── control_depth/      # Per-clip depth video, same dims and frame count
│   ├── clip_001.mp4
│   └── clip_002.mp4
└── captions/           # One JSON per clip
    ├── clip_001.json   # {"prompt": "...", "negative_prompt": "..."}
    └── clip_002.json
```

**Latent frames:** The number of latent frames is reduced to 16 (from the default) because agricultural scenes exhibit low temporal variability — crops and soil don't move significantly frame to frame, so fewer latent frames are sufficient to capture the relevant dynamics.

> **TODO for authors:** Clarify whether "16 latent frames" refers to a model config parameter or the physical clip length. If the clip must still be ×93 physical frames, note this explicitly to avoid reader confusion.

### Caption Generation

Captions are generated using a template populated with metadata from the data capture pipeline (crop type, camera angle, field conditions) and augmented with a VLM for natural language descriptions. Weather data can be appended when available.

```python
# Caption template
CAPTION_TEMPLATE = (
    "A field view of {crop_type} plants growing in rows under {lighting} conditions. "
    "{field_state}. "
    "{vlm_description}"
)

# Example output
caption = {
    "prompt": (
        "A field view of soybean plants growing in rows under bright direct sunlight. "
        "Dense weed growth visible between crop rows. "
        "An angled view showing green oval-shaped leaves on slender stems in dry brown soil, "
        "with strong sunlight and clear shadows."
    ),
    "negative_prompt": "blurry, overexposed, motion blur, synthetic render"
}
```

**VLM backend:** Gemini 1.5 Flash is used for natural language description generation. 

> **TODO for authors:** Please confirm the captioning process used in your case.

---

## Post-Training Configuration

### Control Modality: Depth over Edge

Both depth and edge control modalities were evaluated for this domain. **Depth control produced significantly better results.** Edge control post-training showed only slight improvement over the base model.

The explanation is rooted in outdoor agricultural lighting conditions:

| Condition | Effect on Edge Maps | Effect on Depth Maps |
|---|---|---|
| Strong direct sunlight | Shadow boundaries create spurious edges | No effect — depth is illumination-invariant |
| Auto-exposure variation | Causes frame-to-frame edge inconsistency | No effect |
| Overcast diffuse light | Suppresses real edges; gradient ambiguity | No effect |
| Specular highlights (wet leaves) | False positive edges | No effect |

**Recommendation:** For any outdoor physical AI domain with uncontrolled lighting, depth is the preferred control modality over edge.

### Experiment Configuration

> **TODO for authors:** Add the Python config file snippet. Key fields to include:

```python
# cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/experiments/agriculture_sim2real.py
# TODO: authors to provide actual config — template below

experiment = dict(
    data=dict(
        train_dataset=dict(
            data_path="/path/to/dataset_root/",
            control_type="depth",          # depth control for outdoor scenes
            # latent_frames=16,            # TODO: confirm parameter name
        )
    ),
    training=dict(
        max_iter=4000,
        save_iter=500,
        learning_rate=1e-5,               # TODO: confirm value used
    ),
)
```

### Key Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Control type | `depth` | Robust to outdoor lighting variation |
| Training iterations | 4,000 | Loss plateaus early but quality continues improving — train longer |
| Resolution | 480p | Sufficient for downstream weed detection pipeline |
| Latent frames | 16 | Agricultural scenes have low temporal variability |
| Depth guidance scale | 0.8 (inference) | Strong structural adherence with room for surface detail |

> **Training insight:** Loss curves plateau early, but continued training improves generation quality in ways more visible in output samples than in the loss metric. Recommend visual inspection of generated outputs at checkpoints rather than stopping at loss plateau.

### Launch Training

> **TODO for authors:** Add the `torchrun` command used. Template:

```bash
export EXP=agriculture_sim2real
export NUM_GPUS=8

IMAGINAIRE_OUTPUT_ROOT=/large/storage \
torchrun --nproc_per_node=$NUM_GPUS --master_port=12341 \
  -m scripts.train \
  --config=cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/config.py \
  -- experiment=${EXP} job.wandb_mode=disabled
```

### Convert Checkpoint

```bash
# After training, convert DCP format → PyTorch for inference
python scripts/convert_distcp_to_pt.py \
  ${CKPT_DIR}/model \
  ${CKPT_DIR}

# Use checkpoint_ema_bfloat16.pt for all inference runs
```

---

## Inference

### Generating Synthetic Images

At inference time, construct prompts describing the target scene: crop type, growth stage, field conditions, lighting, and optionally disease symptoms. **Prompt precision matters** — specific descriptions of visual features produce better results than vague ones.

**Depth guidance scale:** Use **0.8**. This provides strong structural adherence to the depth control signal while leaving freedom for realistic surface detail and lighting.

**Depth distribution matching:** Inference-time depth maps must use the same normalization as training. Mismatched normalization degrades output quality — apply the same preprocessing pipeline.

> **Note on single-image input:** If you want to transform a single image rather than a video clip, you can loop the image into a video of the appropriate length for inference. However, note that this is an inference-only workaround — results will be lower quality than video-to-video generation because the model receives no temporal motion information. **Never use looped still images as training data**, as this degrades post-training quality. 

> **TODO for authors:** Confirm the experience with still images vs. videos.

```bash
# Inference with post-trained checkpoint
python examples/inference.py \
  --params_file agriculture_spec.json \
  --checkpoint_path ${CKPT_DIR}/checkpoint_ema_bfloat16.pt \
  -o outputs/validation/
```

### Minimal Quickstart (Inference Only)

For readers who want to evaluate the recipe without running post-training, provide a lightweight
inference path using existing Cosmos Transfer weights plus a single RGB image or short clip and a
matching depth control input. This should be the fastest path to a first result.

#### What users need

- Access to **Cosmos Transfer 2.5 model weights**
- Access to the **inference pipeline** from the Cosmos Transfer repository
- One RGB image or short RGB clip from an agricultural scene
- A matching depth map or depth clip generated with fleet depth or a monocular depth model

> **TODO for authors:** Replace the placeholders below with the exact internal or public instructions your team uses to obtain the required model weights and set up the inference environment.

#### Suggested quickstart flow

1. Install the Cosmos Transfer inference environment and verify that the base model weights areavailable locally.
2. Start with **one agricultural RGB image** and estimate depth using the same normalization convention used elsewhere in this recipe.
3. Convert the single image and depth map into short looping videos if the inference pipeline expects video inputs.
4. Run inference with a domain-specific prompt.
5. Export the result as MP4 and optionally convert a short segment to GIF for the recipe page.

#### Minimal directory layout

```
quickstart/
├── inputs/
│   ├── rgb/frame_001.png
│   └── depth/frame_001.png
├── working/
│   ├── rgb_loop.mp4
│   └── depth_loop.mp4
├── prompts/
│   └── agriculture_quickstart.json
└── outputs/
```

#### Example quickstart params file

```json
{
  "prompt": "An angled view of soybean plants growing in rows on a farm field on a bright sunny day. Green oval-shaped soybean leaves on slender stems in dry brown soil, with strong sunlight and clear shadows. Scattered weeds between the rows.",
  "negative_prompt": "blurry, overexposed, motion blur, synthetic render",
  "control_type": "depth",
  "input_video": "quickstart/working/rgb_loop.mp4",
  "control_video": "quickstart/working/depth_loop.mp4",
  "guidance_scale": 0.8
}
```

#### Example workflow

```bash
# 1) Confirm weights are available locally
# TODO for authors: replace with the exact path or download/setup step
export COSMOS_TRANSFER_WEIGHTS=/path/to/cosmos-transfer-2.5-weights

# 2) Generate a depth map for a single agricultural image
python tools/run_depth_estimation.py \
  --input quickstart/inputs/rgb/frame_001.png \
  --output quickstart/inputs/depth/frame_001.png

# 3) Loop the RGB and depth image into short MP4 clips for inference
python tools/make_loop_video.py \
  --input quickstart/inputs/rgb/frame_001.png \
  --output quickstart/working/rgb_loop.mp4 \
  --frames 93 --fps 10

python tools/make_loop_video.py \
  --input quickstart/inputs/depth/frame_001.png \
  --output quickstart/working/depth_loop.mp4 \
  --frames 93 --fps 10

# 4) Run inference with the base or post-trained checkpoint
python examples/inference.py \
  --params_file quickstart/prompts/agriculture_quickstart.json \
  --checkpoint_path ${COSMOS_TRANSFER_WEIGHTS}/checkpoint_ema_bfloat16.pt \
  -o quickstart/outputs/
```

#### Why this matters

A small inference-only entry point helps readers validate three things before investing in full post-training: 

- whether their agricultural prompts are expressive enough
- whether their depth normalization is compatible with the pipeline
- whether the generated visuals are promising enough to justify a larger training run

> **TODO for authors:** If you have an internal helper notebook or a single-command demo, link it here. A notebook or one-click script would make this recipe significantly easier to adopt.

### Example Prompts

**Healthy soybean rows, bright sunlight:**
```
"An angled view of soybean plants growing in rows on a farm field on a bright sunny day. Green oval-shaped soybean leaves on slender stems in dry brown soil, with strong sunlight and clear shadows. Scattered weeds between the rows."
```

![Healthy soybean rows, bright sunlight](assets/healthly-bright.webp)

**Diseased soybeans — yellowing and wilting:**
```
"An angled view of large soybean plants growing in a row on a farm, with scattered weeds. The crop shows signs of disease with yellowing leaves, brown spots, and wilting foliage."
```

![Diseased soybeans](assets/Diseased.webp)

> **TODO for authors:** Did you use negative propmts?

---

## Results

### Generation Quality: Base vs Post-Trained

Side-by-side comparison on the same agricultural inputs:

| Criterion | Base Cosmos Transfer 2.5 | Post-Trained (Aigen) |
|---|---|---|
| Leaf morphology | Generic / incorrect species | Species-appropriate (trifoliate soybean) |
| Soil texture | Generic brown surface | Realistic dry tilled soil |
| Shadow consistency | Disconnected from plant geometry | Aligned with lighting direction |
| Weed diversity | Limited variation | Natural variation in species, size, placement |
| Depth control adherence | Weak | Strong — structure matches depth maps |

<table style="border: none; border-collapse: collapse;">
<tr style="border: none;">
<td style="border: none; text-align: center;"><img src="assets/BaseCosmosTransfer25Checkpoint.webp" alt="Base Cosmos Transfer 2.5 Checkpoint" /><br><em>Base Cosmos Transfer 2.5 Checkpoint</em></td>
<td style="border: none; text-align: center;"><img src="assets/Aigenpost-trainedCosmosTransfer25.webp" alt="Aigen's post-trained Cosmos Transfer 2.5" /><br><em>Aigen's post-trained Cosmos Transfer 2.5</em></td>
</tr>
</table>

### Downstream Impact

The post-trained model has been integrated into Aigen's perception model training pipeline:

> **Autonomous field weeding achieved using a perception model trained with only 1% real data,** with the remainder generated synthetically using this pipeline.

This demonstrates that post-trained synthetic data is realistic enough to substitute for the majority of real-world data collection and human labeling.

### Cross-Domain Generalization

The post-trained checkpoint was applied to crops outside the training distribution (e.g., fruit trees). Results are promising — the model produces reasonably realistic outputs for unseen crops, though not as consistently as for the soybean, cotton, and tomato fields seen during training.

<table style="border: none; border-collapse: collapse;">
<tr style="border: none;">
<td style="border: none; text-align: center;"><img src="assets/Baselinesim2real.webp" alt="Baseline sim2real" /><br><em>Baseline sim2real</em></td>
<td style="border: none; text-align: center;"><img src="assets/aigen-posttrained.webp" alt="Aigen post-trained sim2real" /><br><em>Aigen post-trained sim2real</em></td>
</tr>
</table>

This suggests post-training efficiently activates the foundation model's latent understanding of natural scenes rather than overfitting to the training crops. For production-quality generation across a wider crop variety, a larger post-training run on more diverse data would be needed.

### Disease Condition Synthesis

Prompt engineering enables diseased crop imagery without disease-specific training data. By describing visual symptoms (yellowing leaves, brown spots, wilting, lesions) in the text prompt, the model produces plausible disease presentations. This is particularly valuable because diseased crop data is rare and difficult to collect systematically in the real world.

---

## Conclusion

Post-training Cosmos Transfer on agricultural fleet video is an effective approach to generating photorealistic synthetic data for precision agriculture robotics. Key findings:

- **Depth over edge** for outdoor scenes with uncontrolled lighting — edge maps are corrupted by shadows and variable sunlight; depth is illumination-invariant
- **Train past loss plateau** — visual quality continues improving after the loss curve flattens
- **Generalizes to unseen crops** — post-training extends the foundation model's latent knowledge, not just overfits to training crops
- **Disease synthesis via prompting** — disease data without disease-specific training examples
- **1% real data** is sufficient for a production weeding system

### Next steps

- **Expand crop diversity** — post-train on a broader set of crops and field conditions to improve cross-domain generalization beyond soybean, cotton, and tomato.
- **Scale training iterations** — longer training runs with larger datasets may further improve realism, especially for out-of-distribution crops.
- **Integrate video-level outputs** — explore using generated video clips (rather than extracted frames) directly in temporal perception pipelines.
- **Benchmark against real data** — systematically compare downstream perception metrics (mAP, IoU) across different real-to-synthetic data ratios to quantify the tradeoff.

---

## Citation

```bibtex
@misc{cosmos_cookbook_sim2real_agriculture_2026,
  title={Sim2Real for Agriculture via Cosmos Transfer Post-Training},
  author={Author Names},
  organization={Aigen},
  year={2026},
  howpublished={\url{https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/transfer2_5/singleview_domain_transfer/post_training.html}},
  note={NVIDIA Cosmos Cookbook}
}
```
