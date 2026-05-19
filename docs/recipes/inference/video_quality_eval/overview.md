# Video Quality Evaluation Pipeline

> **Author:** [Pranav Raj](https://github.com/pranavrajsb)
> **Organization:** Independent

## Overview

| **Model** | **Workload** | **Use Case** |
|-----------|--------------|--------------|
| Any Cosmos generation model | Post-inference evaluation | Automated quality gate for generated videos |

Every Cosmos generation recipe ends at producing a video. This recipe answers the question that comes next: **is what I generated any good?**

It provides a standalone, model-agnostic evaluation pipeline that runs as the final step after any Cosmos Predict, Transfer, or Reason workflow. Rather than trusting a single metric, it combines signals from three complementary evaluation layers — perceptual similarity, semantic text alignment, and VLM-based physical plausibility — and aggregates them into one score with pass/reject classification and failure reason attribution.

The key design principle is **multi-signal evaluation**: a video that scores well on frame similarity but fails the physics check is treated differently from one that fails both. Each layer provides a distinct, orthogonal signal, and their combination is more reliable than any one alone.

## What This Pipeline Adds

The Cosmos Cookbook already documents individual evaluation tools:

- FVD / FID (distributional quality) → [`scripts/metrics/qualitative/fvd_fid/`](../../../../scripts/metrics/qualitative/fvd_fid/)
- Geometric consistency (Sampson Error) → [`scripts/metrics/geometrical_consistency/`](../../../../scripts/metrics/geometrical_consistency/)
- Cosmos Reason as reward model → [`scripts/evaluation/cosmos-reason1-reward-7b/`](../../../../scripts/evaluation/cosmos-reason1-reward-7b/)

This recipe adds what is missing: **CLIP text-video alignment**, **LPIPS perceptual similarity**, **multi-query VLM judging**, and an **aggregate scorer** that unifies all signals into a single actionable verdict.

## Pipeline Architecture

```
Generated videos
       │
       ├─── Layer 1: Perceptual ──────── LPIPS + SSIM + PSNR  (requires reference videos)
       │
       ├─── Layer 2: Semantic ────────── CLIP text-video alignment (requires text prompts)
       │
       └─── Layer 3: VLM Judge ──────── Cosmos Reason 1 × 3 queries (requires checkpoint)
                                                │
                                        Weighted aggregate
                                                │
                               ┌────────────────┼────────────────┐
                            PASS           BORDERLINE          REJECT
                        (score ≥ 0.65)  (0.35–0.65)       (score < 0.35)
                                                │
                                       Failure reason attribution
                                       (which layer(s) drove the low score)
```

**All three layers are optional.** Any subset works — the script activates only the layers you configure and re-weights accordingly.

## Evaluation Layers

### Layer 1 — Perceptual Similarity

Compares generated frames directly against reference (ground truth) frames using three metrics:

| Metric | Measures | Direction |
|--------|----------|-----------|
| **LPIPS** | Perceptual feature distance (AlexNet) | Lower is better |
| **SSIM** | Structural similarity (luminance, contrast, structure) | Higher is better |
| **PSNR** | Peak signal-to-noise ratio | Higher is better |

The three are averaged into a single perceptual score after normalisation to [0, 1].

**When to use:** When you have ground truth reference videos (e.g., evaluating a fine-tuned model against a held-out test set).

**When to skip:** For pure generation tasks (text-to-video, image-to-world) where no reference exists.

### Layer 2 — Semantic Text Alignment

Uses OpenAI CLIP (ViT-B/32) to measure whether the generated video is semantically consistent with its text prompt. Frames are sampled uniformly, encoded, and their mean cosine similarity to the encoded prompt is computed.

| CLIP Cosine Similarity | Interpretation |
|------------------------|----------------|
| > 0.35 | Strong alignment with the text prompt |
| 0.20 – 0.35 | Moderate alignment |
| < 0.20 | Weak alignment |

**When to use:** Any text-conditioned generation (text-to-video, any recipe that takes a language prompt as input).

**When to skip:** Generation tasks with non-text conditioning only (e.g., depth-only or pose-only Transfer).

### Layer 3 — VLM Judge (Cosmos Reason 1)

Uses the [Cosmos Reason 1-7B-Reward](https://huggingface.co/nvidia/Cosmos-Reason1-7B-Reward) model to assess physical plausibility. Rather than a single query, the pipeline asks **three different questions** about the same video and aggregates their scores — a multi-judge pattern that reduces sensitivity to any one prompt:

1. *"Does the video contain any anomalies or artifacts?"*
2. *"Are there any violations of physical laws such as gravity or collision in this video?"*
3. *"Does any object in the video behave in a physically implausible way?"*

Each query yields a 0–1 score (probability of *no* anomaly). The mean across all three is the layer score.

**When to use:** Any generation recipe where physical realism matters (robotics trajectories, AV scene generation, physics simulations).

**When to skip:** Artistic/stylized generation where physical plausibility is not a requirement.

## Setup

### Step 1: Install Dependencies

```bash
pip install -r scripts/examples/video_quality_eval/requirements.txt
```

For CLIP:

```bash
pip install git+https://github.com/openai/CLIP.git
```

### Step 2: Download Cosmos Reason 1 (Layer 3 only)

```bash
huggingface-cli download nvidia/Cosmos-Reason1-7B-Reward \
    --local-dir ./checkpoints/Cosmos-Reason1-7B-Reward \
    --token <YOUR_HF_TOKEN>
```

> **Note:** Requires a HuggingFace account and a token with access to the `nvidia/Cosmos-Reason1-7B-Reward` model.

## Usage

### All Three Layers

```bash
python scripts/examples/video_quality_eval/evaluate.py \
    --pred_dir  ./generated_videos \
    --ref_dir   ./reference_videos \
    --prompts   prompts.txt \
    --reason_ckpt ./checkpoints/Cosmos-Reason1-7B-Reward \
    --output    quality_report.json
```

### VLM + Semantic Only (no reference videos)

Useful for text-to-video generation where no ground truth exists:

```bash
python scripts/examples/video_quality_eval/evaluate.py \
    --pred_dir  ./generated_videos \
    --prompts   prompts.txt \
    --reason_ckpt ./checkpoints/Cosmos-Reason1-7B-Reward \
    --output    quality_report.json
```

### Perceptual Only (offline, no model downloads)

Useful for quick regression checks when comparing model checkpoints:

```bash
python scripts/examples/video_quality_eval/evaluate.py \
    --pred_dir ./generated_videos \
    --ref_dir  ./reference_videos \
    --output   quality_report.json
```

### Text Prompt Format

The `--prompts` file supports two formats:

**Plain text** (one prompt per line, matched to videos by sort order):
```
a robot arm picking up a red cube
a vehicle navigating through rain
a humanoid walking across a warehouse floor
```

**JSON mapping** (stem → prompt, order-independent):
```json
{
  "video_001": "a robot arm picking up a red cube",
  "video_002": "a vehicle navigating through rain"
}
```

## Output

### Console Summary

```
============================================================
VIDEO QUALITY EVALUATION SUMMARY
============================================================
  Total videos evaluated : 50
  PASS                   : 34 (68.0%)
  BORDERLINE             : 11
  REJECT                 : 5
  Mean aggregate score   : 0.693
  Mean perceptual  score : 0.721
  Mean semantic    score : 0.658
  Mean vlm         score : 0.702
============================================================
```

### JSON Report (`quality_report.json`)

```json
{
  "summary": {
    "total_videos": 50,
    "pass": 34,
    "borderline": 11,
    "reject": 5,
    "pass_rate": 0.68,
    "mean_aggregate_score": 0.693,
    "mean_perceptual_score": 0.721,
    "mean_semantic_score": 0.658,
    "mean_vlm_score": 0.702,
    "config": {
      "threshold_pass": 0.65,
      "threshold_reject": 0.35,
      "weight_perceptual": 0.30,
      "weight_semantic": 0.35,
      "weight_vlm": 0.35
    }
  },
  "videos": [
    {
      "video": "generated_videos/video_001.mp4",
      "aggregate_score": 0.812,
      "verdict": "PASS",
      "active_layers": ["perceptual", "semantic", "vlm"],
      "failure_reasons": [],
      "perceptual": {
        "score": 0.841,
        "lpips": 0.123,
        "lpips_score": 0.877,
        "ssim": 0.814,
        "psnr_db": 28.4,
        "psnr_score": 0.710
      },
      "semantic": {
        "score": 0.776,
        "cosine_similarity": 0.294,
        "clip_score": 0.776
      },
      "vlm": {
        "score": 0.820,
        "query_scores": {
          "Does the video contain any anomalies or artifacts?": 0.831,
          "Are there any violations of physical laws such as gravity or collision in this video?": 0.814,
          "Does any object in the video behave in a physically implausible way?": 0.815
        },
        "aggregate": 0.820
      }
    }
  ]
}
```

## Tuning the Pipeline

### Adjusting Thresholds

The defaults (`--threshold_pass 0.65`, `--threshold_reject 0.35`) are conservative starting points. Calibrate them on a small labelled set from your specific domain:

```bash
# Stricter quality gate
python evaluate.py --pred_dir ./generated --threshold_pass 0.75 --threshold_reject 0.50 ...

# More permissive (higher yield)
python evaluate.py --pred_dir ./generated --threshold_pass 0.55 --threshold_reject 0.25 ...
```

### Adjusting Layer Weights

If reference videos are unavailable and only two layers are active, weights are automatically re-normalised. To manually emphasise the VLM judge:

```bash
python evaluate.py \
    --pred_dir ./generated \
    --prompts prompts.txt \
    --reason_ckpt ./checkpoints/Cosmos-Reason1-7B-Reward \
    --weight_semantic 0.30 \
    --weight_vlm 0.70 \
    --output report.json
```

## Integration with Other Recipes

### As a Post-Generation Filter

After any Cosmos generation recipe, pipe outputs through this pipeline and filter by verdict:

```python
import json
report = json.load(open("quality_report.json"))
passed = [v["video"] for v in report["videos"] if v["verdict"] == "PASS"]
borderline = [v["video"] for v in report["videos"] if v["verdict"] == "BORDERLINE"]
rejected = [v["video"] for v in report["videos"] if v["verdict"] == "REJECT"]
print(f"Retained {len(passed)} / {report['summary']['total_videos']} videos")
```

### Best-of-N Sampling

Generate multiple candidates per prompt (different seeds), evaluate all, and keep the highest-scoring one:

```bash
# Generate N candidates
for seed in 0 1 2 3 4; do
    python generate.py --seed $seed --output generated/video_001_seed${seed}.mp4
done

# Evaluate all and select best
python scripts/examples/video_quality_eval/evaluate.py \
    --pred_dir ./generated \
    --prompts  prompts.txt \
    --reason_ckpt ./checkpoints/Cosmos-Reason1-7B-Reward \
    --output report.json
```

Then select the video with the highest `aggregate_score` per prompt prefix.

### Distributional Quality (FVD/FID)

This pipeline evaluates each video individually. For distributional quality across the whole generated set, use the existing FVD/FID scripts first:

```bash
python scripts/metrics/qualitative/fvd_fid/compute_fvd_single_view.py \
    --pred_video_paths "./generated/*.mp4" \
    --gt_video_paths   "./reference/*.mp4" \
    --output_file      fvd_results.json
```

A complete quality workflow: FVD/FID on the set → this pipeline on individual videos → filter by verdict.

## System Requirements

| Component | Requirement |
|-----------|-------------|
| GPU | CUDA-enabled, ≥ 8 GB VRAM (Layer 1 + 2); ≥ 24 GB VRAM (Layer 3) |
| CPU fallback | Supported for Layers 1 and 2; Layer 3 is impractical without GPU |
| Processing time (Layer 3) | ~20–40 seconds per video on an A100 |

## Resources

- **[Cosmos Reason 1 GitHub](https://github.com/nvidia-cosmos/cosmos-reason1)** — Model source and training details
- **[Cosmos Reason 1-7B-Reward on HuggingFace](https://huggingface.co/nvidia/Cosmos-Reason1-7B-Reward)** — Model weights
- **[Evaluation Core Concepts](../../../core_concepts/evaluation/overview.md)** — Background on FVD, FID, and Sampson metrics
- **[GR00T-Dreams Recipe](../../end2end/gr00t-dreams/post-training.md)** — Example of Cosmos Reason 2 used for rejection sampling in a robot trajectory context
- **[OpenAI CLIP](https://github.com/openai/CLIP)** — CLIP model and pretraining details