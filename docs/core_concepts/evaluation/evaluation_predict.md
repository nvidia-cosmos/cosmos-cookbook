# Model Evaluation Predict

This page focuses on evaluation of Predict (generative video) models. It introduces commonly used quality metrics, explains what each metric measures and how it works, and then provides source implementations with step‑by‑step instructions to run them.

## Key Terms

| Term | Definition |
|------|------------|
| **FID** (Fréchet Inception Distance) | A metric that measures the similarity between two sets of images by comparing their feature distributions extracted from a pre-trained neural network. |
| **FVD** (Fréchet Video Distance) | An extension of FID for videos that captures both spatial (appearance) and temporal (motion) quality by comparing video feature distributions. |
| **Fréchet Distance** | A statistical measure of similarity between two probability distributions; used in FID/FVD to quantify how "far apart" generated content is from real content. |
| **Sampson Error** | A geometric error metric that measures the distance from matched keypoints to their corresponding epipolar lines; used to evaluate multi-view consistency. |
| **TSE** (Temporal Sampson Error) | Sampson error computed between consecutive frames within a single camera view; measures temporal stability. |
| **CSE** (Cross-view Sampson Error) | Sampson error computed between simultaneous frames from different camera views; measures multi-view geometric alignment. |
| **Epipolar Geometry** | The geometric relationship between two camera views of the same 3D scene; defines constraints on where corresponding points can appear. |

## Overview: Quality Metrics for Predict Models

Use these metrics to evaluate generative video models (Predict):

| Metric | Measures | Use Case | Better Direction |
|--------|----------|----------|------------------|
| **FID** | Image realism and diversity | Single-frame quality assessment | Lower ↓ |
| **FVD** | Spatio-temporal video quality | Overall video quality with motion coherence | Lower ↓ |
| **TSE** | Temporal geometric consistency | Detecting flickering, jitter, or drift within views | Lower ↓ |
| **CSE** | Cross-view geometric consistency | Multi-camera alignment and 3D consistency | Lower ↓ |

For VLM-based assessment details, refer to [Cosmos Reason as Reward](reason_as_reward.md) and the [Cosmos Reason Benchmark Example](https://github.com/nvidia-cosmos/cosmos-reason1/blob/main/examples/benchmark/README.md).

## Metrics Cheat Sheet

Use this quick reference to interpret your evaluation scores:

### Video Quality (FID/FVD)

| Rating | FID Score | FVD Score | Interpretation |
|--------|-----------|-----------|----------------|
| Excellent | < 30 | < 100 | High-quality generation, close to ground truth |
| Good | 30 – 50 | 100 – 200 | Acceptable quality for most applications |
| Fair | 50 – 100 | 200 – 400 | Noticeable quality gaps; consider improvements |
| Poor | > 100 | > 400 | Significant quality issues; requires attention |

### Geometric Consistency (Sampson Error)

| Rating | TSE/CSE (pixels) | Interpretation |
|--------|------------------|----------------|
| Excellent | < 1.0 | Very high geometric consistency |
| Good | 1.0 – 3.0 | Acceptable for most applications |
| Fair | 3.0 – 5.0 | Noticeable inconsistencies; may need improvement |
| Poor | > 5.0 | Significant geometric errors |

> **Note**: These thresholds are general guidelines. Acceptable ranges may vary depending on your specific use case, dataset characteristics, and downstream application requirements.

## Video Quality Metrics (FID/FVD)

This metric evaluates the quality of generated videos using standardized metrics that compare predicted videos against ground truth.

### Step 1: Install Metrics Dependencies

```bash
# Install all metrics dependencies
pip install -r scripts/metrics/requirements.txt

# Or install individually:
pip install decord torchmetrics[image] torch-fidelity  # For FID
pip install cd-fvd decord einops scipy                  # For FVD
```

### Step 2: Compute FID (Fréchet Inception Distance)

FID measures the quality and diversity of generated frames by comparing feature distributions from a pre‑trained Inception network.

#### What this metric measures

- Distance between distributions of real vs generated image features (lower is better)

#### How this metric works (high level)

- Extract 2048‑D features with InceptionV3 on real and generated frames.
- Fit Gaussians (means μ and covariances Σ) and compute Fréchet distance:

```
FID = ||μ_real - μ_gen||² + Tr(Σ_real + Σ_gen - 2√(Σ_real × Σ_gen))
```

#### Example Command

Run the following command to compute FID:

```bash
python scripts/metrics/compute_fid_single_view.py \
    --pred_video_paths "./path/to/predicted/*.mp4" \
    --gt_video_paths "./path/to/ground_truth/*.mp4" \
    --num_frames 57 \
    --output_file fid_results.json
```

#### Arguments

- `--pred_video_paths`, `--gt_video_paths`, `--num_frames`, `--output_file`

### Step 3: Compute FVD (Fréchet Video Distance)

FVD extends FID to videos by evaluating spatio‑temporal features, capturing both appearance and motion.

#### What this metric measures

- Global video quality including temporal coherence and motion dynamics
- More faithful to user perception than purely frame‑based metrics

#### How this metric works (high level)

- Extract video features for real and generated clips.
- Fit Gaussians and compute Fréchet distance analogously to FID.

#### Best practices

- Use consistent clip length, frame rate, and spatial size.
- Tune `--batch_size` to balance throughput and memory.
- Ensure paired, alphabetically sorted file lists (pred ↔ GT).

#### Example Command

Run the following command to compute FVD:

```bash
python scripts/metrics/compute_fvd_single_view.py \
    --pred_video_paths "./path/to/predicted/*.mp4" \
    --gt_video_paths "./path/to/ground_truth/*.mp4" \
    --num_frames 57 \
    --batch_size 8 \
    --target_size 224 224 \
    --output_file fvd_results.json
```

#### Additional arguments

- `--batch_size`: default: 8
- `--target_size`: default: 224 224

### Output Format

#### FID Results (`fid_results.json`)

```json
{
  "FID": 45.23,
  "num_pred_videos": 100,
  "num_gt_videos": 100,
  "num_frames_per_video": 57,
  "total_pred_frames": 5700,
  "total_gt_frames": 5700
}
```

#### FVD Results (`fvd_results.json`)

```json
{
  "FVD": 123.45,
  "num_pred_videos": 100,
  "num_gt_videos": 100,
  "num_frames_per_video": 57,
  "batch_size": 8,
  "target_size": [224, 224]
}
```

### Important Notes

- **Matching Video Counts**: Both metrics require equal numbers of predicted and ground truth videos.
- **Video Ordering**: Videos are sorted alphabetically—ensure consistent naming between predicted and GT sets
- **Memory Management**: FID loads all frames into memory; FVD processes in batches—adjust the `--batch_size` parameter for FVD if needed
- **GPU Usage**: These scripts use the GPU if available; otherwise, they fall back to the CPU.
- **Supported Formats**: These scripts support common video formats (MP4, AVI, MOV) via `decord`.

### Example: Complete Evaluation Pipeline

```bash
# Define video paths
VIDEO_PRED="./generated_videos/*.mp4"
VIDEO_GT="./ground_truth_videos/*.mp4"

# Run FID evaluation
python scripts/metrics/compute_fid_single_view.py \
    --pred_video_paths "$VIDEO_PRED" \
    --gt_video_paths "$VIDEO_GT" \
    --output_file evaluation_fid.json

# Run FVD evaluation
python scripts/metrics/compute_fvd_single_view.py \
    --pred_video_paths "$VIDEO_PRED" \
    --gt_video_paths "$VIDEO_GT" \
    --output_file evaluation_fvd.json
```

## Geometrical Consistency Metrics (Sampson Error)

These metrics evaluate the geometric consistency of multi‑view videos and diagnose temporal instability and cross‑view misalignment that are not captured by FID/FVD.

This metric has the following benefits:

- Lower errors, leading to smoother motion (temporal consistency) and better multi‑view geometry
- Useful for multi‑camera or stitched 2×3 grid outputs

### Step 1: Setup Conda Environment for Sampson Metrics

Create and activate the dedicated conda environment for Sampson error evaluation:

```bash
# Navigate to the sampson metrics directory
cd scripts/metrics/geometrical_consistency/sampson/

# Create the conda environment from the yml file
conda env create -f environment.yml

# Activate the environment
conda activate sampson
```

> **Note**: The environment includes Python 3.10, CUDA 12.4.0 support, and the necessary vision packages (OpenCV, Kornia, PyColmap).

### Step 2: Prepare Multi‑View Videos

Ensure your videos are in the required 2×3 grid format (in MP4):

```
[LEFT    FRONT    RIGHT ]
[REAR_L  REAR_T   REAR_R]
```

### Step 3: Compute Sampson Error Metrics

Run the evaluation script to compute both Temporal Sampson Error (TSE) and Cross‑view Sampson Error (CSE):

```bash
# Single video
python scripts/metrics/geometrical_consistency/sampson/run_cse_tse.py \
    --input path/to/video.mp4 \
    --output ./sampson_results \
    --verbose

# Directory of videos
python scripts/metrics/geometrical_consistency/sampson/run_cse_tse.py \
    --input ./path/to/videos/ \
    --pattern "*.mp4" \
    --output ./sampson_results
```

#### Arguments

- `--input`, `--output`, `--pattern`, `--verbose`

### Metrics Explanation

#### Sampson Error (overview)

- Provides airst‑order approximation of point‑to‑epipolar‑line distance given matched keypoints and the fundamental matrix; lower is better.

#### Temporal Sampson Error (TSE)

- Measures geometric consistency across consecutive frames within a view; lower values indicate smoother motion and fewer temporal artifacts.

#### Cross‑view Sampson Error (CSE)

- Measures geometric consistency across simultaneous views; lower values indicate better multi‑view alignment.

### Output Format

#### Per‑Video Results (`cse_tse/{video_id}.json`)

```json
{
  "video_path": "/path/to/video.mp4",
  "clip_id": "video_id",
  "results": {
    "T": {  // Temporal errors
      "front": {
        "mean": 2.345,
        "median": 2.123,
        "frame_values": [...]
      },
      "cross_left": {...},
      "cross_right": {...}
    },
    "C": {  // Cross-view errors
      "front-cross_right": {
        "mean": 3.456,
        "median": 3.234,
        "frame_values": [...]
      },
      "front-cross_left": {...}
    }
  }
}
```

#### Aggregate Statistics (`aggregate_stats.json`)

```json
{
  "num_videos": 10,
  "temporal": {
    "front": {"mean": 2.5, "median": 2.3, "std": 0.8},
    "overall": {"mean": 2.6, "median": 2.4, "std": 0.9}
  },
  "cross_view": {
    "front-cross_right": {...},
    "overall": {...}
  }
}
```

#### Visualization Plots (`cse_tse/{video_id}.png`)

Generated plots show the following:

- **Solid lines**: Temporal Sampson Errors (TSE) for each view
- **Dashed lines**: Cross-view Sampson Errors (CSE) for view pairs
- **Y-axis**: Error in √pixels (capped at 10 for visibility)
- **X-axis**: Frame number

### Interpreting Results

**Error Value Ranges:**

- **Excellent** (< 1.0 pixels): Very high geometric consistency
- **Good** (1.0 - 3.0 pixels): Acceptable consistency for most applications
- **Fair** (3.0 - 5.0 pixels): Noticeable inconsistencies; may need improvement
- **Poor** (> 5.0 pixels): Significant geometric errors

**What High Errors Indicate:**

- **High TSE**: Temporal instability (flickering, jitter, or drift in individual views)
- **High CSE**: Poor multi-view consistency (misaligned views, incorrect geometry)
- **Frame spikes**: Sudden error increases suggest problematic frames or scene changes

### Example: Complete Sampson Evaluation Pipeline

```bash
# Setup environment
conda activate sampson

# Define video paths
VIDEO_DIR="./generated_multi_view_videos"
OUTPUT_DIR="./evaluation_sampson"

# Run Sampson error evaluation on all generated videos
python scripts/metrics/geometrical_consistency/sampson/run_cse_tse.py \
    --input "$VIDEO_DIR" \
    --pattern "*_gen.mp4" \
    --output "$OUTPUT_DIR" \
    --verbose

# Results will be in:
# - $OUTPUT_DIR/cse_tse/*.json (per-video metrics)
# - $OUTPUT_DIR/cse_tse/*.png (visualization plots)
# - $OUTPUT_DIR/aggregate_stats.json (summary statistics)
```

### Important Notes

Note the following about Geometrical Consistency metrics:

- **Video Format**: Videos must be in 2×3 grid format, with 6 camera views.
- **Feature Matching**: These metrics use SIFT features for correspondence matching between frames/views.
- **Memory Usage**: Sufficient RAM is required for feature extraction and matching.
- **GPU Support**: These metrics automatically use GPU acceleration when available for faster processing.
