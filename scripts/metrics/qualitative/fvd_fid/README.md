# Video Metrics Evaluation Scripts

This folder contains scripts for evaluating video generation quality using Fréchet Inception Distance (FID) and Fréchet Video Distance (FVD) metrics.

## Scripts

### 1. `compute_fid_single_view.py`

Computes the **Fréchet Inception Distance (FID)** between predicted and ground truth videos by comparing individual frames.

FID measures the quality and diversity of generated images by comparing the distribution of features extracted from a pre-trained Inception network.

### 2. `compute_fvd_single_view.py`

Computes the **Fréchet Video Distance (FVD)** between predicted and ground truth videos by considering temporal consistency.

FVD extends FID to videos by using an I3D network that captures both spatial and temporal features, making it suitable for evaluating video generation quality.

## Installation

Install the required dependencies:

```bash
# Install all dependencies at once
pip install -r scripts/metrics/requirements.txt

# Or install individually:
# For FID computation
pip install decord torchmetrics[image] torch-fidelity

# For FVD computation
pip install cd-fvd decord einops scipy


## Usage

### FID Computation

```bash
python compute_fid_single_view.py \
    --pred_video_paths "./path/to/predicted/*.mp4" \
    --gt_video_paths "./path/to/ground_truth/*.mp4" \
    --num_frames 57 \
    --output_file fid_results.json
```

**Arguments:**

- `--pred_video_paths` (required): Path pattern for predicted videos (supports glob patterns)
- `--gt_video_paths` (required): Path pattern for ground truth videos (supports glob patterns)
- `--num_frames` (optional): Number of frames to use from each video (default: all frames)
- `--output_file` (optional): Output JSON file for results (default: `fid_results.json`)

### FVD Computation

```bash
python compute_fvd_single_view.py \
    --pred_video_paths "./path/to/predicted/*.mp4" \
    --gt_video_paths "./path/to/ground_truth/*.mp4" \
    --num_frames 57 \
    --batch_size 8 \
    --target_size 224 224 \
    --output_file fvd_results.json
```

**Arguments:**

- `--pred_video_paths` (required): Path pattern for predicted videos (supports glob patterns)
- `--gt_video_paths` (required): Path pattern for ground truth videos (supports glob patterns)
- `--num_frames` (optional): Number of frames to use from each video (default: all frames)
- `--batch_size` (optional): Batch size for FVD computation (default: 8)
- `--target_size` (optional): Target size for resizing frames as height width (default: 224 224)
- `--output_file` (optional): Output JSON file for results (default: `fvd_results.json`)

## Examples

### Example 1: Evaluate all videos in directories

```bash
# Compute FID for all MP4 videos
python compute_fid_single_view.py \
    --pred_video_paths "./generated_videos/*.mp4" \
    --gt_video_paths "./real_videos/*.mp4"

python scripts/metrics/compute_fid_single_view.py \
    --pred_video_paths "./av_multiview_eval_10/videos_cosmos1_single_views/all_views/*.mp4" \
    --gt_video_paths "./av_multiview_eval_10/videos_gt_single_views/all_views/*.mp4"


# Compute FVD with custom settings
python compute_fvd_single_view.py \
    --pred_video_paths "./generated_videos/*.mp4" \
    --gt_video_paths "./real_videos/*.mp4" \
    --num_frames 100 \
    --batch_size 16
```

### Example 2: Evaluate specific video formats

```bash
# Evaluate AVI videos
python compute_fid_single_view.py \
    --pred_video_paths "./outputs/**/*.avi" \
    --gt_video_paths "./references/**/*.avi"
```

### Example 3: Run both metrics

```bash
# Run both FID and FVD evaluation
VIDEO_PRED="./model_outputs/*.mp4"
VIDEO_GT="./ground_truth/*.mp4"

python compute_fid_single_view.py \
    --pred_video_paths "$VIDEO_PRED" \
    --gt_video_paths "$VIDEO_GT" \
    --output_file results_fid.json

python compute_fvd_single_view.py \
    --pred_video_paths "$VIDEO_PRED" \
    --gt_video_paths "$VIDEO_GT" \
    --output_file results_fvd.json
```

## Output Format

Both scripts generate JSON files with evaluation results:

### FID Output (`fid_results.json`)

```json
{
  "FID": 45.23,
  "num_pred_videos": 100,
  "num_gt_videos": 100,
  "num_frames_per_video": 57,
  "total_pred_frames": 5700,
  "total_gt_frames": 5700,
  "pred_video_pattern": "./predicted/*.mp4",
  "gt_video_pattern": "./ground_truth/*.mp4"
}
```

### FVD Output (`fvd_results.json`)

```json
{
  "FVD": 123.45,
  "num_pred_videos": 100,
  "num_gt_videos": 100,
  "num_frames_per_video": 57,
  "batch_size": 8,
  "target_size": [224, 224],
  "pred_video_pattern": "./predicted/*.mp4",
  "gt_video_pattern": "./ground_truth/*.mp4"
}
```

## Important Notes

1. **Video Count Matching**: Both scripts require the same number of predicted and ground truth videos. They will raise an error if the counts don't match.

2. **Memory Usage**:
   - FID loads all frames into memory, so be careful with very long videos or large datasets
   - FVD processes videos in batches to manage memory usage
   - Adjust `--batch_size` for FVD if you encounter memory issues

3. **Video Ordering**: Videos are sorted alphabetically by filename. Ensure your predicted and ground truth videos have matching names or consistent ordering.

4. **Supported Formats**: The scripts use `decord` for video reading, which supports most common video formats (MP4, AVI, MOV, etc.)

5. **GPU Usage**: Both scripts automatically use GPU if available (`cuda`), otherwise fall back to CPU.

**FID:**

```bibtex
@inproceedings{heusel2017gans,
  title={GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium},
  author={Heusel, Martin and Ramsauer, Hubert and Unterthiner, Thomas and Nessler, Bernhard and Hochreiter, Sepp},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017}
}
```

**FVD:**

```bibtex
@inproceedings{unterthiner2018towards,
  title={Towards Accurate Generative Models of Video: A New Metric & Challenges},
  author={Unterthiner, Thomas and van Steenkiste, Sjoerd and Kurach, Karol and Marinier, Raphael and Michalski, Marcin and Gelly, Sylvain},
  booktitle={arXiv preprint arXiv:1812.01717},
  year={2018}
}
```
