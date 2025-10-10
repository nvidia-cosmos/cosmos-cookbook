# Sampson Error Evaluation for Multi-View Video Consistency

## Overview

The `run_cse_tse.py` script evaluates geometrical consistency in multi-view videos using Sampson Error metrics. It measures both temporal consistency (within single camera views over time) and cross-view consistency (between different camera views at the same time).

### Key Metrics

- **Temporal Sampson Error (TSE)**: Measures geometric consistency across consecutive frames within a single camera view
- **Cross-view Sampson Error (CSE)**: Measures geometric consistency between different camera views at the same timestamp

## Environment Setup

### Creating the Conda Environment

The required conda environment can be created using the provided `environment.yml` file:

```bash
# Navigate to the sampson directory
cd scripts/metrics/geometrical_consistency/sampson/

# Create the conda environment from the yml file
conda env create -f environment.yml

# Activate the environment
conda activate sampson
```

### Updating an Existing Environment

If you need to update an existing environment with new dependencies:

```bash
conda env update -f environment.yml --prune
```

### Environment Details

The `environment.yml` includes:

- Python 3.10
- CUDA 12.4.0 support
- Computer vision packages (OpenCV, Kornia)
- Video processing tools (imageio-ffmpeg, av, mediapy)
- Feature matching libraries (pycolmap)
- Scientific computing packages (NumPy, SciPy, matplotlib)

## Quick Start

### 1. Activate Environment

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sampson
```

### 2. Run Evaluation

```bash
python run_cse_tse.py --input /path/to/videos --output /path/to/results --verbose
```

## Script Features

### `run_cse_tse.py` Capabilities

- **Batch Processing**: Process single videos or entire directories of videos
- **Automatic Feature Detection**: Uses SIFT features for correspondence matching between frames/views
- **Fundamental Matrix Estimation**: Computes fundamental matrices using RANSAC for robust estimation
- **Error Visualization**: Generates plots showing error trends across frames
- **Aggregate Statistics**: Computes summary statistics across all processed videos
- **Flexible Input**: Supports custom video file patterns and paths

### Evaluation Process

1. **Feature Extraction**: Extracts SIFT features from each frame of each camera view
2. **Feature Matching**: Matches features between:
   - Consecutive frames (for temporal consistency)
   - Different camera views (for cross-view consistency)
3. **Fundamental Matrix Computation**: Estimates fundamental matrices using matched features
4. **Sampson Error Calculation**: Computes Sampson distance for each matched feature pair
5. **Statistical Aggregation**: Computes mean and median errors per frame and across videos

### Metrics Detail

#### Temporal Sampson Error (TSE)

- Evaluated for views: `front`, `cross_left`, `cross_right`
- Measures drift and jitter in individual camera streams
- Lower values indicate better temporal consistency

#### Cross-view Sampson Error (CSE)

- Evaluated for view pairs: `front-cross_right`, `front-cross_left`
- Measures multi-view geometric consistency
- Lower values indicate better cross-view alignment

## Video Format Requirements

Videos must be in **2x3 grid format** (MP4):

```
[LEFT    FRONT    RIGHT ]
[REAR_L  REAR_T   REAR_R]
```

## Output Files

### Per-Video Results

For each processed video, the script generates:

#### 1. `cse_tse/{video_id}.json`

Contains detailed numerical results:

```json
{
  "video_path": "/path/to/video.mp4",
  "clip_id": "video_id",
  "results": {
    "T": {  // Temporal errors
      "front": {
        "mean": 2.345,
        "median": 2.123,
        "frame_values": [...]  // Per-frame median values
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

#### 2. `cse_tse/{video_id}.png`

Visualization plot showing:

- Solid lines: Temporal Sampson Errors (TSE) for each view
- Dashed lines: Cross-view Sampson Errors (CSE) for view pairs
- Y-axis: Error in √pixels (capped at 10 for visibility)
- X-axis: Frame number

### Aggregate Statistics

#### `aggregate_stats.json`

Summary statistics across all processed videos:

```json
{
  "num_videos": 10,
  "temporal": {
    "front": {
      "mean": 2.5, "median": 2.3, "std": 0.8,
      "min": 1.2, "max": 4.5, "count": 10
    },
    "overall": {
      "mean": 2.6, "median": 2.4, "std": 0.9
    }
  },
  "cross_view": {
    "front-cross_right": {...},
    "overall": {...}
  }
}
```

## Command Line Options

```bash
python run_cse_tse.py [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--input PATH` | Input video file or directory containing videos | Required |
| `--output PATH` | Output directory for evaluation results | `./eval-output` |
| `--pattern STR` | File pattern for filtering video files (e.g., `*.mp4`, `*_gen.mp4`) | `*.mp4` |
| `--verbose` | Enable verbose output with detailed processing information | Off |

### Usage Examples

```bash
# Process a single video
python run_cse_tse.py --input video.mp4 --output results/

# Process all MP4 files in a directory
python run_cse_tse.py --input /path/to/videos/ --output results/

# Process only generated videos with verbose output
python run_cse_tse.py --input videos/ --pattern "*_gen.mp4" --verbose

# Use default output directory
python run_cse_tse.py --input videos/
```

## Interpreting Results

### Error Value Ranges

- **Excellent** (< 1.0 pixels): Very high geometric consistency
- **Good** (1.0 - 3.0 pixels): Acceptable consistency for most applications
- **Fair** (3.0 - 5.0 pixels): Noticeable inconsistencies, may need improvement
- **Poor** (> 5.0 pixels): Significant geometric errors

### What the Metrics Tell You

- **High TSE**: Indicates temporal instability (flickering, jitter, or drift)
- **High CSE**: Indicates poor multi-view consistency (misaligned views, incorrect geometry)
- **Frame value spikes**: Sudden increases suggest problematic frames or scene changes

## Troubleshooting

### Common Issues

1. **"Error loading data" message**
   - Ensure video is in correct 2x3 grid format
   - Check video codec compatibility (H.264 recommended)

2. **High error values**
   - May indicate low-quality generation or incorrect camera parameters
   - Check if input videos have sufficient texture for feature matching

3. **Missing results for some views**
   - Can occur with textureless or blurry regions
   - Insufficient feature matches between frames/views
