# Sampling Visualization Tools

This directory contains tools for video sampling and visualization, supporting both local and S3 paths.

install the prerequisites (if not present):

```bash
conda install -c conda-forge opencv
pip install streamlit
```

---

## 1. Grid Preview Video Generation (`grid_preview_generation.py`)

This script recursively scans a directory (local or S3) for `.mp4` files, randomly samples them, and generates a grid preview video like below.

![Grid Preview Example](grid_preview.png){ width=600px }

**Usage:**

```bash
python scripts/curation/tools/sampling_visualization/grid_preview_generation.py \
    --input_dir <input_dir_or_s3> \
    --output_video <output_video_path_or_s3> \
    [--cols 10] [--rows 10] [--video_length 5] [--fps 30]
```

**Arguments:**

- `--input_dir`: Input directory or S3 path (required)
- `--output_video`: Output video path (local or S3, required)
- `--cols`: Number of columns in the grid (default: 10)
- `--rows`: Number of rows in the grid (default: 10)
- `--video_length`: Output video duration in seconds (default: 5)
- `--fps`: Frames per second (default: 30)

**Examples:**

```bash
python scripts/curation/tools/sampling_visualization/grid_preview_generation.py --input_dir ./videos --output_video ./preview.mp4 --cols 5 --rows 4 --video_length 8
python scripts/curation/tools/sampling_visualization/grid_preview_generation.py --input_dir s3://my-bucket/videos --output_video s3://my-bucket/preview.mp4
```

---

## 2. Video Sampling Visualization Streamlit App (`streamlit_sample_video_app.py`)

This Streamlit app browses `.mp4` files in a local or S3 directory, randomly samples them, and displays them in a paginated 3x4 grid (12 videos per page).

**Usage:**

```bash
streamlit run scripts/curation/tools/sampling_visualization/streamlit_sample_video_app.py -- --input_dir <input_dir_or_s3> [--nsamples 100]
```

**Arguments:**

- `--input_dir`: Input directory or S3 path (required)
- `--nsamples`: Number of videos to sample (default: 100)

**Examples:**

```bash
streamlit run scripts/curation/tools/sampling_visualization/streamlit_sample_video_app.py -- --input_dir ./videos --nsamples 36
streamlit run scripts/curation/tools/sampling_visualization/streamlit_sample_video_app.py -- --input_dir s3://my-bucket/videos
```

### Running Streamlit in the Background (nops/nohup)

To keep the Streamlit app running after closing the terminal, use `nops` or `nohup`:

```bash
nops streamlit run scripts/curation/tools/sampling_visualization/streamlit_sample_video_app.py -- --input_dir <input_dir_or_s3> [--nsamples 100]
```

Or:

```bash
nohup streamlit run scripts/curation/tools/sampling_visualization/streamlit_sample_video_app.py -- --input_dir <input_dir_or_s3> [--nsamples 100] > streamlit.log 2>&1 &
```

This allows the app to keep running even after you disconnect. Access the app in your browser at `http://localhost:8501`.
