# Data Standardization Tools

This directory contains tools for standardizing video datasets, including batch video format conversion and filename normalization. Both tools support local directories and S3 buckets.

## Installation

1. **Create a new conda environment**
   Create a new conda environment with Python 3.12, or re-use an environment of your choice.

   ```bash
   conda create -n data_curation python=3.12
   conda activate data_curation
   ```

1. **Install ffmpeg and s5cmd**
   Make sure `ffmpeg` is installed and available in your system path.

   ```bash
   conda install -c conda-forge ffmpeg
   pip install s5cmd
   ```

1. **(Optional) Set up AWS credentials**
   If you are working with S3 buckets, ensure your AWS credentials are configured in `~/.aws/credentials`.

---

## Usage

### 1. Video Format Conversion (`mp4_conversion.py`)

Convert all supported video files in a directory or S3 bucket to `.mp4` format, with options for resolution, framerate, and codec.

**Command:**

```bash
python mp4_conversion.py <input_dir_or_s3> <output_dir_or_s3> [--resolution 1920:1080] [--framerate 30] [--codec libx265]
```

**Arguments:**

- `input_dir`: Input directory or S3 bucket containing video files (required)
- `output_dir`: Output directory or S3 bucket for converted .mp4 files (required)
- `--resolution`: Output video resolution, e.g., 1920:1080 (default: 1920:1080)
- `--framerate`: Output video framerate (default: 30)
- `--codec`: Video codec to use (default: libx265 for H.265)

**Examples:**

- Local to local:

  ```bash
  python mp4_conversion.py ./input_videos ./output_videos
  ```

- S3 to S3:

  ```bash
  python mp4_conversion.py s3://my-bucket/input s3://my-bucket/output
  ```

---

### 2. Filename Normalization (`file_renaming.py`)

Recursively rename files in a directory or S3 bucket to be processing-friendly (replace whitespaces with underscores, and remove special characters).

**Command:**

```bash
python file_renaming.py <input_dir_or_s3>
```

**Arguments:**

- `input_dir`: Input directory or S3 bucket to process (required)

**Examples:**

- Local directory:

  ```bash
  python file_renaming.py ./my_dataset
  ```

- S3 bucket:

  ```bash
  python file_renaming.py s3://my-bucket/my_dataset
  ```

---

## Notes

- For S3 operations, `s5cmd` must be installed and in your `PATH`.
- For S3 buckets, the scripts will download data to a temporary local directory, process it, and sync results back to S3.
- Make sure you have sufficient permissions for the S3 buckets you are working with.

---

If you have any questions or encounter issues, please contact the maintainer or open an issue in this repository.
