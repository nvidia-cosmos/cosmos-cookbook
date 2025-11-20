# Setup and System Requirements

This guide describes how to set up your environment to run the **Cosmos Transfer 2.5 + FiftyOne** workflow to augment the **BioTrove moth dataset** and explore results in the FiftyOne App.

The setup has three main parts:

1. System and software requirements
2. Installing Cosmos Transfer 2.5 and its dependencies
3. Installing and configuring FiftyOne and the dataset

---

## System Requirements

### Minimum Hardware Requirements

- **GPU**:
  - 1 or more NVIDIA GPUs
  - Ampere architecture or newer (e.g., RTX 30 Series, A100, H100 or later recommended)
- **Storage**:
  - Sufficient disk space for:
    - Cosmos Transfer 2.5 repository and model weights
    - FiftyOne datasets and derived videos/images (edge maps, outputs, last frames)

### Supported Platform

- **Operating System**: Linux x86-64
  - Recommended: **Ubuntu ≥ 22.04** (glibc ≥ 2.35)
- **NVIDIA Driver**:
  - **≥ 570.124.06**, compatible with CUDA **12.8.1** (or CUDA 12+)
- **Python**:
  - **Python 3.10** (aligns with Cosmos Transfer 2.5 requirements)

---

## Software Requirements

You will need the following software components:

- **CUDA Toolkit** compatible with your driver (CUDA 12+)
- **PyTorch ≥ 2.5** with CUDA support
- **TorchVision**
- **Git**
- **FFmpeg** (CLI) – for converting images to videos and handling video I/O
- **Python packages**:
  - `fiftyone`
  - `opencv-python` (for Canny edges and video manipulation)
  - Cosmos Transfer 2.5 Python package (installed from the repo)
  - Additional Cosmos dependencies (typically installed via its setup guide), such as:
    - `json5`
    - `gradio` (optional UI)
    - `easyio` (for multi-storage backends)

> For the most accurate list of Cosmos Transfer 2.5 dependencies, always refer to the official **Cosmos Transfer 2.5 Setup Guide**.

---

## Installation

### 1. Create and Activate a Python Environment

You can use either `conda` or `venv`. Example with `conda`:

```bash
conda create -n cosmos-transfer2_5-biotrove python=3.10 -y
conda activate cosmos-transfer2_5-biotrove
```

Or with `venv`:

```bash
python3.10 -m venv cosmos-transfer2_5-biotrove
source cosmos-transfer2_5-biotrove/bin/activate
```

---

### 2. Install FFmpeg

On Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

You should be able to run:

```bash
ffmpeg -version
```

without errors.

---

### 3. Install FiftyOne and Supporting Python Packages

Install FiftyOne and core Python dependencies:

```bash
pip install fiftyone opencv-python
```

Optional, but recommended for working with notebooks and visualization:

```bash
pip install jupyterlab umap-learn
```

---

### 4. Clone and Install Cosmos Transfer 2.5

Clone the Cosmos Transfer 2.5 repository and install it in editable mode.

```bash
git clone https://github.com/nvidia-cosmos/Cosmos Transfer 2.5.git
cd Cosmos Transfer 2.5
pip install -e .
```

Then follow the **Cosmos Transfer 2.5 Setup Guide** for environment configuration and model weight downloads.

---

### 5. Configure Environment Variables and Paths

Set `COSMOS_DIR`:

```bash
export COSMOS_DIR=/path/to/Cosmos Transfer 2.5
```

Optional:

```bash
export LIST_FILE=/path/to/video_list.txt
export MAX_VIDS=100
```

---

## Dataset Setup (BioTrove Moth Dataset)

This recipe uses a subset of the BioTrove dataset from the Hugging Face Hub:

```python
dataset_src = fouh.load_from_hub(
    "pjramg/moth_biotrove",
    persistent=True,
    overwrite=True,
    max_samples=2,
)
```

---

## Verification

### Verify Cosmos Transfer 2.5

```bash
cd $COSMOS_DIR
python examples/inference.py --help
```

### Verify FiftyOne + FFmpeg

```python
dataset_src = fouh.load_from_hub(
    "pjramg/moth_biotrove",
    persistent=True,
    overwrite=False,
    max_samples=2,
)
```

Minimal FFmpeg test:

```bash
ffmpeg -loop 1 -i some_image.jpg -t 1 -c:v libx264 -pix_fmt yuv420p test.mp4
```

---

## Next Steps

You can go to the [inference tutorial](inference.md) to complete the Cosmos Transfer 2.5 + FiftyOne workflow. And visit this [tutorial](https://github.com/voxel51/fiftyone/blob/f5ee552d207c5fff71f12bda9700fa9fe9d57b3c/docs/source/tutorials/cosmos-transfer-integration.ipynb) to run it directly in your environment.
