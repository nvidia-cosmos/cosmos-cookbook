# Video Processing Tools

This directory contains a collection of tools for video processing and classification. These tools help with camera type classification, black bar removal, and content quality assessment.

## Deploying a VLM

Before using the classification tools, you must deploy a Vision-Language Model (VLM) service. You can choose to deploy either an off-the-shelf QWen model or a Cosmos Reason model.

### Deploying QWen (Off-the-Shelf)

To launch a QWen model instance, run the following command (replace the placeholders as needed):

```bash
docker run --runtime nvidia --gpus '"device=<gpu_device_index>"' \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HUGGING_FACE_HUB_TOKEN=<hugging_face_api_key>" \
  -p <port_number>:8000 --ipc=host \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --limit-mm-per-prompt "image=10,video=10"
```

Replace `<gpu_device_index>`, `<hugging_face_api_key>`, and `<port_number>` with your actual GPU index, Hugging Face API key, and desired port number.

### Deploying Cosmos Reason as VLM

To deploy the Cosmos Reason model (checkpoints are available on our DGX dev system), use the following steps:

1. Start a Docker container (replace `<gpu_device_index>` and `<docker_image>` as appropriate):

    ```bash
    docker run --gpus device="<gpu_device_index>" --rm -it --network host --ipc host \
      --entrypoint /bin/bash \
      -v /var/inf/response:/var/inf/response \
      -v ./:./imaginaire4 \
      -v /home/checkpoints:/config/models \
      <docker_image>
    ```

    `<docker_image>` should be a recent imaginaire4 Docker image.

2. Once inside the container, start the VLM server:

    ```bash
    python3 -m vllm.entrypoints.openai.api_server --model /config/models/qwen2p5_vl_7B_instruct --limit_mm_per_prompt image=10,video=10
    ```

**Note:** Adjust the mount paths, environment variables, and parameters as needed for your specific environment.

## Prerequisites

- FFmpeg (for pillarbox/letterbox removal)
- NVCF API key (for camera type and content classification)
- Required Python packages:

  ```bash
  pip install openai opencv-python
  ```

- Locally deployed VLM, accessble through an IP

## Tools Overview

### 1. Camera Type Classifier (`camera_type_classifier.py`)

Classifies videos into three categories:

- Static CCTV-style camera
- Static dash-camera (parked)
- Moving dash-camera (in-motion)

#### Usage

```bash
python camera_type_classifier.py --input_dir /path/to/videos --output_file results.json --model_name [reason|qwen7b]
```

#### Parameters

- `--input_dir`: Directory containing videos to process
- `--output_file`: Output JSON file name (default: camera_classification_results.json)
- `--model_name`: Model to use for inference (default: reason)

### 2. Pillarbox/Letterbox Removal (`pillarbox_letterbox_removal.py`)

Removes black bars (pillarbox/letterbox) from videos while preserving aspect ratio.

#### Usage

```bash
python pillarbox_letterbox_removal.py --input_dir /path/to/videos --output_dir /path/to/output
```

#### Parameters

- `--input_dir`: Input directory containing videos (default: current directory)
- `--output_dir`: Directory to save cropped videos (required)

### 3. Poor Content Classifier (`poor_content_classifier.py`)

Classifies videos as either GOOD or BAD content based on quality criteria.

#### Usage

```bash
python poor_content_classifier.py --input_dir /path/to/videos --model_name [reason|qwen7b] --inference_endpoint [optional_custom_endpoint]
```

#### Parameters

- `--input_dir`: Directory containing videos to process
- `--model_name`: Model to use for inference (default: reason)
- `--inference_endpoint`: Optional custom inference endpoint URL

## Environment Setup

1. Set up API keys:

   ```bash
   export NVCF_API_KEY="your_api_key"  # For camera type classifier
   export NGC_API_KEY="your_api_key"   # For poor content classifier
   ```

2. Ensure FFmpeg is installed:

   ```bash
   # Ubuntu/Debian
   sudo apt-get install ffmpeg

   # macOS
   brew install ffmpeg
   ```

## Output Format

### Camera Type Classifier

Outputs a JSON file with the following structure:

```json
{
    "video_path": {
        "category": "category_number",
        "full_response": "model_response"
    }
}
```

### Poor Content Classifier

Outputs a JSON file named `bad_content_classification_results.json` with:

```json
{
    "video_path": {
        "category": "good|bad",
        "full_response": "model_response"
    }
}
```

## Notes

- All tools process videos recursively in the input directory
- The classification tools use AI models that require API access
- The pillarbox/letterbox removal tool uses FFmpeg for video processing
- Output files are saved in the specified output directory or input directory
