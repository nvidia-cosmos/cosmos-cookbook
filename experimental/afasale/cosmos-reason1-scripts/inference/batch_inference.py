#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "accelerate",
#   "boto3",
#   "qwen-vl-utils",
#   "rich",
#   "torch",
#   "torchcodec",
#   "torchvision",
#   "transformers>=4.51.3",
# ]
# [tool.uv]
# exclude-newer = "2025-07-31T00:00:00Z"
# ///

"""Run inference on a model with a given prompt.

Example:

```shell
./inference.py --prompt 'Please describe the video.' --videos video.mp4
```

Or process a directory of videos:

```shell
./scripts/batch_inference.py --prompt "Did the ego vehicle change lanes in the video? (A) Yes (B) No (C) Not sure (D) Not applicable" --video-dir /mnt/pvc/workspace/cosmos-reason1/datasets/nuscenes/cam_front/test/ --output-dir ./outputs/
```

Or process videos but only save analysis results (skip copying videos):

```shell
./scripts/batch_inference.py --prompt "Describe the driving scenario" --video-dir /path/to/videos/ --output-dir ./results/ --no-copy-videos
```

Or process videos from S3 and save results to S3:

```shell
./scripts/batch_inference.py --prompt "Did the ego vehicle change lanes?" --video-dir s3://my-bucket/videos/ --output-dir s3://my-bucket/results/
```

Or process videos from S3 and save results locally:

```shell
./scripts/batch_inference.py --prompt "Describe the scene" --video-dir s3://my-bucket/videos/ --output-dir ./local-results/
```
"""

import argparse
import shutil
import tempfile
import os
from pathlib import Path
from urllib.parse import urlparse

import boto3
import qwen_vl_utils
import transformers
from rich import print


def is_s3_path(path):
    """Check if a path is an S3 URL."""
    return str(path).startswith('s3://')


def parse_s3_path(s3_path):
    """Parse S3 path and return bucket and key."""
    parsed = urlparse(s3_path)
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')
    return bucket, key


def download_s3_file(s3_client, s3_path, local_path):
    """Download a file from S3 to local path."""
    bucket, key = parse_s3_path(s3_path)
    s3_client.download_file(bucket, key, local_path)


def upload_s3_file(s3_client, local_path, s3_path):
    """Upload a local file to S3."""
    bucket, key = parse_s3_path(s3_path)
    s3_client.upload_file(local_path, bucket, key)


def list_s3_videos(s3_client, s3_dir):
    """List all video files in an S3 directory."""
    bucket, prefix = parse_s3_path(s3_dir)
    video_extensions = {'.mp4'}
    video_files = []
    
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                if any(key.lower().endswith(ext) for ext in video_extensions):
                    video_files.append(f's3://{bucket}/{key}')
    
    return sorted(video_files)


def process_single_video(video_path, prompt, system_prompt, images, model, processor, generation_config, fps, max_pixels):
    """Process a single video and return the response."""
    user_content = []
    
    # Add images if provided
    for image in images or []:
        user_content.append(
            {"type": "image", "image": image, "max_pixels": max_pixels}
        )
    
    # Add video
    user_content.append(
        {
            "type": "video",
            "video": video_path,
            "fps": fps,
            "max_pixels": max_pixels,
        }
    )
    user_content.append({"type": "text", "text": prompt})
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    # Process the messages
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = qwen_vl_utils.process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Run inference
    generated_ids = model.generate(**inputs, generation_config=generation_config)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0]


def get_video_files(directory, s3_client=None):
    """Get all video files from a directory and its subdirectories recursively."""
    if is_s3_path(directory):
        if s3_client is None:
            raise ValueError("S3 client required for S3 paths")
        return list_s3_videos(s3_client, directory)
    else:
        video_extensions = {'.mp4'}
        video_files = []
        
        for file_path in Path(directory).rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in video_extensions:
                video_files.append(str(file_path))
        
        return sorted(video_files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="User prompt message")
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are a helpful video analyzer. Answer the question with provided options in the following format: <think>\nyour reasoning\n</think>\n\n<answer>\nyour answer\n</answer>.",
        help="System prompt message",
    )
    parser.add_argument("--images", type=str, nargs="*", help="Image paths to include with each video")
    parser.add_argument("--videos", type=str, nargs="*", help="Video paths (for single video processing)")
    parser.add_argument("--video-dir", type=str, help="Directory containing videos to process")
    parser.add_argument("--output-dir", type=str, help="Directory to save output .txt files (defaults to same directory as input videos)")
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/Cosmos-Reason1-7B",
        help="Model name (https://huggingface.co/collections/nvidia/cosmos-reason1-67c9e926206426008f1da1b7)",
    )
    parser.add_argument(
        "--fps", type=int, default=8, help="Downsample video frame rate"
    )
    parser.add_argument(
        "--max-pixels", type=int, default=81920, help="Downsample media max pixels"
    )
    parser.add_argument("--copy-videos", action="store_true", help="Copy video files to output directory (only save text results)")
    args = parser.parse_args()

    # Initialize S3 client if needed
    s3_client = None
    use_s3 = (args.video_dir and is_s3_path(args.video_dir)) or (args.output_dir and is_s3_path(args.output_dir))
    if use_s3:
        s3_client = boto3.client('s3')
        print("Initialized S3 client")

    # Create output directory if specified and not S3
    if args.output_dir and not is_s3_path(args.output_dir):
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
    elif args.output_dir and is_s3_path(args.output_dir):
        print(f"S3 output directory: {args.output_dir}")

    # Create temporary directory for S3 downloads
    temp_dir = None
    if args.video_dir and is_s3_path(args.video_dir):
        temp_dir = tempfile.mkdtemp()
        print(f"Created temporary directory for S3 downloads: {temp_dir}")

    # Determine video files to process
    if args.video_dir:
        video_files = get_video_files(args.video_dir, s3_client)
        if not video_files:
            print(f"[red]Error: No video files found in {args.video_dir}[/red]")
            return
        print(f"Found {len(video_files)} video files in {args.video_dir}")
    elif args.videos:
        video_files = args.videos
    else:
        print("[red]Error: Must specify either --videos or --video-dir[/red]")
        return

    print(f"Loading model: {args.model}")
    
    # Load the model once
    model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto"
    )
    processor: transformers.Qwen2_5_VLProcessor = (
        transformers.AutoProcessor.from_pretrained(args.model, use_fast=True)
    )

    generation_config = transformers.GenerationConfig(
        do_sample=True,
        max_new_tokens=4096,
        repetition_penalty=1.05,
        temperature=0.6,
        top_p=0.95,
    )

    print("Model loaded successfully!")
    print(f"Processing {len(video_files)} videos...")

    # Process each video
    try:
        for i, video_path in enumerate(video_files, 1):
            print(f"\n[{i}/{len(video_files)}] Processing: {video_path}")
            
            try:
                # Handle S3 video download if needed
                local_video_path = video_path
                if is_s3_path(video_path):
                    video_filename = Path(video_path).name
                    local_video_path = os.path.join(temp_dir, video_filename)
                    print(f"Downloading from S3: {video_path}")
                    download_s3_file(s3_client, video_path, local_video_path)
                    print(f"Downloaded to: {local_video_path}")
                
                # Process the video
                response = process_single_video(
                    local_video_path, args.prompt, args.system_prompt, args.images,
                    model, processor, generation_config, args.fps, args.max_pixels
                )
                
                # Create output text file path
                video_path_obj = Path(video_path)
                if args.output_dir:
                    if is_s3_path(args.output_dir):
                        # S3 output path
                        output_s3_path = f"{args.output_dir.rstrip('/')}/{video_path_obj.stem}.txt"
                        local_output_path = os.path.join(temp_dir or ".", f"{video_path_obj.stem}.txt")
                    else:
                        # Local output path
                        output_path = Path(args.output_dir) / f"{video_path_obj.stem}.txt"
                        local_output_path = str(output_path)
                else:
                    # Save in same directory as video with .txt extension
                    if is_s3_path(video_path):
                        # For S3 videos without output-dir, save locally
                        local_output_path = f"{video_path_obj.stem}.txt"
                    else:
                        output_path = video_path_obj.with_suffix('.txt')
                        local_output_path = str(output_path)
                
                # Save response to text file
                with open(local_output_path, 'w', encoding='utf-8') as f:
                    f.write(f"Video: {video_path}\n")
                    f.write(f"Prompt: {args.prompt}\n")
                    f.write(f"Response:\n{response}\n")
                
                # Upload to S3 if output directory is S3
                if args.output_dir and is_s3_path(args.output_dir):
                    print(f"Uploading result to S3: {output_s3_path}")
                    upload_s3_file(s3_client, local_output_path, output_s3_path)
                    print(f"[green]✓ Uploaded response to: {output_s3_path}[/green]")
                    # Clean up local temp file
                    os.remove(local_output_path)
                else:
                    print(f"[green]✓ Saved response to: {local_output_path}[/green]")
                
                # Copy video file to output directory if specified and enabled
                if args.output_dir and args.copy_videos:
                    if is_s3_path(args.output_dir):
                        # Upload video to S3
                        video_s3_path = f"{args.output_dir.rstrip('/')}/{video_path_obj.name}"
                        if is_s3_path(video_path):
                            # S3 to S3 copy - use local downloaded file
                            upload_s3_file(s3_client, local_video_path, video_s3_path)
                        else:
                            # Local to S3 upload
                            upload_s3_file(s3_client, video_path, video_s3_path)
                        print(f"[green]✓ Uploaded video to: {video_s3_path}[/green]")
                    else:
                        # Local copy
                        video_output_path = Path(args.output_dir) / video_path_obj.name
                        if is_s3_path(video_path):
                            # S3 to local copy - use downloaded file
                            shutil.copy2(local_video_path, video_output_path)
                        else:
                            shutil.copy2(video_path, video_output_path)
                        print(f"[green]✓ Copied video to: {video_output_path}[/green]")
                
                # Clean up downloaded S3 video file
                if is_s3_path(video_path) and os.path.exists(local_video_path):
                    os.remove(local_video_path)
                
                print(f"Response preview: {response[:200]}...")
                
            except Exception as e:
                print(f"[red]✗ Error processing {video_path}: {str(e)}[/red]")
                continue

        print(f"\n[green]Completed processing {len(video_files)} videos![/green]")
        
    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    main()
