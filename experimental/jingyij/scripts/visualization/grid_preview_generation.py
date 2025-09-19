import argparse
import os
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scripts.curation.tools.common.s3_utils import (
    is_s3_path,
    sync_local_to_s3,
    sync_s3_to_local,
)


def find_mp4_files(directory: str) -> List[Path]:
    """Find all .mp4 files in a local directory."""
    dir_path = Path(directory)
    return list(dir_path.rglob("*.mp4"))


def find_s3_mp4_files(s3_path: str) -> List[str]:
    """
    Find all .mp4 files in an S3 directory using s5cmd.
    
    Args:
        s3_path: S3 directory path
        
    Returns:
        List of S3 paths to .mp4 files
    """
    # Ensure AWS profile is set
    if "AWS_PROFILE" not in os.environ:
        os.environ["AWS_PROFILE"] = "lha-share"
    
    # Use s5cmd to list files recursively
    s3_prefix = s3_path.rstrip('/') + '/'
    cmd = ["s5cmd", "ls", f"{s3_prefix}**/*.mp4"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        video_files = []
        
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                # s5cmd ls output format: "2023/01/01 00:00:00    1234 s3://bucket/path/file.mp4"
                # Sometimes the format might be different, so let's be more robust
                parts = line.strip().split()
                if len(parts) >= 1:
                    # Look for the part that contains .mp4 and starts with s3://
                    for part in parts:
                        if part.endswith('.mp4') and part.startswith('s3://'):
                            video_files.append(part)
                            break
                    else:
                        # If no s3:// found, the last part might be the relative path
                        # We need to construct the full S3 path
                        if parts[-1].endswith('.mp4'):
                            if not parts[-1].startswith('s3://'):
                                # Construct full S3 path
                                full_s3_path = f"{s3_prefix}{parts[-1]}"
                                video_files.append(full_s3_path)
                            else:
                                video_files.append(parts[-1])
        
        print(f"Found {len(video_files)} .mp4 files in S3")
        return video_files
        
    except subprocess.CalledProcessError as e:
        print(f"Failed to list S3 files: {e.stderr}")
        # Fallback: try non-recursive listing
        cmd_simple = ["s5cmd", "ls", f"{s3_prefix}*.mp4"]
        try:
            result = subprocess.run(cmd_simple, capture_output=True, text=True, check=True)
            video_files = []
            
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        for part in parts:
                            if part.endswith('.mp4') and part.startswith('s3://'):
                                video_files.append(part)
                                break
                        else:
                            if parts[-1].endswith('.mp4'):
                                if not parts[-1].startswith('s3://'):
                                    full_s3_path = f"{s3_prefix}{parts[-1]}"
                                    video_files.append(full_s3_path)
                                else:
                                    video_files.append(parts[-1])
            
            print(f"Found {len(video_files)} .mp4 files in S3 (non-recursive)")
            return video_files
            
        except subprocess.CalledProcessError as e2:
            print(f"Failed to list S3 files: {e2.stderr}")
            raise RuntimeError(f"Could not list files in S3 path: {s3_path}")


def create_grid_video(
    video_paths: List[str], output_path: str, duration: int = 5, fps: int = 30, grid_size: Tuple[int, int] = (10, 10)
) -> str:
    caps = []
    temp_files = []  # Keep track of temporary files for S3 videos
    
    for video_path in video_paths:
        actual_path = video_path
        
        # If it's an S3 path, download to temporary file
        if is_s3_path(video_path):
            temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            temp_path = temp_file.name
            temp_file.close()
            temp_files.append(temp_path)
            
            try:
                print(f"Downloading {video_path} for processing...")
                sync_s3_to_local(video_path, temp_path)
                actual_path = temp_path
            except Exception as e:
                print(f"Warning: Could not download {video_path}: {e}")
                continue
        
        cap = cv2.VideoCapture(str(actual_path))
        if not cap.isOpened():
            print(f"Warning: Could not open {actual_path}")
            continue
        caps.append(cap)
        
    if not caps:
        raise ValueError("No valid videos found")

    target_width = 192
    target_height = 108
    grid_w = grid_size[0] * target_width
    grid_h = grid_size[1] * target_height
    total_frames = duration * fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    out = cv2.VideoWriter(output_path, fourcc, fps, (grid_w, grid_h))

    for _ in range(total_frames):  # frame_idx is not used
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        for i, cap in enumerate(caps[: grid_size[0] * grid_size[1]]):
            ret, frame = cap.read()
            if not ret:
                # If video ends, pad with black frame
                frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            else:
                frame = cv2.resize(frame, (target_width, target_height))
            row = i // grid_size[0]
            col = i % grid_size[0]
            y = row * target_height
            x = col * target_width
            grid[y: y + target_height, x: x + target_width] = frame
        out.write(grid)
    
    # Clean up
    for cap in caps:
        cap.release()
    out.release()
    
    # Remove temporary files
    for temp_file in temp_files:
        try:
            os.unlink(temp_file)
        except Exception as e:
            print(f"Warning: Could not remove temporary file {temp_file}: {e}")
    
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a grid preview video from multiple videos")
    parser.add_argument("--input_dir", required=True, help="Input directory or S3 bucket containing videos")
    parser.add_argument("--output_video", required=True, help="Output video file path (local or S3)")
    parser.add_argument("--cols", type=int, default=10, help="Number of columns in the grid (default: 10)")
    parser.add_argument("--rows", type=int, default=10, help="Number of rows in the grid (default: 10)")
    parser.add_argument("--video_length", type=int, default=5, help="Length of the output video in seconds (default: 5)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")

    args = parser.parse_args()

    temp_output_dir: Optional[str] = None
    output_local_path = args.output_video

    try:
        # Find videos
        if is_s3_path(args.input_dir):
            print(f"Searching for .mp4 files in S3: {args.input_dir}")
            all_videos_str = find_s3_mp4_files(args.input_dir)
        else:
            print(f"Searching for .mp4 files in local directory: {args.input_dir}")
            all_videos_paths = find_mp4_files(args.input_dir)
            all_videos_str = [str(p) for p in all_videos_paths]

        if not all_videos_str:
            print("No .mp4 files found in input directory")
            return

        # Handle S3 output
        if is_s3_path(args.output_video):
            temp_output_dir = tempfile.mkdtemp()
            output_local_path = os.path.join(temp_output_dir, "grid_preview.mp4")

        # Sample videos
        num_samples = args.cols * args.rows
        if len(all_videos_str) < num_samples:
            print(f"Warning: Only {len(all_videos_str)} videos found, using all available videos")
            sampled_videos = all_videos_str
        else:
            sampled_videos = random.sample(all_videos_str, num_samples)

        print(f"Selected {len(sampled_videos)} videos for grid preview")
        
        # Create grid video
        print("Creating grid video...")
        create_grid_video(
            sampled_videos,
            output_local_path,
            duration=args.video_length,
            fps=args.fps,
            grid_size=(args.cols, args.rows),
        )

        # Upload to S3 if needed
        if is_s3_path(args.output_video) and temp_output_dir is not None:
            print(f"Uploading grid video to S3: {args.output_video}")
            sync_local_to_s3(output_local_path, args.output_video)

        print(f"\nGrid video created successfully at: {args.output_video}")

    finally:
        if temp_output_dir:
            shutil.rmtree(temp_output_dir)


if __name__ == "__main__":
    main()
