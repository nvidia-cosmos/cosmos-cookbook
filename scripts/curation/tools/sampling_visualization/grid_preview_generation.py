import argparse
import os
import random
import shutil
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
    dir_path = Path(directory)
    return list(dir_path.rglob("*.mp4"))


def create_grid_video(
    video_paths: List[Path],
    output_path: str,
    duration: int = 5,
    fps: int = 30,
    grid_size: Tuple[int, int] = (10, 10),
) -> str:
    caps = []
    for video_path in video_paths:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Warning: Could not open {video_path}")
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
            grid[y : y + target_height, x : x + target_width] = frame
        out.write(grid)
    for cap in caps:
        cap.release()
    out.release()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a grid preview video from multiple videos"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Input directory or S3 bucket containing videos",
    )
    parser.add_argument(
        "--output_video", required=True, help="Output video file path (local or S3)"
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=10,
        help="Number of columns in the grid (default: 10)",
    )
    parser.add_argument(
        "--rows", type=int, default=10, help="Number of rows in the grid (default: 10)"
    )
    parser.add_argument(
        "--video_length",
        type=int,
        default=5,
        help="Length of the output video in seconds (default: 5)",
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Frames per second (default: 30)"
    )

    args = parser.parse_args()

    temp_input_dir: Optional[str] = None
    temp_output_dir: Optional[str] = None
    output_local_path = args.output_video

    try:
        # Handle S3 input
        if is_s3_path(args.input_dir):
            temp_input_dir = tempfile.mkdtemp()
            sync_s3_to_local(args.input_dir, temp_input_dir)
            input_dir = temp_input_dir
        else:
            input_dir = args.input_dir

        # Handle S3 output
        if is_s3_path(args.output_video):
            temp_output_dir = tempfile.mkdtemp()
            output_local_path = os.path.join(temp_output_dir, "grid_preview.mp4")

        # Find and sample videos
        all_videos = find_mp4_files(input_dir)
        num_samples = args.cols * args.rows
        if len(all_videos) < num_samples:
            print(
                f"Warning: Only {len(all_videos)} videos found, using all available videos"
            )
            sampled_videos = all_videos
        else:
            sampled_videos = random.sample(all_videos, num_samples)

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
            sync_local_to_s3(temp_output_dir, args.output_video)

        print(f"\nGrid video created successfully at: {args.output_video}")

    finally:
        if temp_input_dir:
            shutil.rmtree(temp_input_dir)
        if temp_output_dir:
            shutil.rmtree(temp_output_dir)


if __name__ == "__main__":
    main()
