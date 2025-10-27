# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""
Merge COSMOS and X-Mobility datasets by replacing camera images.

This script processes COSMOS video files and merges them with corresponding X-Mobility
dataset Parquet files by replacing the camera images while preserving segmentation data.
Supports both full dataset merging and segmented video processing modes.
"""

import argparse
import io
import shutil
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

MAX_VIDEO_LENGTH = 121


def bytes_to_image(image_bytes: bytes) -> np.ndarray:
    """
    Convert byte data to a numpy array image.

    Args:
        image_bytes: Image data in bytes format.

    Returns:
        Image as a numpy array.
    """
    image = Image.open(io.BytesIO(image_bytes))
    return np.array(image)


def image_to_bytes(image: np.ndarray) -> bytes:
    """
    Convert numpy array image to bytes using JPEG encoding.

    Args:
        image: Image as a numpy array.

    Returns:
        Encoded image as bytes.

    Raises:
        ValueError: If image encoding fails.
    """
    success, encoded_image = cv2.imencode(".jpg", image)
    if not success:
        raise ValueError("Could not encode image to JPEG format")
    return encoded_image.tobytes()


def parse_segment_info(video_stem: str) -> Tuple[str, int]:
    """
    Parse video filename to extract base filename and segment number.

    Args:
        video_stem: Video filename without extension (stem).

    Returns:
        Tuple of (base_filename, segment_number).

    Examples:
        "goal_0020_0hz_1_1" -> ("goal_0020_0hz", 0)
        "video_2_1_1" -> ("video", 2)
        "video" -> ("video", 0)
    """
    segments = video_stem.split("_")

    # Check if this looks like a segmented video (ends with _X_1_1 or _1_1)
    if len(segments) >= 3 and segments[-1] == "1" and segments[-2] == "1":
        # Try to get segment number from third-to-last position
        try:
            segment = int(segments[-3])
            # If successful, this is format like "video_2_1_1" where 2 is the segment
            base_filename = "_".join(segments[:-3])
        except (ValueError, IndexError):
            # Third-to-last is not a number, so it's format like "goal_0020_0hz_1_1"
            # The _1_1 is just part of naming convention, treat as segment 0
            segment = 0
            # Remove last 2 parts (e.g., "_1_1")
            base_filename = "_".join(segments[:-2])
    else:
        segment = 0
        base_filename = video_stem

    return base_filename, segment


def process_video_simple(
    video_path: Path,
    pqt_path: Path,
    output_pqt_path: Path,
    metadata_path: Optional[Path] = None,
    output_metadata_path: Optional[Path] = None,
) -> int:
    """
    Process a video file and merge with full Parquet data (simple mode).

    Args:
        video_path: Path to the input video file.
        pqt_path: Path to the input Parquet file.
        output_pqt_path: Path to save the output Parquet file.
        metadata_path: Optional path to metadata JSON file.
        output_metadata_path: Optional path to save output metadata.

    Returns:
        Number of frames processed.

    Raises:
        FileNotFoundError: If required input files don't exist.
        ValueError: If video processing fails.
    """
    if not pqt_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {pqt_path}")

    # Read parquet file
    df = pd.read_parquet(pqt_path)

    # Open video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frame_number = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number >= len(df):
                print(f"  Warning: Video has more frames than Parquet data")
                break

            # Resize frame to match semantic image dimensions
            sem_image_shape = df["perspective_semantic_image_shape"].iloc[frame_number]
            frame = cv2.resize(frame, (sem_image_shape[1], sem_image_shape[0]))

            # Convert to bytes and update dataframe
            frame_bytes = image_to_bytes(frame)
            df.loc[frame_number, "camera_image"] = frame_bytes

            frame_number += 1

    finally:
        cap.release()

    if frame_number < len(df):
        print(
            f"  Warning: Parquet has {len(df)} rows but only {frame_number} video frames"
        )

    # Save updated Parquet file
    df.to_parquet(output_pqt_path)

    # Copy metadata if it exists
    if metadata_path and metadata_path.exists() and output_metadata_path:
        shutil.copy2(metadata_path, output_metadata_path)

    return frame_number


def process_video_segmented(
    video_path: Path,
    pqt_path: Path,
    output_pqt_path: Path,
    output_video_path: Path,
    segment: int,
    max_length: int = MAX_VIDEO_LENGTH,
    fps: int = 30,
) -> int:
    """
    Process a segmented video file and merge with corresponding Parquet segment.

    Args:
        video_path: Path to the input video file.
        pqt_path: Path to the input Parquet file (full dataset).
        output_pqt_path: Path to save the output Parquet segment.
        output_video_path: Path to save the output video file.
        segment: Segment number (0-indexed).
        max_length: Maximum frames per segment.
        fps: Frames per second for output video.

    Returns:
        Number of frames processed.

    Raises:
        FileNotFoundError: If required input files don't exist.
        ValueError: If video processing fails.
    """
    if not pqt_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {pqt_path}")

    # Read parquet file
    df = pd.read_parquet(pqt_path)

    # Extract the relevant segment from the dataframe
    start_idx = segment * max_length
    end_idx = (segment + 1) * max_length

    if start_idx >= len(df):
        raise ValueError(
            f"Segment {segment} exceeds available data (only {len(df)} frames)"
        )

    if end_idx > len(df):
        df_segment = df.iloc[start_idx:]
    else:
        df_segment = df.iloc[start_idx:end_idx]

    df_segment = df_segment.reset_index(drop=True)

    # Open input video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get dimensions from first frame's semantic image
    sem_image_shape = df_segment["perspective_semantic_image_shape"].iloc[0]
    video_width, video_height = sem_image_shape[1], sem_image_shape[0]

    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(
        str(output_video_path), fourcc, fps, (video_width, video_height)
    )

    frame_number = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number >= len(df_segment):
                print(f"  Warning: Video has more frames than segment data")
                break

            # Resize frame to match semantic image dimensions
            sem_image_shape = df_segment["perspective_semantic_image_shape"].iloc[
                frame_number
            ]
            frame = cv2.resize(frame, (sem_image_shape[1], sem_image_shape[0]))

            # Convert to bytes and update dataframe
            frame_bytes = image_to_bytes(frame)
            df_segment.loc[frame_number, "camera_image"] = frame_bytes

            # Write to output video
            out_video.write(frame)

            frame_number += 1

    finally:
        cap.release()
        out_video.release()

    if frame_number < len(df_segment):
        print(
            f"  Warning: Segment has {len(df_segment)} rows but only {frame_number} video frames"
        )

    # Save updated Parquet segment
    df_segment.to_parquet(output_pqt_path)

    return frame_number


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Merge COSMOS videos with X-Mobility dataset Parquet files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir_cosmos",
        type=str,
        required=True,
        help="Path to COSMOS dataset directory containing video files",
    )
    parser.add_argument(
        "--input_dir_xmob",
        type=str,
        required=True,
        help="Path to X-Mobility dataset directory containing Parquet files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for merged dataset",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["simple", "segmented"],
        default="simple",
        help='Processing mode: "simple" for full dataset, "segmented" for video segments',
    )
    parser.add_argument(
        "--segment_length",
        type=int,
        default=MAX_VIDEO_LENGTH,
        help="Maximum frames per segment (used in segmented mode)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for output videos (used in segmented mode)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for the dataset merging script."""
    args = parse_arguments()

    # Convert to Path objects and validate
    input_dir_cosmos = Path(args.input_dir_cosmos)
    input_dir_xmob = Path(args.input_dir_xmob)
    output_dir = Path(args.output_dir)

    if not input_dir_cosmos.exists():
        raise FileNotFoundError(f"COSMOS directory not found: {input_dir_cosmos}")

    if not input_dir_xmob.exists():
        raise FileNotFoundError(f"X-Mobility directory not found: {input_dir_xmob}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all video files
    video_files = list(input_dir_cosmos.rglob("*.mp4"))

    if not video_files:
        print(f"No .mp4 files found in {input_dir_cosmos}")
        return

    print(f"Found {len(video_files)} video file(s) to process")
    print(f"Mode: {args.mode}\n")

    processed_count = 0
    error_count = 0

    # Process each video file
    for video_path in tqdm(video_files, desc="Processing videos"):
        try:
            # Get relative path to maintain directory structure
            rel_path = video_path.relative_to(input_dir_cosmos)
            output_subdir = output_dir / rel_path.parent
            output_subdir.mkdir(parents=True, exist_ok=True)

            if args.mode == "simple":
                # Simple mode: process full dataset
                filename = video_path.stem

                pqt_path = input_dir_xmob / rel_path.parent / f"{filename}.pqt"
                metadata_path = (
                    input_dir_xmob / rel_path.parent / f"{filename}_metadata.json"
                )

                new_pqt_path = output_subdir / f"{filename}.pqt"
                new_metadata_path = output_subdir / f"{filename}_metadata.json"

                frame_count = process_video_simple(
                    video_path, pqt_path, new_pqt_path, metadata_path, new_metadata_path
                )

            else:  # segmented mode
                # Parse segment information from filename
                base_filename, segment = parse_segment_info(video_path.stem)

                pqt_path = input_dir_xmob / rel_path.parent / f"{base_filename}.pqt"

                new_pqt_path = output_subdir / f"{video_path.stem}.pqt"
                output_video_path = output_subdir / f"{video_path.stem}.mp4"

                frame_count = process_video_segmented(
                    video_path,
                    pqt_path,
                    new_pqt_path,
                    output_video_path,
                    segment,
                    args.segment_length,
                    args.fps,
                )

            processed_count += 1

        except Exception as e:
            print(f"\nError processing {video_path.name}: {e}")
            error_count += 1
            continue

    # Print summary
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Successfully processed: {processed_count}")
    print(f"  Errors: {error_count}")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
