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
Convert X-Mobility dataset from Parquet format to video files.

This script processes Parquet files containing camera images and semantic segmentation
data, converting them into MP4 video files with optional segmentation overlays.
"""

import argparse
import io
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def create_fixed_colormap() -> np.ndarray:
    """
    Create a fixed colormap of 16 distinct pastel colors for segmentation visualization.

    Returns:
        Array of RGB color values (16 x 3) with dtype uint8.
    """
    colors = [
        [255, 182, 193],  # Light pink
        [176, 224, 230],  # Powder blue
        [255, 218, 185],  # Peach
        [221, 160, 221],  # Plum
        [176, 196, 222],  # Light steel blue
        [152, 251, 152],  # Pale green
        [255, 255, 224],  # Light yellow
        [230, 230, 250],  # Lavender
        [255, 228, 225],  # Misty rose
        [240, 255, 240],  # Honeydew
        [255, 240, 245],  # Lavender blush
        [224, 255, 255],  # Light cyan
        [250, 235, 215],  # Antique white
        [245, 255, 250],  # Mint cream
        [255, 228, 196],  # Bisque
        [240, 248, 255],  # Alice blue
    ]
    return np.array(colors, dtype=np.uint8)


def create_video_writer(
    output_path: str, fps: int, width: int, height: int, segment: int = 0
) -> Tuple[cv2.VideoWriter, str]:
    """
    Create a video writer with automatic segment numbering.

    Args:
        output_path: Base output path for the video file.
        fps: Frames per second for the output video.
        width: Width of the video frames.
        height: Height of the video frames.
        segment: Segment number (0 for single video, >0 for multi-part videos).

    Returns:
        Tuple of (VideoWriter object, actual output path).
    """
    # Create segmented output path if segment > 0
    if segment > 0:
        path = Path(output_path)
        if "_segmentation" not in path.stem:
            output_path = str(path.parent / f"{path.stem}_{segment}{path.suffix}")
        else:
            # Handle segmentation videos separately
            base_stem = path.stem.replace("_segmentation", "")
            output_path = str(
                path.parent / f"{base_stem}_{segment}_segmentation{path.suffix}"
            )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height)), output_path


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


def process_semantic_image(
    semantic_data: np.ndarray, shape: Tuple[int, int], colormap: np.ndarray
) -> np.ndarray:
    """
    Process semantic segmentation data into a BGR image.

    Args:
        semantic_data: Raw semantic segmentation data.
        shape: Target shape (height, width) for the segmentation image.
        colormap: Color mapping for segmentation classes.

    Returns:
        Processed semantic image in BGR format.
    """
    semantic_image = np.array(semantic_data).reshape(shape[:2])
    semantic_image = colormap[semantic_image % len(colormap)]

    # Ensure BGR format
    if semantic_image.ndim == 2 or semantic_image.shape[-1] == 1:
        semantic_image = cv2.cvtColor(semantic_image, cv2.COLOR_GRAY2BGR)
    elif semantic_image.shape[-1] == 4:
        semantic_image = cv2.cvtColor(semantic_image, cv2.COLOR_RGBA2BGR)

    return semantic_image


def create_video_writers(
    output_dir: Path,
    file_stem: str,
    fps: int,
    camera_dims: Tuple[int, int],
    semantic_dims: Tuple[int, int],
    segment: int = 0,
) -> Tuple:
    """
    Create video writers for camera and semantic segmentation videos.

    Args:
        output_dir: Directory to save videos.
        file_stem: Base name for output files.
        fps: Frames per second for output videos.
        camera_dims: (width, height) for camera video.
        semantic_dims: (width, height) for semantic video.
        segment: Segment number for multi-part videos.

    Returns:
        Tuple of (camera_writer, camera_output, semantic_writer, semantic_output).
    """
    camera_writer, camera_output = create_video_writer(
        str(output_dir / f"{file_stem}.mp4"),
        fps,
        camera_dims[0],
        camera_dims[1],
        segment,
    )

    semantic_writer, semantic_output = create_video_writer(
        str(output_dir / f"{file_stem}_segmentation.mp4"),
        fps,
        semantic_dims[0],
        semantic_dims[1],
        segment,
    )

    return camera_writer, camera_output, semantic_writer, semantic_output


def process_parquet_file(
    file_path: Path,
    output_dir: Path,
    fps: int,
    segment_length: int,
    colormap: np.ndarray,
    input_dir: Path,
) -> None:
    """
    Process a single Parquet file and generate video outputs.

    Args:
        file_path: Path to the input Parquet file.
        output_dir: Base output directory.
        fps: Frames per second for output videos.
        segment_length: Number of frames per video segment.
        colormap: Color mapping for segmentation classes.
        input_dir: Base input directory for computing relative paths.
    """
    rel_path = file_path.relative_to(input_dir)
    output_subdir = output_dir / rel_path.parent
    output_subdir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {rel_path}")

    # Load data and determine dimensions
    df = pd.read_parquet(file_path)
    first_camera_image = bytes_to_image(df["camera_image"].iloc[0])

    camera_height, camera_width = first_camera_image.shape[:2]
    semantic_height, semantic_width = df["perspective_semantic_image_shape"].iloc[0][:2]

    # Initialize tracking variables
    current_segment = 0
    frame_count = 0

    (
        camera_writer,
        camera_output,
        semantic_writer,
        semantic_output,
    ) = create_video_writers(
        output_subdir,
        file_path.stem,
        fps,
        (camera_width, camera_height),
        (semantic_width, semantic_height),
    )

    try:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{rel_path.name}"):
            # Start new segment if needed
            if frame_count >= segment_length:
                camera_writer.release()
                semantic_writer.release()
                print(f"  Segment {current_segment} saved")

                current_segment += 1
                frame_count = 0

                (
                    camera_writer,
                    camera_output,
                    semantic_writer,
                    semantic_output,
                ) = create_video_writers(
                    output_subdir,
                    file_path.stem,
                    fps,
                    (camera_width, camera_height),
                    (semantic_width, semantic_height),
                    current_segment,
                )

            # Process and write camera frame
            camera_image = bytes_to_image(row["camera_image"])
            camera_image = cv2.cvtColor(camera_image, cv2.COLOR_RGB2BGR)
            camera_writer.write(camera_image)

            # Process and write semantic frame
            semantic_image = process_semantic_image(
                row["perspective_semantic_image"],
                row["perspective_semantic_image_shape"],
                colormap,
            )
            semantic_writer.write(semantic_image)

            frame_count += 1

    except Exception as e:
        print(f"Error processing {rel_path}: {e}")
        raise

    finally:
        camera_writer.release()
        semantic_writer.release()
        print(f"  Completed: {current_segment + 1} segment(s) saved\n")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Convert X-Mobility Parquet dataset to video files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to dataset directory containing .pqt files",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory for generated videos",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for output videos",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=121,
        help="Maximum number of frames per video segment",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for the video conversion script."""
    args = parse_arguments()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare colormap
    colormap = create_fixed_colormap()

    # Process all Parquet files
    parquet_files = list(input_dir.rglob("*.pqt"))

    if not parquet_files:
        print(f"No .pqt files found in {input_dir}")
        return

    print(f"Found {len(parquet_files)} Parquet file(s) to process\n")

    for file_path in parquet_files:
        process_parquet_file(
            file_path, output_dir, args.fps, args.length, colormap, input_dir
        )

    print(f"All files processed successfully! Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
