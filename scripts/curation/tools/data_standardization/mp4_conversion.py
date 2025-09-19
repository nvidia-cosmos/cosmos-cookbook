import argparse
import os
import shutil
import subprocess
import tempfile
from typing import List

from scripts.curation.tools.common.s3_utils import (
    is_s3_path,
    sync_local_to_s3,
    sync_s3_to_local,
)

# List of common video file extensions to be converted to .mp4
VIDEO_EXTENSIONS: List[str] = [".mkv", ".avi", ".mov", ".flv", ".wmv", ".webm", ".mpg", ".mpeg", ".3gp", ".m4v"]


def convert_to_mp4(input_dir: str, output_dir: str, resolution: str, framerate: int, codec: str) -> None:
    """
    Recursively search for all supported video files in input_dir and convert them to .mp4 format
    with the specified resolution, framerate, and codec. The converted files are saved in output_dir
    with the same relative path structure as input_dir.
    """
    for root, _, files in os.walk(input_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            # Convert only supported video formats that are not already .mp4
            if ext in VIDEO_EXTENSIONS and ext != ".mp4":
                input_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, rel_path)
                os.makedirs(output_subdir, exist_ok=True)
                output_file = os.path.splitext(file)[0] + ".mp4"
                output_path = os.path.join(output_subdir, output_file)

                cmd = [
                    "ffmpeg",
                    "-i",
                    input_path,
                    "-vf",
                    f"scale={resolution}",
                    "-r",
                    str(framerate),
                    "-c:v",
                    codec,
                    "-c:a",
                    "aac",
                    "-y",  # Overwrite output files without asking
                    output_path,
                ]
                print(f"Converting: {input_path} -> {output_path}")
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Recursively convert all common video files (e.g., .mkv, .avi, .mov, etc.) in a directory or S3 bucket "
            "to .mp4 format with specified settings. Maintains the directory structure in the output location."
        )
    )
    parser.add_argument("input_dir", help="Input directory or S3 bucket containing video files of various formats")
    parser.add_argument("output_dir", help="Output directory or S3 bucket for converted .mp4 files")
    parser.add_argument("--resolution", default="1920:1080", help="Output video resolution, e.g., 1920:1080 (default: 1920:1080)")
    parser.add_argument("--framerate", type=int, default=30, help="Output video framerate (default: 30)")
    parser.add_argument("--codec", default="libx265", help="Video codec to use (default: libx265 for H.265)")

    args = parser.parse_args()

    # Handle S3/local input and output
    temp_input_dir = None
    temp_output_dir = None

    try:
        # Prepare input directory
        if is_s3_path(args.input_dir):
            temp_input_dir = tempfile.mkdtemp()
            sync_s3_to_local(args.input_dir, temp_input_dir)
            input_dir = temp_input_dir
        else:
            input_dir = args.input_dir

        # Prepare output directory
        if is_s3_path(args.output_dir):
            temp_output_dir = tempfile.mkdtemp()
            output_dir = temp_output_dir
        else:
            output_dir = args.output_dir

        # Convert videos
        convert_to_mp4(input_dir, output_dir, args.resolution, args.framerate, args.codec)

        # If output is S3, sync back
        if is_s3_path(args.output_dir):
            sync_local_to_s3(output_dir, args.output_dir)

    finally:
        # Clean up temporary directories
        if temp_input_dir:
            shutil.rmtree(temp_input_dir)
        if temp_output_dir:
            shutil.rmtree(temp_output_dir)
