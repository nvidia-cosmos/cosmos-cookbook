#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = "==3.8.*"
# dependencies = [
#   "nuscenes-devkit",
#   "opencv-python",
# ]
# [tool.uv]
# exclude-newer = "2025-07-31T00:00:00Z"
# ///

from __future__ import annotations

import argparse
import concurrent.futures
import os
import shlex
import shutil
import subprocess
import sys
import tempfile

import cv2
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm


def compile_nuscenes_videos(
    nusc: NuScenes,
    output_dir: str,
    channels: list,
    ffmpeg_args: list[str],
    keep_going: bool,
    num_workers: int,
):
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "Error: ffmpeg not found. Please install it: https://ffmpeg.org/download.html"
        )

    def fn(scene: dict, channel: str):
        scene_name = scene["name"]
        with tempfile.TemporaryDirectory() as tmp_dir:
            _compile_nuscenes_video(
                nusc=nusc,
                scene=scene,
                channel=channel,
                output_dir=os.path.join(output_dir, scene_name),
                tmp_dir=tmp_dir,
                ffmpeg_args=ffmpeg_args,
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for scene in nusc.scene:
            for channel in channels:
                futures[executor.submit(fn, scene, channel)] = (scene["name"], channel)
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            scene_name, channel = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing scene '{scene_name}' channel '{channel}': {e}")
                if not keep_going:
                    sys.exit(1)
        executor.shutdown(wait=True)


def _compile_nuscenes_video(
    nusc: NuScenes,
    scene: dict,
    channel: str,
    output_dir: str,
    tmp_dir: str,
    ffmpeg_args: list[str],
):
    first_sample = nusc.get("sample", scene["first_sample_token"])
    frame_paths = []
    if channel not in first_sample["data"]:
        raise ValueError("Channel not found in the first sample.")
    sample = first_sample
    sample_token = sample["data"][channel]
    while sample_token:
        sample = nusc.get("sample_data", sample_token)
        image_path = os.path.join(nusc.dataroot, sample["filename"])
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at '{image_path}'")
        frame_paths.append(image_path)
        sample_token = sample["next"]

    # Compute frames per second
    num_samples = len(frame_paths)
    if num_samples < 2:
        raise RuntimeError(f"Too few samples {num_samples} < 2")
    fps = round(
        1e6 * (num_samples - 1) / (sample["timestamp"] - first_sample["timestamp"])
    )

    # Compile videos for each camera channel in this scene using ffmpeg
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, f"{channel}.mp4")

    # Determine image dimensions from the first frame
    first_frame_img = cv2.imread(frame_paths[0])
    height, width, _ = first_frame_img.shape

    # Create frame list
    list_path = os.path.join(tmp_dir, "list.txt")
    with open(list_path, "w") as f:
        for frame_path in frame_paths:
            f.write(f"file '{frame_path}'\n")

    # Run ffmpeg
    command = [
        "ffmpeg",
        "-y",  # Overwrite output files without asking
        "-f",
        "concat",  # Use the concat demuxer
        "-safe",
        "0",  # Allow absolute paths in concat list
        "-i",
        list_path,  # Input is the list file
        "-vf",
        f"scale={width}:{height},format=yuv420p",  # Ensure compatible pixel format
        "-c:v",
        "libx264",
        "-r",
        str(fps),  # Output framerate
        "-pix_fmt",
        "yuv420p",  # Recommended for broader compatibility
        *ffmpeg_args,
        video_path,
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"COMMAND: {shlex.join(command)}\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}\n"
        )


def main():
    # Define common camera channels for choices
    ALL_CAMERA_CHANNELS = [
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    parser = argparse.ArgumentParser(
        description="Compile NuScenes camera frames into separate video clips using FFmpeg.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # Show default values in help
    )

    parser.add_argument(
        "dataroot", type=str, help="Path to the NuScenes dataset root directory."
    )

    parser.add_argument(
        "output_dir", type=str, help="Base directory to save the output videos."
    )
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="NuScenes dataset version  (e.g., 'v1.0-mini', 'v1.0-trainval', 'v1.0-test').",
    )
    parser.add_argument(
        "--cameras",
        nargs="*",
        default=ALL_CAMERA_CHANNELS,
        choices=ALL_CAMERA_CHANNELS,
        help="Space-separated list of camera channels to process. "
        "Defaults to all standard cameras if not specified.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=23,
        choices=range(0, 52),
        metavar="[0-51]",
        help="H.264 Constant Rate Factor (CRF). Lower values mean higher quality (and larger files). "
        "0 is lossless, 23 is a good default balance, 28 is lower quality.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue processing even if some scenes fail.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of workers to use for processing.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    compile_nuscenes_videos(
        nusc=nusc,
        output_dir=args.output_dir,
        channels=args.cameras,
        ffmpeg_args=[
            "-crf",
            str(args.crf),
        ],
        keep_going=args.keep_going,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
