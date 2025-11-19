#!/usr/bin/env -S uv run --script
#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "cosmos-reason1-utils",
#   "datasets",
#   "rich",
#   "tqdm",
# ]
# [tool.uv]
# exclude-newer = "2025-07-31T00:00:00Z"
# [tool.uv.sources]
# cosmos-reason1-utils = {path = "../../../cosmos_reason1_utils", editable = true}
# ///

"""Create dataset from local videos and pickle-based prompts.

This script creates a HuggingFace dataset from local video files and prompts
stored in pickle format, matching the output format of download_videophy2.py.
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import List, Optional, Union

import sys
sys.path.insert(0, str(Path.home() / "repos" / "cosmos-reason1" / "cosmos_reason1_utils" / "src"))
import datasets
from cosmos_reason1_utils.text import create_conversation
from rich import print
from tqdm import tqdm
import subprocess


def load_pickle(pickle_path: Union[str, Path]) -> dict:
    """Load data from pickle file.

    Args:
        pickle_path: Path to pickle file

    Returns:
        Loaded pickle data
    """
    with open(pickle_path, "rb") as f:
        return pickle.load(f)

def check_video(video_path):
    """Check if video is valid."""
    if not os.path.exists(video_path):
        return False
    
    try:
        subprocess.run(
            ['ffmpeg', '-v', 'error', '-i', video_path, '-f', 'null', '-'],
            check=True,
            capture_output=True
        )
        return True
    except subprocess.CalledProcessError:
        return False


def find_videos(video_dir: Path, extensions: tuple = (".mp4", ".avi", ".mov", ".mkv")) -> List[Path]:
    """Find all video files in a directory.

    Args:
        video_dir: Directory to search for videos
        extensions: Tuple of video file extensions to look for

    Returns:
        List of video file paths
    """
    videos = []
    for ext in extensions:
        videos.extend(video_dir.glob(f"*{ext}"))
    return sorted(videos)

def remove_timestamps_from_events(events_text):
    """
    Remove timestamp information from event strings.
    
    Converts:
        'Event 1: <1.6> <4.9> grasping the first piece'
    To:
        'Event 1: grasping the first piece'
    
    Args:
        events_text (str): String containing events with timestamps, separated by newlines
    
    Returns:
        str: String with timestamps removed
    """
    import re
    
    lines = events_text.strip().split('\n')
    cleaned_events = []
    
    for line in lines:
        if line.strip():
            # Pattern to match: 'Event N: <time1> <time2> description'
            # and replace with: 'Event N: description'
            pattern = r'(Event\s+\d+:)\s*<[\d.]+>\s*<[\d.]+>\s*(.+)'
            match = re.match(pattern, line.strip())
            
            if match:
                event_label = match.group(1)  # 'Event 1:'
                description = match.group(2)   # 'grasping the first piece'
                cleaned_line = f"{event_label} {description}"
                cleaned_events.append(cleaned_line)
            else:
                # If pattern doesn't match, keep original line
                cleaned_events.append(line.strip())
    
    return '\n'.join(cleaned_events)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory path for the HuggingFace dataset.",
    )

    parser.add_argument(
        "--prompts_pickle",
        type=str,
        required=True,
        help="Path to pickle file containing prompts. Expected format: "
        "{'system_prompt': str, 'user_prompt': str, 'responses': List[str]} "
        "where responses[i] corresponds to the i-th video.",
    )

    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Directory containing video files.",
    )

    args = parser.parse_args()

    # Validate input arguments
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts from pickle
    print(f"📦 Loading prompts from: {args.prompts_pickle}")
    prompts_data = load_pickle(args.prompts_pickle)
    
    # Validate pickle format
    required_keys = ["system_prompt", "responses"]
    missing_keys = [key for key in required_keys if key not in prompts_data]
    if missing_keys:
        raise ValueError(
            f"Pickle file missing required keys: {missing_keys}. "
            f"Expected keys: {required_keys}"
        )
    
    system_prompt = prompts_data["system_prompt"]
    # user_prompt = prompts_data["user_prompt"]
    responses = prompts_data["responses"]
    
    print(f"✅ Loaded prompts with {len(responses)} responses")
    print(f"   System prompt: {system_prompt[:100]}...")

    # Process samples
    processed_samples = []
    failed_count = 0
    skipped_large_videos = 0
    MAX_VIDEO_SIZE_MB = 5
    MAX_VIDEO_SIZE_BYTES = MAX_VIDEO_SIZE_MB * 1024 * 1024  # 5 MB in bytes

    print("\n🔄 Processing samples...")
    for demo_name, response in tqdm(responses.items()):
        # Construct video path: remove underscore from demo_name
        # e.g., "coffee_d0_demo_0" -> "coffee_d0_demo0_agentview.mp4"
        video_name = demo_name.replace("demo_", "demo") + "_agentview.mp4"
        video_path = os.path.join(args.video_dir, video_name)
        
        if not os.path.exists(video_path):
            continue
        
        # if not check_video(video_path):
        #     failed_count += 1
        #     continue
        
        # Check video file size
        video_size_bytes = os.path.getsize(video_path)
        video_size_mb = video_size_bytes / (1024 * 1024)
        
        if video_size_bytes > MAX_VIDEO_SIZE_BYTES:
            print(f"[yellow]Skipping {video_name}: size {video_size_mb:.2f} MB exceeds {MAX_VIDEO_SIZE_MB} MB limit[/yellow]")
            skipped_large_videos += 1
            continue
        
        try:
            user_prompt = remove_timestamps_from_events(response)
            # Create conversation
            conversation = create_conversation(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                videos=[str(video_path)],
                response=f"<answer>\n{response}\n</answer>",
            )

            processed_samples.append({"conversations": json.dumps(conversation)})
        except Exception as e:
            print(f"[red]Failed to process sample {demo_name}: {e}[/red]")
            failed_count += 1

    print(f"\n✅ Successfully processed: {len(processed_samples)} samples")
    print(f"⏭️  Skipped (too large): {skipped_large_videos} samples")
    print(f"❌ Failed: {failed_count} samples")

    if not processed_samples:
        print("❌ No samples were successfully processed!")
        return

    # Create and save dataset
    final_dataset = datasets.Dataset.from_generator(lambda: processed_samples)
    print("\n📊 Final dataset:", final_dataset)
    print("\n📝 Sample conversation:")
    print(processed_samples[0]["conversations"])

    final_dataset.save_to_disk(str(output_dir))
    print(f"\n💾 Dataset saved to: {output_dir}")


if __name__ == "__main__":
    main()


