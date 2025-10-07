#!/usr/bin/env -S uv run --script
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
#   "pyyaml",
#   "rich",
#   "tqdm",
#   "requests",
#   "urllib3",
# ]
# [tool.uv]
# exclude-newer = "2025-07-31T00:00:00Z"
# [tool.uv.sources]
# cosmos-reason1-utils = {path = "../../../cosmos_reason1_utils", editable = true}
# ///

"""Download video physics datasets and create conversations for physics reasoning.

Default dataset: https://huggingface.co/datasets/videophysics/videophy2_train
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import datasets
import requests
import yaml
from cosmos_reason1_utils.text import PromptConfig, create_conversation
from rich import print
from tqdm import tqdm

ROOT = Path(__file__).parents[3]
VIDEO_DIR = "video_data/"


def download_video(video_url: str, video_index: int) -> Optional[str]:
    """Download video from URL to shared video directory.

    Args:
        video_url: URL of the video to download
        video_index: Index of the video in the dataset

    Returns:
        Local path to downloaded video, or None if download failed
    """
    try:
        os.makedirs(VIDEO_DIR, exist_ok=True)

        filename = f"{video_index}.mp4"
        local_path = os.path.join(VIDEO_DIR, filename)

        if os.path.exists(local_path):
            print(f"File already exists, reusing: {filename}")
            return local_path

        print(f"Downloading: {video_url}")
        print(f"Saving to: {local_path}")

        response = requests.get(
            video_url, stream=True, timeout=60, headers={"User-Agent": "Mozilla/5.0"}
        )
        response.raise_for_status()

        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Successfully downloaded: {filename}")
        return local_path

    except Exception as e:
        print(f"[red]Failed to download {video_url}: {e}[/red]")
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=str, help="Output huggingface dataset path.", required=True
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="videophysics/videophy2_train",
        help="HuggingFace dataset name to download.",
    )
    parser.add_argument("--split", type=str, default="train", help="Split to download.")
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of workers for parallel video downloads.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prompt template
    with open(f"{ROOT}/prompts/video_reward.yaml", "rb") as f:
        prompt_config = PromptConfig.model_validate(yaml.safe_load(f))
    system_prompt = prompt_config.system_prompt
    user_prompt = prompt_config.user_prompt

    # Load raw dataset
    dataset = datasets.load_dataset(args.dataset, split=args.split)
    print(f"Loaded dataset with {len(dataset)} samples")
    print(f"Using {args.workers} workers for parallel downloads")
    print("Dataset features:", dataset.features)
    print("Sample:", dataset[0])

    # Process dataset
    def process_sample(sample_with_index: tuple) -> Optional[dict]:
        """Process a single sample: download video and create conversation."""
        video_index, sample = sample_with_index
        video_url = sample.get("video_url")
        pc_score = sample.get("pc")  # Physics commonsense score

        if not video_url or pc_score is None:
            print(f"⚠️ Skipping sample with missing video_url or pc")
            return None

        # Download video
        local_video_path = download_video(video_url, video_index + 1)
        if not local_video_path:
            return None

        # Create conversation
        conversation = create_conversation(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            videos=[local_video_path],
            response=f"<answer>\n{pc_score}\n</answer>",
        )

        return {"conversations": json.dumps(conversation)}

    # Process samples in parallel
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(
            tqdm(
                executor.map(process_sample, enumerate(dataset)),
                total=len(dataset),
                desc=f"Processing samples with {args.workers} workers",
            )
        )

    # Filter out failed samples
    processed_samples = [result for result in results if result is not None]
    failed_count = len(results) - len(processed_samples)

    print(f"Successfully processed: {len(processed_samples)} samples")
    print(f"Failed: {failed_count} samples")

    if not processed_samples:
        print("❌ No samples were successfully processed!")
        return

    # Create and save dataset
    final_dataset = datasets.Dataset.from_generator(lambda: processed_samples)
    print("Final dataset:", final_dataset)
    print("Sample conversation:", processed_samples[0]["conversations"])

    final_dataset.save_to_disk(str(output_dir))
    print(f"Dataset saved to: {output_dir}")


if __name__ == "__main__":
    main()
