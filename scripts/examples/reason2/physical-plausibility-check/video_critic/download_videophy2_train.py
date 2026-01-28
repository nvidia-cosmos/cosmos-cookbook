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

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "cosmos-reason2-utils[inference]",
#   "datasets",
#   "pyyaml",
#   "rich",
#   "tqdm",
#   "requests",
# ]
# [tool.uv.sources]
# cosmos-reason2-utils = { path = "../../cosmos_reason2_utils", editable = true }
# ///

"""Download video physics datasets and create conversations for physics reasoning.

Default dataset: https://huggingface.co/datasets/videophysics/videophy2_train
"""

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import datasets
import yaml
from rich import print
from tqdm import tqdm
import requests

from cosmos_reason2_utils.text import create_conversation

ROOT = Path(__file__).parents[2]
VIDEO_DIR = "/mnt/nfs/shunz/video_assets/"


def download_video(video_url: str, video_index: int, dataset_name: str, split: str) -> Optional[str]:
    """Download video from URL to shared video directory.
    
    Args:
        video_url: URL of the video to download
        video_index: Index of the video in the dataset
        dataset_name: Name of the dataset
        split: Dataset split (train, test, etc.)
        
    Returns:
        Local path to downloaded video, or None if download failed
    """
    try:
        # Create dataset-specific video directory
        dataset_name_clean = dataset_name.replace('/', '_')
        dataset_split_dir = f"{dataset_name_clean}_{split}"
        video_dir = os.path.join(VIDEO_DIR, dataset_split_dir)
        os.makedirs(video_dir, exist_ok=True)
        
        # Create filename based on video index
        filename = f"{video_index}.mp4"
        local_path = os.path.join(video_dir, filename)
        
        # Skip if already downloaded
        if os.path.exists(local_path):
            return local_path
            
        # Download the video with better error handling for parallel processing
        response = requests.get(video_url, stream=True, timeout=60, 
                               headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        
        # Write to file
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        return local_path
        
    except Exception as e:
        print(f"Failed to download {video_url}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, help="Output huggingface dataset path.", required=True)
    parser.add_argument("--dataset", type=str, default="videophysics/videophy2_train", 
                       help="HuggingFace dataset name to download.")
    parser.add_argument("--split", type=str, default="train", help="Split to download.")
    parser.add_argument("--prompt_path", type=str, default="prompts/video_reward.yaml", help="Prompt to use.")
    parser.add_argument("--workers", type=int, default=8, 
                       help="Number of workers for parallel video downloads.")
    parser.add_argument("--first_n", type=int, default=None,
                       help="Only use the first n samples from the dataset.")
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prompt template
    prompt_path = args.prompt_path
    if not os.path.isabs(prompt_path):
        prompt_path = os.path.join(ROOT, prompt_path)
    
    with open(prompt_path, 'r') as f:
        prompt_config = yaml.safe_load(f)
    
    system_prompt = prompt_config.get('system_prompt', '')
    user_prompt = prompt_config.get('user_prompt', '')

    # Load raw dataset
    dataset = datasets.load_dataset(
        args.dataset, split=args.split
    )
    print(f"Loaded dataset with {len(dataset)} samples")
    print(f"Using {args.workers} workers for parallel downloads")
    print("Dataset features:", dataset.features)
    print("Sample:", dataset[0])
    
    # Limit to first n samples if requested
    if args.first_n is not None:
        if args.first_n > len(dataset):
            print(f"⚠️ Warning: --first_n ({args.first_n}) is larger than dataset size ({len(dataset)}). Using all samples.")
        else:
            print(f"\n📌 Limiting dataset to first {args.first_n} samples")
            dataset = dataset.select(range(args.first_n))
            print(f"Dataset size after limiting: {len(dataset)} samples")
    
    # Process dataset
    def process_sample(sample_with_index: tuple) -> Optional[dict]:
        """Process a single sample: download video and create conversation."""
        video_index, sample = sample_with_index
        video_url = sample.get("video_url")
        pc_score = sample.get("pc")  # Physics commonsense score
        
        if not video_url or pc_score is None:
            print(f"⚠️ Skipping sample with missing video_url or pc")
            return None
            
        # Download video to shared directory
        local_video_path = download_video(video_url, video_index + 1, args.dataset, args.split)
        if not local_video_path:
            return None
            
        # Validate downloaded file exists
        if not Path(local_video_path).is_file():
            print(f"❌ Downloaded video file not found: {local_video_path}")
            return None

        # Create conversation
        conversation = create_conversation(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            videos=[local_video_path],
            response=f"{pc_score}",
        )
        
        return {
            # Store conversation as string
            "conversations": json.dumps(conversation),
        }

    # Use ThreadPoolExecutor with map (automatically preserves order)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(
            executor.map(process_sample, enumerate(dataset)),
            total=len(dataset),
            desc=f"Processing samples with {args.workers} workers"
        ))
    
    # Filter out None results (failed samples)
    processed_samples = [result for result in results if result is not None]
    failed_count = len(results) - len(processed_samples)
    
    print(f"Successfully processed: {len(processed_samples)} samples")
    print(f"Failed: {failed_count} samples")
    
    if not processed_samples:
        print("❌ No samples were successfully processed!")
        return
    
    # Create new dataset
    final_dataset = datasets.Dataset.from_generator(lambda: processed_samples)
    print("Final dataset:", final_dataset)
    for i in range(min(3, len(processed_samples))):
        print(f"Sample conversation {i+1}:", processed_samples[i]["conversations"])
    
    # Save dataset
    final_dataset.save_to_disk(str(output_dir))
    print(f"Dataset saved to: {output_dir}")


if __name__ == "__main__":
    main()
