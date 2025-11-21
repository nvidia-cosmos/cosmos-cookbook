#!/usr/bin/env python3
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

"""Create training and evaluation datasets from local video files.

This script creates train/eval splits from local video files with human-labeled quality scores.
Supports stratified splitting to maintain label distribution in both sets.
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

try:
    import datasets
    import pandas as pd
    import yaml
    from rich import print
    from tqdm import tqdm
except ImportError as e:
    print(f"Error: Missing required package: {e}")
    print("Please install dependencies:")
    print("  pip install datasets pandas openpyxl pyyaml rich tqdm")
    sys.exit(1)


def extract_filename_from_url(url: str) -> Optional[str]:
    """Extract local filename from URL.
    
    This function extracts the filename from video URLs. Customize this
    function based on your URL structure. Examples:
    
    - Simple: Extract just the filename
      url = "https://example.com/videos/video_001.mp4" -> "video_001.mp4"
    
    - Complex: Parse structured paths and reconstruct filenames
      url = "https://example.com/action/wave/segment_01/video_001.mp4"
      -> "wave_segment_01_video_001.mp4"
    """
    # Example 1: Simple filename extraction
    # Uncomment and modify for your use case:
    # return url.split('/')[-1]  # Returns last part of URL
    
    # Example 2: Custom pattern matching
    # Modify this pattern to match your URL structure
    pattern = r'([\w]+)/(com_\d+_\d+_[a-f0-9]+)_segment_(\d+)_left/gpu_(\d+)/video_(\d+)/output\.mp4'
    match = re.search(pattern, url)
    if match:
        action = match.group(1)
        timestamp_id = match.group(2)
        segment = match.group(3)
        gpu = match.group(4)
        video = match.group(5)
        filename = f'{action}_{timestamp_id}_segment_{segment}_left_gpu_{gpu}_video_{video}_output.mp4'
        return filename
    
    # Fallback: Try extracting just the filename
    if '/' in url:
        return url.split('/')[-1]
    
    return None


def balance_dataset_labels(dataset: datasets.Dataset, verbose: bool = True) -> datasets.Dataset:
    """Balance dataset by resampling so each label appears the same number of times."""
    random.seed(42)

    # Extract PC labels and group samples by label
    label_to_indices = {}
    for i, sample in enumerate(dataset):
        pc_score = sample.get("pc")
        if pc_score is not None:
            if pc_score not in label_to_indices:
                label_to_indices[pc_score] = []
            label_to_indices[pc_score].append(i)

    if verbose:
        print("\n📊 Original label distribution:")
        for label in sorted(label_to_indices.keys()):
            count = len(label_to_indices[label])
            print(f"  Label {label}: {count} samples")

    # target samples per label is the average number of samples per label
    target_samples_per_label = len(dataset) // len(label_to_indices)

    if verbose:
        print(f"\n🎯 Target samples per label: {target_samples_per_label}")

    # Resample each label to target count
    balanced_indices = []
    for label, indices in label_to_indices.items():
        if len(indices) >= target_samples_per_label:
            if verbose:
                print(f"Downsampling label {label} from {len(indices)} to {target_samples_per_label}")
            selected_indices = random.sample(indices, target_samples_per_label)
        else:
            if verbose:
                print(f"Upsampling label {label} from {len(indices)} to {target_samples_per_label}")
            selected_indices = random.choices(indices, k=target_samples_per_label)

        balanced_indices.extend(selected_indices)

    # Shuffle the balanced indices
    random.shuffle(balanced_indices)

    # Create new balanced dataset
    balanced_data = [dataset[i] for i in balanced_indices]
    balanced_dataset = datasets.Dataset.from_list(balanced_data)

    if verbose:
        print("\n📊 Final balanced label distribution:")
        final_label_counts = Counter(sample["pc"] for sample in balanced_dataset)
        for label in sorted(final_label_counts.keys()):
            print(f"  Label {label}: {final_label_counts[label]} samples")
        print(f"\n✅ Dataset balanced: {len(dataset)} → {len(balanced_dataset)} samples")

    return balanced_dataset


def stratified_split(dataset: datasets.Dataset, eval_size: float = 0.1, random_seed: int = 42):
    """Split dataset into train/eval while maintaining label distribution."""
    random.seed(random_seed)
    
    # Group indices by label
    label_to_indices = {}
    for i, sample in enumerate(dataset):
        pc_score = sample.get("pc")
        if pc_score is not None:
            if pc_score not in label_to_indices:
                label_to_indices[pc_score] = []
            label_to_indices[pc_score].append(i)
    
    train_indices = []
    eval_indices = []
    
    # Split each label proportionally
    for label, indices in label_to_indices.items():
        random.shuffle(indices)
        split_point = int(len(indices) * (1 - eval_size))
        train_indices.extend(indices[:split_point])
        eval_indices.extend(indices[split_point:])
    
    # Shuffle to mix labels
    random.shuffle(train_indices)
    random.shuffle(eval_indices)
    
    # Create datasets
    train_data = [dataset[i] for i in train_indices]
    eval_data = [dataset[i] for i in eval_indices]
    
    train_dataset = datasets.Dataset.from_list(train_data)
    eval_dataset = datasets.Dataset.from_list(eval_data)
    
    return train_dataset, eval_dataset


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output_dir", type=str, help="Output directory for train/eval datasets.", required=True
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Path to data directory containing videos, prompts, and Excel file.",
    )
    parser.add_argument(
        "--excel_file",
        type=str,
        default="transfer25_human_labeled.xlsx",
        help="Excel file with video URLs and labels.",
    )
    parser.add_argument(
        "--eval_size",
        type=float,
        default=0.1,
        help="Fraction of data to use for evaluation (default: 0.1 = 10%%).",
    )
    parser.add_argument(
        "--balance_labels",
        action="store_true",
        help="Balance dataset labels before splitting.",
    )
    parser.add_argument(
        "--scale_labels",
        action="store_true",
        help="Map binary labels (0,1) to 1-5 scale: 0→1 (bad), 1→5 (good).",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    # Set random seed
    random.seed(args.random_seed)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve paths
    data_dir = Path(args.data_dir).resolve()
    video_dir = data_dir / "transfer1_generated_videos"
    prompt_dir = data_dir / "prompts"
    excel_path = data_dir / args.excel_file

    print(f"📂 Data directory: {data_dir}")
    print(f"📹 Video directory: {video_dir}")
    print(f"📝 Prompt directory: {prompt_dir}")
    print(f"📊 Excel file: {excel_path}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎲 Random seed: {args.random_seed}")
    print(f"📊 Eval size: {args.eval_size * 100:.0f}%")

    # Read Excel file
    print("\n📖 Reading Excel file...")
    df = pd.read_excel(excel_path, skiprows=1, names=["video_url", "label"])
    print(f"Found {len(df)} labeled videos")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")

    # Extract filenames from URLs
    df["filename"] = df["video_url"].apply(extract_filename_from_url)
    missing_filenames = df["filename"].isna().sum()
    if missing_filenames > 0:
        print(f"⚠️  Warning: {missing_filenames} URLs couldn't be parsed")
        df = df[df["filename"].notna()].reset_index(drop=True)

    # Verify files exist
    df["video_path"] = df["filename"].apply(lambda x: str(video_dir / x))
    df["prompt_path"] = df["filename"].apply(
        lambda x: str(prompt_dir / x.replace("_output.mp4", "_prompt.txt"))
    )
    
    df["video_exists"] = df["video_path"].apply(os.path.exists)
    df["prompt_exists"] = df["prompt_path"].apply(os.path.exists)

    missing_videos = (~df["video_exists"]).sum()
    missing_prompts = (~df["prompt_exists"]).sum()
    
    if missing_videos > 0:
        print(f"⚠️  Warning: {missing_videos} videos not found locally")
    if missing_prompts > 0:
        print(f"⚠️  Warning: {missing_prompts} prompts not found locally")
    
    df = df[df["video_exists"] & df["prompt_exists"]].reset_index(drop=True)
    print(f"✅ {len(df)} samples have both video and prompt files")

    # Scale labels if requested
    if args.scale_labels:
        print("\n🔄 Scaling labels: 0→1 (bad physics), 1→5 (good physics)")
        df["pc"] = df["label"].apply(lambda x: 1 if x == 0 else 5)
    else:
        df["pc"] = df["label"]

    # Read prompts
    print("\n📝 Reading prompt files...")
    prompts = []
    for prompt_path in tqdm(df["prompt_path"], desc="Loading prompts"):
        try:
            with open(prompt_path, "r") as f:
                prompt_text = f.read().strip()
                prompts.append(prompt_text)
        except Exception as e:
            print(f"⚠️  Error reading {prompt_path}: {e}")
            prompts.append("")
    
    df["caption"] = prompts

    # Create dataset
    dataset_dict = {
        "caption": df["caption"].tolist(),
        "video_url": df["video_path"].tolist(),
        "pc": df["pc"].tolist(),
    }

    full_dataset = datasets.Dataset.from_dict(dataset_dict)
    print(f"\n📦 Created full dataset with {len(full_dataset)} samples")

    # Balance if requested (before splitting)
    if args.balance_labels:
        print("\n⚖️  Balancing dataset labels...")
        full_dataset = balance_dataset_labels(full_dataset)

    # Perform stratified split
    print(f"\n✂️  Splitting dataset: {(1-args.eval_size)*100:.0f}% train, {args.eval_size*100:.0f}% eval")
    train_dataset, eval_dataset = stratified_split(full_dataset, eval_size=args.eval_size, random_seed=args.random_seed)

    # Print split statistics
    print(f"\n📊 Split Statistics:")
    print(f"  Train: {len(train_dataset)} samples")
    train_label_counts = Counter(train_dataset["pc"])
    for label in sorted(train_label_counts.keys()):
        print(f"    Label {label}: {train_label_counts[label]} samples ({train_label_counts[label]/len(train_dataset)*100:.1f}%)")
    
    print(f"\n  Eval: {len(eval_dataset)} samples")
    eval_label_counts = Counter(eval_dataset["pc"])
    for label in sorted(eval_label_counts.keys()):
        print(f"    Label {label}: {eval_label_counts[label]} samples ({eval_label_counts[label]/len(eval_dataset)*100:.1f}%)")

    # Save datasets
    train_path = output_dir / "train"
    eval_path = output_dir / "eval"
    
    print(f"\n💾 Saving datasets...")
    train_dataset.save_to_disk(str(train_path))
    eval_dataset.save_to_disk(str(eval_path))
    
    print(f"✅ Train dataset saved to: {train_path}")
    print(f"✅ Eval dataset saved to: {eval_path}")

    # Save split info
    split_info = {
        "total_samples": len(full_dataset),
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "eval_fraction": args.eval_size,
        "random_seed": args.random_seed,
        "balanced": args.balance_labels,
        "scaled_labels": args.scale_labels,
        "train_label_distribution": dict(train_label_counts),
        "eval_label_distribution": dict(eval_label_counts),
    }
    
    split_info_path = output_dir / "split_info.json"
    with open(split_info_path, "w") as f:
        json.dump(split_info, f, indent=2)
    
    print(f"📄 Split info saved to: {split_info_path}")

    print("\n" + "="*80)
    print("✅ DATASET CREATION COMPLETE!")
    print("="*80)
    print(f"\nDataset locations:")
    print(f"  Train: {train_path}")
    print(f"  Eval:  {eval_path}")
    print(f"\nTo load in Python:")
    print(f"  import datasets")
    print(f"  train_ds = datasets.load_from_disk('{train_path}')")
    print(f"  eval_ds = datasets.load_from_disk('{eval_path}')")
    print("="*80)


if __name__ == "__main__":
    main()


