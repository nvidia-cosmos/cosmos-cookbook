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

"""
Compute normalization statistics for relative actions in dVRK datasets.

This script computes statistics (mean, std, min, max, q01, q99) for:
1. 20D relative actions (output of compute_rel_actions with 6D rotation format)
2. 16D observation states

The stats are saved to stats.json in the dataset's meta directory.

Usage:
    python compute_rel_action_stats.py --dataset-path /path/to/lerobot/dataset

Example:
    python compute_rel_action_stats.py \
        --dataset-path /lustre/fsw/portfolios/healthcareeng/users/nigeln/cache/huggingface/lerobot/jhu_lerobot/suturebot_lerobot
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.transform.state_action import (
    compute_rel_actions,
)


def compute_stats(data: np.ndarray) -> dict:
    """Compute statistics for a numpy array."""
    return {
        "mean": np.mean(data, axis=0).tolist(),
        "std": np.std(data, axis=0).tolist(),
        "min": np.min(data, axis=0).tolist(),
        "max": np.max(data, axis=0).tolist(),
        "q01": np.quantile(data, 0.01, axis=0).tolist(),
        "q99": np.quantile(data, 0.99, axis=0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute normalization stats for relative actions")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to LeRobot dataset")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: dataset_path/meta/stats.json)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    parquet_files = sorted(dataset_path.glob("data/*/*.parquet"))

    if not parquet_files:
        print(f"No parquet files found in {dataset_path / 'data'}")
        return

    print(f"Found {len(parquet_files)} parquet files")

    all_rel_actions = []
    all_states = []

    for pf in tqdm(parquet_files, desc="Processing episodes"):
        df = pd.read_parquet(pf)

        # Extract actions and compute relative actions
        actions = np.vstack(df["action"].values)  # [T, 20]
        rel_actions = compute_rel_actions(actions)  # [T-1, 20]
        all_rel_actions.append(rel_actions)

        # Extract states
        if "observation.state" in df.columns:
            states = np.vstack(df["observation.state"].values)  # [T, 16]
            all_states.append(states)

    # Stack all data
    all_rel_actions = np.vstack(all_rel_actions)
    print(f"Total relative actions: {all_rel_actions.shape}")

    # Compute stats
    stats = {
        "action": compute_stats(all_rel_actions),
    }

    if all_states:
        all_states = np.vstack(all_states)
        print(f"Total states: {all_states.shape}")
        stats["observation.state"] = compute_stats(all_states)

    # Save stats
    output_path = Path(args.output) if args.output else dataset_path / "meta" / "stats.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved stats to {output_path}")
    print(f"\nAction stats (20D relative with 6D rotation):")
    print(f"  mean: {stats['action']['mean'][:3]}... (showing first 3)")
    print(f"  std:  {stats['action']['std'][:3]}... (showing first 3)")

    if "observation.state" in stats:
        print(f"\nState stats (16D):")
        print(f"  mean: {stats['observation.state']['mean'][:3]}... (showing first 3)")
        print(f"  std:  {stats['observation.state']['std'][:3]}... (showing first 3)")


if __name__ == "__main__":
    main()
