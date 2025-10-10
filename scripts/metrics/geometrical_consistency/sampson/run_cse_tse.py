#!/usr/bin/env python
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

"""
Direct CSE/TSE evaluation script that works with any video directory.
This version doesn't depend on eval_list.json.
Completely independent version - all dependencies are local.
"""
import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from baselines_data_direct import SimpleClip, get_all_videos, get_data_direct
from mvbench.data.base import CameraView
from mvbench.metrics.sampson import (
    CrossViewSampsonMetric,
    SampsonFundamentalMethod,
    TemporalSampsonMetric,
)
from mvbench.utils.camera_model import IdealPinholeCamera
from tqdm import tqdm


def evaluate_single_video(video_path: Path, output_dir: Path, verbose: bool = False):
    """Evaluate Cross-view and Temporal Sampson Errors for a single video."""

    clip = SimpleClip.from_video_path(video_path)
    clip_output_dir = output_dir / "cse_tse"
    clip_output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Processing {clip.clip_id} from {video_path}")

    target_intrinsic = IdealPinholeCamera(fov_x_deg=120.0, width=960, height=540)
    temporal_metric = TemporalSampsonMetric(
        fundamental_method=SampsonFundamentalMethod.UNKNOWN_INTRINSIC,
        target_intrinsic=target_intrinsic,
    )
    cross_metric = CrossViewSampsonMetric(
        fundamental_method=SampsonFundamentalMethod.UNKNOWN_INTRINSIC,
        target_intrinsic=target_intrinsic,
    )

    try:
        data = get_data_direct(video_path)
    except Exception as e:
        print(f"Error loading data for {clip.clip_id}: {e}")
        return None

    eval_results = {"C": {}, "T": {}}
    visualize_results = {}

    # Temporal consistency evaluation for each view
    for view in [CameraView.FRONT, CameraView.CROSS_LEFT, CameraView.CROSS_RIGHT]:
        try:
            res_mean, res_median = temporal_metric.compute(data, view)
            # Handle torch tensors or numpy arrays
            if res_mean is not None and res_median is not None:
                # Convert to numpy if it's a torch tensor
                if hasattr(res_mean, "cpu"):
                    res_mean = res_mean.cpu().numpy()
                if hasattr(res_median, "cpu"):
                    res_median = res_median.cpu().numpy()

                eval_results["T"][view.value] = {
                    "mean": float(np.nanmean(res_mean)),
                    "median": float(np.nanmedian(res_median)),
                    "frame_values": res_median.tolist(),
                }
                visualize_results[f"T-{view.value}"] = res_median
        except Exception as e:
            if verbose:
                import traceback

                print(f"  Warning: Error computing temporal metric for {view.value}:")
                traceback.print_exc()
            continue

    # Cross-view consistency evaluation for view pairs
    for view_pair in [
        (CameraView.FRONT, CameraView.CROSS_RIGHT),
        (CameraView.FRONT, CameraView.CROSS_LEFT),
    ]:
        try:
            res_mean, res_median = cross_metric.compute(data, view_pair)
            pair_key = f"{view_pair[0].value}-{view_pair[1].value}"
            # Handle torch tensors or numpy arrays
            if res_mean is not None and res_median is not None:
                # Convert to numpy if it's a torch tensor
                if hasattr(res_mean, "cpu"):
                    res_mean = res_mean.cpu().numpy()
                if hasattr(res_median, "cpu"):
                    res_median = res_median.cpu().numpy()

                eval_results["C"][pair_key] = {
                    "mean": float(np.nanmean(res_mean)),
                    "median": float(np.nanmedian(res_median)),
                    "frame_values": res_median.tolist(),
                }
                visualize_results[f"C-{pair_key}"] = res_median
        except Exception as e:
            if verbose:
                import traceback

                print(f"  Warning: Error computing cross-view metric for {view_pair}:")
                traceback.print_exc()
            continue

    # Generate visualization
    if visualize_results:
        plt.figure(figsize=(12, 6))
        plt.grid(True, alpha=0.3)
        plt.ylabel("Sampson Error √(pixels)", fontsize=12)
        plt.xlabel("Frame", fontsize=12)
        plt.title(f"CSE/TSE Evaluation: {clip.clip_id}", fontsize=14)

        # Plot with different colors for CSE vs TSE
        for key, value in visualize_results.items():
            if value is not None:
                if key.startswith("T"):
                    plt.plot(value, label=key, linestyle="-", alpha=0.8)
                else:  # Cross-view
                    plt.plot(value, label=key, linestyle="--", alpha=0.8)

        plt.legend(loc="upper right")
        plt.ylim(0, 10)
        plt.tight_layout()
        plt.savefig(clip_output_dir / f"{clip.clip_id}.png", dpi=100)
        plt.close()

    # Save numerical results
    result_path = clip_output_dir / f"{clip.clip_id}.json"
    with open(result_path, "w") as f:
        json.dump(
            {
                "video_path": str(video_path),
                "clip_id": clip.clip_id,
                "chunk_id": clip.chunk_id,
                "results": eval_results,
            },
            f,
            indent=2,
        )

    return eval_results


def compute_aggregate_stats(all_results: dict) -> dict:
    """Compute aggregate statistics across all videos."""

    stats = {"num_videos": len(all_results), "temporal": {}, "cross_view": {}}

    # Collect all values for aggregation
    temporal_values = {view: [] for view in ["front", "cross_left", "cross_right"]}
    cross_values = {"front-cross_right": [], "front-cross_left": []}

    for video_id, results in all_results.items():
        if results and "T" in results:
            for view, values in results["T"].items():
                if values and "median" in values and values["median"] is not None:
                    temporal_values[view].append(values["median"])

        if results and "C" in results:
            for pair, values in results["C"].items():
                if values and "median" in values and values["median"] is not None:
                    cross_values[pair].append(values["median"])

    # Compute aggregate stats for temporal
    for view, values in temporal_values.items():
        if values:
            stats["temporal"][view] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "count": len(values),
            }

    # Compute aggregate stats for cross-view
    for pair, values in cross_values.items():
        if values:
            stats["cross_view"][pair] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "count": len(values),
            }

    # Overall averages
    all_temporal = []
    for values in temporal_values.values():
        all_temporal.extend(values)

    all_cross = []
    for values in cross_values.values():
        all_cross.extend(values)

    if all_temporal:
        stats["temporal"]["overall"] = {
            "mean": float(np.mean(all_temporal)),
            "median": float(np.median(all_temporal)),
            "std": float(np.std(all_temporal)),
        }

    if all_cross:
        stats["cross_view"]["overall"] = {
            "mean": float(np.mean(all_cross)),
            "median": float(np.median(all_cross)),
            "std": float(np.std(all_cross)),
        }

    return stats


def main():
    parser = ArgumentParser(
        description="Direct CSE/TSE evaluation for video directories"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input directory containing video files (or single video file)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./eval-output"),
        help="Output directory for results (default: ./eval-output)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.mp4",
        help="File pattern for video files (default: *.mp4)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    # Handle single file or directory
    if args.input.is_file():
        videos = [args.input]
        print(f"Processing single video: {args.input}")
    else:
        videos = get_all_videos(args.input, args.pattern)
        print(f"Found {len(videos)} videos in {args.input}")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Process all videos
    all_results = {}
    failed_videos = []

    with tqdm(videos, desc="Processing videos") as pbar:
        for video_path in pbar:
            pbar.set_description(f"Processing {video_path.name}")
            try:
                result = evaluate_single_video(video_path, args.output, args.verbose)
                if result is not None:
                    clip = SimpleClip.from_video_path(video_path)
                    all_results[clip.clip_id] = result
                else:
                    failed_videos.append(str(video_path))
            except Exception as e:
                print(f"\nError processing {video_path}: {e}")
                failed_videos.append(str(video_path))

    # Compute and save aggregate statistics
    if all_results:
        stats = compute_aggregate_stats(all_results)

        # Save aggregate stats
        stats_path = args.output / "aggregate_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Successfully processed: {len(all_results)}/{len(videos)} videos")

        if "temporal" in stats and "overall" in stats["temporal"]:
            print(f"\nTemporal Sampson Error (TSE):")
            print(f"  Mean: {stats['temporal']['overall']['mean']:.3f} pixels")
            print(f"  Median: {stats['temporal']['overall']['median']:.3f} pixels")
            print(f"  Std: {stats['temporal']['overall']['std']:.3f} pixels")

        if "cross_view" in stats and "overall" in stats["cross_view"]:
            print(f"\nCross-view Sampson Error (CSE):")
            print(f"  Mean: {stats['cross_view']['overall']['mean']:.3f} pixels")
            print(f"  Median: {stats['cross_view']['overall']['median']:.3f} pixels")
            print(f"  Std: {stats['cross_view']['overall']['std']:.3f} pixels")

        print(f"\nDetailed results saved to: {args.output}")
        print(f"Aggregate statistics saved to: {stats_path}")

        if failed_videos:
            print(f"\nFailed to process {len(failed_videos)} videos:")
            for video in failed_videos[:5]:
                print(f"  - {video}")
            if len(failed_videos) > 5:
                print(f"  ... and {len(failed_videos) - 5} more")
    else:
        print("\nNo videos were successfully processed.")


if __name__ == "__main__":
    main()
