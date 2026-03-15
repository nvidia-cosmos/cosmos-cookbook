#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Intended Script Location: cosmos_predict2/_src/predict2/action/inference/inference_dvrk.py

Action conditioned inference script for dVRK surgical post-training.

This script runs autoregressive video generation on episodes from LeRobot datasets.
For each episode, it:
1. Uses the first frame as conditioning
2. Generates 12 frames using the model with ground-truth actions
3. Uses the last predicted frame as conditioning for the next chunk
4. Stitches all chunks into a full episode video

Uses LeRobotDataset directly to ensure actions are transformed identically to training
(relative action computation + normalization via the transform pipeline).

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python cosmos_predict2/_src/predict2/action/inference/inference_dvrk.py \
        --experiment=cosmos_predict2p5_2B_action_conditioned_suturebot_13frame_4nodes_release_oss \
        --ckpt_path /path/to/checkpoint/model_ema_bf16.pt \
        --dataset_path /path/to/suturebot_lerobot \
        --save_root results/dvrk_eval \
        --data_split train \
        --episode_ids 0,1,2
"""

import argparse
import json
import os
from pathlib import Path

import mediapy
import numpy as np
import pandas as pd
import torch
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.dataset import (
    LeRobotDataset,
)
from cosmos_predict2._src.predict2.action.inference.inference_pipeline import (
    ActionVideo2WorldInference,
)
from loguru import logger

# Constants matching training config in groot_configs.py for dVRK
NUM_FRAMES = 13
TIMESTEP_INTERVAL = 3
CHUNK_SIZE = 12  # NUM_FRAMES - 1 (actions per window)


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the dVRK inference script."""
    parser = argparse.ArgumentParser(
        description="Action conditioned Cosmos-Predict 2.5 inference script"
    )

    # Model arguments
    parser.add_argument(
        "--experiment", type=str, required=True, help="Experiment config name"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to the checkpoint (.pt file)",
    )
    parser.add_argument(
        "--s3_cred", type=str, default="credentials/s3_checkpoint.secret"
    )

    # Data arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the LeRobot-format dataset",
    )
    parser.add_argument(
        "--data_split",
        type=str,
        default="test",
        choices=["train", "test", "full"],
        help="Data split to use for evaluation",
    )
    parser.add_argument(
        "--episode_ids",
        type=str,
        default=None,
        help="Comma-separated episode IDs to evaluate (e.g., '0,1,2'). Defaults to all episodes in the split.",
    )

    # Inference arguments
    parser.add_argument(
        "--guidance", type=float, default=7, help="Classifier-free guidance scale"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Output arguments
    parser.add_argument(
        "--save_root", type=str, default="results/dvrk_eval", help="Output directory"
    )
    parser.add_argument("--save_fps", type=int, default=10, help="FPS for saved videos")
    parser.add_argument(
        "--save_comparison",
        action="store_true",
        help="Save side-by-side comparison with ground truth",
    )

    # Context parallel arguments
    parser.add_argument(
        "--context_parallel_size",
        type=int,
        default=1,
        help="Context parallel size (number of GPUs)",
    )

    return parser.parse_args()


def build_episode_index_map(
    dataset: LeRobotDataset,
) -> dict[int, list[tuple[int, int]]]:
    """
    Build a mapping from episode_id to list of dataset indices for that episode.

    Args:
        dataset: The LeRobotDataset instance

    Returns:
        Dict mapping episode_id -> list of (dataset_idx, base_index) tuples sorted by base_index
    """
    # Access the underlying WrappedLeRobotSingleDataset
    # LeRobotDataset wraps one or more WrappedLeRobotSingleDataset
    inner_dataset = dataset.lerobot_datasets[0]
    all_steps = inner_dataset._all_steps  # List of (episode_id, base_index) tuples

    # Build mapping: episode_id -> [(dataset_idx, base_index), ...]
    episode_map: dict[int, list[tuple[int, int]]] = {}
    for dataset_idx, (episode_id, base_index) in enumerate(all_steps):
        if episode_id not in episode_map:
            episode_map[episode_id] = []
        episode_map[episode_id].append((dataset_idx, base_index))

    # Sort each episode's entries by base_index
    for episode_id in episode_map:
        episode_map[episode_id].sort(key=lambda x: x[1])

    return episode_map


def get_episode_ids_in_split(dataset: LeRobotDataset) -> list[int]:
    """
    Get the unique episode IDs present in the dataset (after split is applied).

    Args:
        dataset: The LeRobotDataset instance (already has split applied)

    Returns:
        Sorted list of episode IDs in the dataset
    """
    inner_dataset = dataset.lerobot_datasets[0]
    all_steps = inner_dataset._all_steps
    episode_ids = sorted(set(ep_id for ep_id, _ in all_steps))
    return episode_ids


def find_chunk_indices(
    episode_map: dict[int, list[tuple[int, int]]],
    episode_id: int,
    chunk_size: int = CHUNK_SIZE,
    timestep_interval: int = TIMESTEP_INTERVAL,
) -> list[int] | None:
    """
    Find dataset indices for non-overlapping chunks of an episode.

    For autoregressive inference, we need windows starting at base_index 0, 36, 72, ...
    (increments of chunk_size * timestep_interval = 12 * 3 = 36)

    Args:
        episode_map: Mapping from episode_id to list of (dataset_idx, base_index)
        episode_id: The episode to get chunks for
        chunk_size: Number of actions per chunk (default 12)
        timestep_interval: Temporal downsampling factor (default 3)

    Returns:
        List of dataset indices for non-overlapping chunks, or None if episode
        doesn't have base_index=0 (i.e., episode is only partially in the split)
    """
    if episode_id not in episode_map:
        return None

    entries = episode_map[
        episode_id
    ]  # [(dataset_idx, base_index), ...] sorted by base_index
    base_index_to_dataset_idx = {base_idx: ds_idx for ds_idx, base_idx in entries}

    # Must have base_index=0 to start autoregressive inference from beginning
    if 0 not in base_index_to_dataset_idx:
        return None

    # We need base_indices: 0, 36, 72, 108, ... (chunk_size * timestep_interval increments)
    stride = chunk_size * timestep_interval
    chunk_indices = []

    base_index = 0
    while base_index in base_index_to_dataset_idx:
        chunk_indices.append(base_index_to_dataset_idx[base_index])
        base_index += stride

    return chunk_indices


def main():
    torch.set_grad_enabled(False)
    args = parse_arguments()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load dataset with transforms - ensures actions are processed identically to training
    logger.info(
        f"Loading LeRobotDataset from {args.dataset_path} with split '{args.data_split}'"
    )

    dataset = LeRobotDataset(
        num_frames=NUM_FRAMES,
        time_division_factor=4,
        time_division_remainder=1,
        max_pixels=1920 * 1080,
        data_file_keys=("video",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
        repeat=1,
        args=None,
        dataset_path=args.dataset_path,
        data_split=args.data_split,
        embodiment="suturebot",
        downscaled_res=False,
    )

    # Build mapping from episode_id to dataset indices
    episode_map = build_episode_index_map(dataset)
    logger.info(
        f"Built index map for {len(episode_map)} episodes in '{args.data_split}' split"
    )

    # Determine which episodes to evaluate
    if args.episode_ids:
        episode_ids = [int(x) for x in args.episode_ids.split(",")]
    else:
        # Use all episodes in the split that have base_index=0 (complete episodes)
        episode_ids = get_episode_ids_in_split(dataset)

    logger.info(f"Requested episodes: {episode_ids}")

    # Initialize inference pipeline
    logger.info(f"Loading model from {args.ckpt_path}")
    video2world = ActionVideo2WorldInference(
        args.experiment,
        args.ckpt_path,
        args.s3_cred,
        context_parallel_size=args.context_parallel_size,
    )

    mem_bytes = torch.cuda.memory_allocated()
    logger.info(f"GPU memory after model load: {mem_bytes / (1024**3):.2f} GB")

    # Create output directories
    os.makedirs(args.save_root, exist_ok=True)
    os.makedirs(os.path.join(args.save_root, "predicted"), exist_ok=True)
    if args.save_comparison:
        os.makedirs(os.path.join(args.save_root, "comparison"), exist_ok=True)

    # -------------------------------------------------------------------------
    # Action log: records three stages of the action pipeline for each chunk so
    # the pipeline can be verified offline against the raw parquet data.
    #
    # Stage 1 — raw_parquet_actions_13x20:
    #     Absolute Cartesian setpoints as stored in the parquet file.
    #     Shape (13, 20): 13 frames sampled at TIMESTEP_INTERVAL=3,
    #     layout [psm1_xyz(3), psm1_rot6d(6), psm1_jaw(1),
    #             psm2_xyz(3), psm2_rot6d(6), psm2_jaw(1)].
    #
    # Stage 2 — normalized_rel_actions_12x20:
    #     Output of dataset[idx]["action"]: RelativeActionTransform converts the
    #     13-frame window to 12 per-chunk relative actions, then mean/std
    #     normalization is applied using stats.json.
    #     Shape (12, 20), dtype float32.
    #
    # Stage 3 — padded_actions_12x44:
    #     Stage 2 zero-padded to the model's unified 44D action space.
    #     Shape (12, 44). This is the tensor passed to step_inference().
    # -------------------------------------------------------------------------
    stats_path = os.path.join(args.dataset_path, "meta", "stats.json")
    with open(stats_path) as f:
        _stats = json.load(f)
    action_mean = np.array(_stats["action"]["mean"])
    action_std = np.array(_stats["action"]["std"])

    inner_dataset = dataset.lerobot_datasets[0]

    action_log: dict = {
        "dataset_path": args.dataset_path,
        "stats_path": stats_path,
        "action_mean_20d": action_mean.tolist(),
        "action_std_20d": action_std.tolist(),
        "num_frames": NUM_FRAMES,
        "timestep_interval": TIMESTEP_INTERVAL,
        "episodes": {},
    }

    # Process each episode
    for episode_id in episode_ids:
        logger.info(f"Processing episode {episode_id}")

        try:
            # Find dataset indices for non-overlapping chunks of this episode
            # Returns None if episode doesn't have base_index=0 (partial episode in split)
            chunk_indices = find_chunk_indices(episode_map, episode_id)

            if chunk_indices is None:
                logger.warning(
                    f"Episode {episode_id} doesn't start at base_index=0 in this split, skipping"
                )
                continue

            if len(chunk_indices) == 0:
                logger.warning(f"No chunks found for episode {episode_id}, skipping")
                continue

            logger.info(f"Episode {episode_id}: {len(chunk_indices)} chunks")

            # Load raw parquet for this episode once (used for action log Stage 1)
            parquet_files = sorted(
                Path(args.dataset_path).glob(
                    f"data/chunk-*/episode_{episode_id:06d}.parquet"
                )
            )
            df_episode = pd.read_parquet(parquet_files[0]) if parquet_files else None

            episode_action_log = []
            predicted_chunks = []
            gt_chunks = []  # For comparison
            current_frame = None

            for chunk_idx, dataset_idx in enumerate(chunk_indices):
                # Get data from dataset - actions are already transformed (relative + normalized)
                data = dataset[dataset_idx]

                # video shape: (C, T, H, W) -> need (T, H, W, C) for inference
                video = data["video"].permute(1, 2, 3, 0).numpy()  # (T, H, W, C)
                actions = data[
                    "action"
                ].numpy()  # (chunk_size, action_dim) - already normalized

                # Save Stage 2 actions before padding
                actions_20d = actions.copy()  # (12, 20) normalized relative actions

                # The Open-H pre-trained model uses a unified 44D action space.
                # SutureBot has 20D actions; zero-pad to 44D to match the model's
                # action_embedder input (same padding applied by MixedLeRobotDataset
                # during training).
                if actions.shape[-1] < 44:
                    pad = 44 - actions.shape[-1]
                    actions = np.pad(actions, ((0, 0), (0, pad)), mode="constant")

                # -----------------------------------------------------------------
                # Build action log entry for this chunk
                # -----------------------------------------------------------------
                _, base_index = inner_dataset._all_steps[dataset_idx]
                frame_indices = [
                    int(base_index) + t * TIMESTEP_INTERVAL for t in range(NUM_FRAMES)
                ]

                raw_parquet_actions = None
                if df_episode is not None and frame_indices[-1] < len(df_episode):
                    raw_parquet_actions = np.vstack(
                        [np.array(df_episode["action"].iloc[i]) for i in frame_indices]
                    )

                episode_action_log.append(
                    {
                        "chunk_idx": chunk_idx,
                        "dataset_idx": int(dataset_idx),
                        "base_index": int(base_index),
                        "frame_indices": frame_indices,
                        # Stage 1: absolute Cartesian setpoints from parquet (13 frames × 20D)
                        "raw_parquet_actions_13x20": raw_parquet_actions.tolist()
                        if raw_parquet_actions is not None
                        else None,
                        # Stage 2: relative + normalized actions from dataset pipeline (12 × 20D)
                        "normalized_rel_actions_12x20": actions_20d.tolist(),
                        # Stage 3: zero-padded to model's unified action space (12 × 44D)
                        "padded_actions_12x44": actions.tolist(),
                    }
                )
                # -----------------------------------------------------------------

                if chunk_idx == 0:
                    # First chunk: use ground truth first frame as conditioning
                    current_frame = video[0]  # (H, W, C)

                # Store ground truth for comparison
                gt_chunks.append(video)

                # Run inference
                next_frame, video_chunk = video2world.step_inference(
                    img_array=current_frame,
                    action=actions.astype(np.float32),
                    guidance=args.guidance,
                    seed=args.seed + chunk_idx,
                    num_latent_conditional_frames=1,
                )

                predicted_chunks.append(video_chunk)

                # Use last predicted frame as next conditioning
                current_frame = next_frame

                logger.info(f"  Chunk {chunk_idx + 1}/{len(chunk_indices)} complete")

            action_log["episodes"][str(episode_id)] = episode_action_log

            if not predicted_chunks:
                logger.warning(f"No chunks generated for episode {episode_id}")
                continue

            # Stitch chunks together
            # First chunk: all frames, subsequent chunks: skip first frame (it's the conditioning)
            stitched_predicted = [predicted_chunks[0]]
            for chunk in predicted_chunks[1:]:
                stitched_predicted.append(chunk[1:])
            predicted_video = np.concatenate(stitched_predicted, axis=0)

            # Save predicted video
            pred_path = os.path.join(
                args.save_root, "predicted", f"episode_{episode_id:04d}.mp4"
            )
            mediapy.write_video(pred_path, predicted_video, fps=args.save_fps)
            logger.info(f"Saved predicted video to {pred_path}")

            # Save side-by-side comparison if requested
            if args.save_comparison:
                # Stitch ground truth the same way
                stitched_gt = [gt_chunks[0]]
                for chunk in gt_chunks[1:]:
                    stitched_gt.append(chunk[1:])
                gt_video = np.concatenate(stitched_gt, axis=0)

                # Trim to same length
                min_len = min(len(gt_video), len(predicted_video))
                gt_video = gt_video[:min_len]
                predicted_video_trimmed = predicted_video[:min_len]

                # Concatenate side by side (GT on left, predicted on right)
                comparison = np.concatenate([gt_video, predicted_video_trimmed], axis=2)

                comp_path = os.path.join(
                    args.save_root, "comparison", f"episode_{episode_id:04d}.mp4"
                )
                mediapy.write_video(comp_path, comparison, fps=args.save_fps)
                logger.info(f"Saved comparison video to {comp_path}")

        except Exception as e:
            logger.error(f"Error processing episode {episode_id}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Save action log
    log_path = os.path.join(args.save_root, "action_log.json")
    with open(log_path, "w") as f:
        json.dump(action_log, f, indent=2)
    logger.info(f"Saved action log to {log_path}")

    # Cleanup
    video2world.cleanup()
    logger.info("Done!")


if __name__ == "__main__":
    main()
