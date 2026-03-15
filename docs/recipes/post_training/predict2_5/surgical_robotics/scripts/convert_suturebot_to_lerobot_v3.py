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
A script to convert DVRK (da Vinci Research Kit) robotics data into the LeRobot v3 format.

This script processes DVRK surgical robot datasets organized in directory structures
with CSV kinematics data and camera views. It extracts dual-arm PSM states, 6D-rotation
actions, and a single endoscope view into a LeRobotDataset for the Hugging Face Hub.

Expected DVRK Dataset Structure:
--------------------------------
The script expects a directory structure organized by tissue and subtasks:

/path/to/dataset/
├── tissue_10/                          # Tissue phantom number
│   ├── 1_suture_throw/                 # Subtask directory
│   │   ├── episode_001/                # Individual episode
│   │   │   ├── left_img_dir/           # Left endoscope images
│   │   │   │   └── frame000000_left.jpg
│   │   │   ├── right_img_dir/          # Right endoscope images
│   │   │   │   └── frame000000_right.jpg
│   │   │   ├── endo_psm1/              # PSM1 wrist camera
│   │   │   │   └── frame000000_psm1.jpg
│   │   │   ├── endo_psm2/              # PSM2 wrist camera
│   │   │   │   └── frame000000_psm2.jpg
│   │   │   └── ee_csv.csv              # Kinematics data (16D state + actions)
│   │   └── episode_002/
│   └── 2_needle_pass_recovery/         # Recovery demonstrations
└── tissue_11/

Data Format:
------------
- **Actions**: 20D dual-PSM Cartesian poses + jaw positions (6D rotation representation)
- **States**: 16D dual-PSM current poses + jaw positions
- **Images**: 1 camera view (left endoscope, stored as observation.images.main)
- **Metadata**: Tool types, instruction text

Usage:
------
    # Default output to HF_LEROBOT_HOME:
    python convert_suturebot_to_lerobot_v3.py --data-path /path/to/dataset --repo-id dataset-name

    # Custom output location (set env var before running):
    export HF_LEROBOT_HOME=/custom/output
    python convert_suturebot_to_lerobot_v3.py --data-path /path/to/dataset --repo-id dataset-name

Dependencies:
-------------
- lerobot == 0.3.3
- torchcodec
- tyro
- pandas
- PIL
- numpy
- scipy
"""

import json
import os
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tyro
from PIL import Image
from tqdm import tqdm

try:
    from lerobot.constants import HF_LEROBOT_HOME
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ModuleNotFoundError as e:
    try:
        import lerobot

        ver = getattr(lerobot, "__version__", "unknown")
    except Exception:
        ver = "not installed"
    raise SystemExit(
        f"This script requires lerobot==0.3.3 (package layout changed in 0.4.x). "
        f"You have lerobot {ver}. Install with: pip install lerobot==0.3.3"
    ) from e

states_name = [
    "psm1_pose.position.x",
    "psm1_pose.position.y",
    "psm1_pose.position.z",
    "psm1_pose.orientation.x",
    "psm1_pose.orientation.y",
    "psm1_pose.orientation.z",
    "psm1_pose.orientation.w",
    "psm1_jaw",
    "psm2_pose.position.x",
    "psm2_pose.position.y",
    "psm2_pose.position.z",
    "psm2_pose.orientation.x",
    "psm2_pose.orientation.y",
    "psm2_pose.orientation.z",
    "psm2_pose.orientation.w",
    "psm2_jaw",
]
actions_name = [
    "psm1_sp.position.x",
    "psm1_sp.position.y",
    "psm1_sp.position.z",
    "psm1_sp.orientation.x",
    "psm1_sp.orientation.y",
    "psm1_sp.orientation.z",
    "psm1_sp.orientation.w",
    "psm1_jaw_sp",
    "psm2_sp.position.x",
    "psm2_sp.position.y",
    "psm2_sp.position.z",
    "psm2_sp.orientation.x",
    "psm2_sp.orientation.y",
    "psm2_sp.orientation.z",
    "psm2_sp.orientation.w",
    "psm2_jaw_sp",
]

ACTION_NAMES_6D = [
    "psm1_pos_x",
    "psm1_pos_y",
    "psm1_pos_z",
    "psm1_rot_r11",
    "psm1_rot_r12",
    "psm1_rot_r13",
    "psm1_rot_r21",
    "psm1_rot_r22",
    "psm1_rot_r23",
    "psm1_jaw",
    "psm2_pos_x",
    "psm2_pos_y",
    "psm2_pos_z",
    "psm2_rot_r11",
    "psm2_rot_r12",
    "psm2_rot_r13",
    "psm2_rot_r21",
    "psm2_rot_r22",
    "psm2_rot_r23",
    "psm2_jaw",
]


def quat_to_6d_rotation(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [qx, qy, qz, qw] to 6D rotation representation.
    Returns first two rows of rotation matrix flattened (row-major order).
    """
    qx, qy, qz, qw = quat
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n < 1e-12:
        return np.zeros(6, dtype=np.float32)
    s = 2.0 / n
    x, y, z = qx * s, qy * s, qz * s
    wx, wy, wz = qw * x, qw * y, qw * z
    xx, xy, xz = qx * x, qx * y, qx * z
    yy, yz, zz = qy * y, qy * z, qz * z

    r11 = 1.0 - (yy + zz)
    r12 = xy - wz
    r13 = xz + wy
    r21 = xy + wz
    r22 = 1.0 - (xx + zz)
    r23 = yz - wx

    return np.array([r11, r12, r13, r21, r22, r23], dtype=np.float32)


def rotation_6d_to_matrix(rot6d):
    """
    Convert 6D rotation representation to rotation matrix.
    6D rotation is the first two ROWS of a rotation matrix (row-major format),
    orthonormalized via Gram-Schmidt.

    This matches the incoming dVRK/SutureBot data format:
        [r11, r12, r13, r21, r22, r23] = [row1, row2]

    Args:
        rot6d: Array of shape (..., 6) containing [row1 (3), row2 (3)]

    Returns:
        Rotation matrices of shape (..., 3, 3)
    """
    shape = rot6d.shape[:-1]
    rot6d = rot6d.reshape(*shape, 2, 3)

    # First row (normalized)
    row1 = rot6d[..., 0, :]
    row1 = row1 / (np.linalg.norm(row1, axis=-1, keepdims=True) + 1e-8)

    # Second row (orthogonalized and normalized)
    row2 = rot6d[..., 1, :]
    row2 = row2 - np.sum(row1 * row2, axis=-1, keepdims=True) * row1
    row2 = row2 / (np.linalg.norm(row2, axis=-1, keepdims=True) + 1e-8)

    # Third row (cross product)
    row3 = np.cross(row1, row2)

    # Stack into rotation matrix (as rows)
    R = np.stack([row1, row2, row3], axis=-2)
    return R


def compute_rel_actions(actions):
    """
    Computes relative actions for a dual-arm robot.
    Global translation delta, local (tooltip frame) rotation delta in 6D format.

    Reference: https://github.com/real-stanford/universal_manipulation_interface

    actions[0] is used as the base pose, actions[1:] are the targets.

    Input per-arm: [xyz (3), 6D_rotation (6), gripper (1)] = 10
    Dual-arm input: [n_actions, arm1 (10) + arm2 (10)] = [n_actions, 20]
    Output per-arm: [delta_xyz (3), delta_rot6d (6), gripper (1)] = 10
    Dual-arm output: [n_actions-1, arm1 (10) + arm2 (10)] = [n_actions-1, 20]

    The relative rotation R_rel = R_base.T @ R_target is represented in 6D format
    (first two rows of the rotation matrix, flattened).
    """
    if isinstance(actions, torch.Tensor):
        actions = actions.numpy()

    base = actions[0]
    targets = actions[1:]
    n_targets = targets.shape[0]
    rel_actions = np.zeros((n_targets, 20))

    for arm in range(2):
        i = arm * 10  # Both input and output use same stride
        R_base = rotation_6d_to_matrix(base[i + 3 : i + 9])
        R_tgt = rotation_6d_to_matrix(targets[:, i + 3 : i + 9])

        # Global translation delta
        rel_actions[:, i : i + 3] = targets[:, i : i + 3] - base[i : i + 3]
        # Relative rotation in 6D format (first 2 rows of R_rel, flattened)
        R_rel = R_base.T @ R_tgt  # [n_targets, 3, 3]
        rel_actions[:, i + 3 : i + 9] = R_rel[:, :2, :].reshape(n_targets, 6)
        # Gripper (absolute value, not delta)
        rel_actions[:, i + 9] = targets[:, i + 9]

    return rel_actions


def read_images(image_dir: str, file_pattern: str) -> np.ndarray:
    """Reads images from a directory into a NumPy array."""
    images = []
    # count images in the dir
    num_images = len(
        [
            name
            for name in os.listdir(image_dir)
            if os.path.isfile(os.path.join(image_dir, name))
        ]
    )
    for idx in range(num_images):
        filename = os.path.join(image_dir, file_pattern.format(idx))
        if not os.path.exists(filename):
            print(f"Warning: {filename} does not exist.")
            continue
        img = Image.open(filename)
        img_array = np.array(img)[..., :3]  # Ensure 3 channels
        images.append(img_array)
    if images:
        return np.stack(images)
    else:
        return np.empty((0, 0, 0, 3), dtype=np.uint8)


def process_episode(dataset, episode_path, states_name, actions_name, subtask_prompt):
    """Processes a single episode, save the data to lerobot format"""

    # Paths to image directories
    left_dir = os.path.join(episode_path, "left_img_dir")
    csv_file = os.path.join(episode_path, "ee_csv.csv")

    # Read CSV to determine the number of frames (excluding header)
    df = pd.read_csv(csv_file)

    # Read images from each camera
    left_images = read_images(left_dir, "frame{:06d}_left.jpg")
    num_frames = min(len(df), left_images.shape[0])

    # Read kinematics data and convert to structured array with headers
    kinematics_data = np.array(
        [tuple(row) for row in df.to_numpy()],
        dtype=[(col, df[col].dtype.str) for col in df.columns],
    )
    # print(kinematics_data[0])

    for i in range(num_frames):
        frame = {
            "observation.state": np.hstack(
                [kinematics_data[n][i] for n in states_name]
            ).astype(np.float32),
            "action": _build_action_6d(kinematics_data, i, actions_name),
            "instruction.text": subtask_prompt,
            "observation.meta.tool.psm1": "Large Needle Driver",
            "observation.meta.tool.psm2": "Debakey Forceps",
        }

        if left_images.size > 0:
            frame["observation.images.main"] = left_images[i]
        # Use synthetic timestamps (frame_idx / fps) to match video PTS values.
        timestamp_sec = i / 30.0  # fps = 30
        dataset.add_frame(frame, task=subtask_prompt, timestamp=timestamp_sec)

    return dataset


def _build_action_6d(kinematics_data, index, actions_name):
    psm1_action = np.array([kinematics_data[n][index] for n in actions_name[:8]])
    psm2_action = np.array([kinematics_data[n][index] for n in actions_name[8:]])

    psm1_pos = psm1_action[:3]
    psm1_quat = psm1_action[3:7]
    psm1_jaw = psm1_action[7]
    psm2_pos = psm2_action[:3]
    psm2_quat = psm2_action[3:7]
    psm2_jaw = psm2_action[7]

    psm1_rot6d = quat_to_6d_rotation(psm1_quat)
    psm2_rot6d = quat_to_6d_rotation(psm2_quat)

    action = np.concatenate(
        [
            psm1_pos,
            psm1_rot6d,
            [psm1_jaw],
            psm2_pos,
            psm2_rot6d,
            [psm2_jaw],
        ]
    ).astype(np.float32)
    return action


def _compute_stats(data: np.ndarray) -> dict:
    """Compute statistics for a numpy array."""
    return {
        "mean": np.mean(data, axis=0).tolist(),
        "std": np.std(data, axis=0).tolist(),
        "min": np.min(data, axis=0).tolist(),
        "max": np.max(data, axis=0).tolist(),
        "q01": np.quantile(data, 0.01, axis=0).tolist(),
        "q99": np.quantile(data, 0.99, axis=0).tolist(),
    }


def _vector_length(shape: list) -> int:
    """Extract vector length from shape, handling tuple/list formats."""
    if not shape:
        raise ValueError("Shape is empty")
    # Handle both list and tuple shapes
    shape_list = list(shape)
    if len(shape_list) > 1:
        raise ValueError(f"Expected 1D vector, got shape {shape}")
    return int(shape_list[0])


def _derive_state_entries(features: dict) -> dict:
    """Derive state entries from features dict."""
    state_entries = {}
    for key, meta in features.items():
        if not key.startswith("observation.state"):
            continue
        # Skip non-vector features (like images)
        if meta.get("dtype") == "video":
            continue
        try:
            length = _vector_length(meta["shape"])
            state_entries[key] = {
                "start": 0,
                "end": length,
                "rotation_type": None,
                "absolute": True,
                "dtype": meta.get("dtype", "float32"),
                "range": None,
                "original_key": key,
            }
        except ValueError:
            continue
    return state_entries


def _derive_action_entries(features: dict) -> dict:
    """Derive action entries from features dict."""
    action_entries = {}
    for key, meta in features.items():
        if key != "action" and not key.startswith("action."):
            continue
        try:
            length = _vector_length(meta["shape"])
            action_entries[key] = {
                "start": 0,
                "end": length,
                "rotation_type": None,
                "absolute": False,
                "dtype": meta.get("dtype", "float32"),
                "range": None,
                "original_key": key,
            }
        except ValueError:
            continue
    return action_entries


def _derive_video_entries(features: dict) -> dict:
    """Derive video entries from features dict."""
    video_entries = {}
    for key, meta in features.items():
        if not key.startswith("observation.images"):
            continue
        video_entries[key] = {
            "original_key": key,
        }
    return video_entries


def _derive_annotation_entries(features: dict) -> dict | None:
    """Derive annotation entries from features dict."""
    annotation_entries = {}
    for key in features:
        if key.startswith("annotation.") or key.startswith("language."):
            annotation_entries[key] = {
                "original_key": key,
            }
    return annotation_entries or None


def generate_modality_metadata(
    features: dict, embodiment: str, description: str = None
) -> dict:
    """
    Generate modality metadata from features dict.

    Args:
        features: Dict of feature definitions (from LeRobotDataset or info.json)
        embodiment: Robot type identifier
        description: Optional description for the dataset

    Returns:
        Dict containing modality metadata for state, action, video, and annotation entries
    """
    state_entries = _derive_state_entries(features)
    action_entries = _derive_action_entries(features)
    video_entries = _derive_video_entries(features)
    annotation_entries = _derive_annotation_entries(features)

    return {
        "state": state_entries,
        "action": action_entries,
        "video": video_entries,
        "annotation": annotation_entries,
        "embodiment": embodiment,
        "description": description
        or "Auto-generated modality metadata derived from dataset features.",
        "version": "v2.0",
    }


def _write_modality_metadata(dataset_path: Path, features: dict, robot_type: str):
    """
    Generate and write modality.json to the dataset's meta directory.

    Args:
        dataset_path: Path to the dataset root directory
        features: Dict of feature definitions
        robot_type: Robot type identifier (embodiment)
    """
    meta_dir = dataset_path / "meta"
    if not meta_dir.exists():
        meta_dir.mkdir(parents=True, exist_ok=True)

    try:
        modality_metadata = generate_modality_metadata(
            features,
            embodiment=robot_type,
            description="DVRK surgical robot dataset with dual-arm PSM control",
        )
        modality_path = meta_dir / "modality.json"
        with modality_path.open("w", encoding="utf-8") as f:
            json.dump(modality_metadata, f, indent=2)
            f.write("\n")
        print(
            f"Generated modality.json with "
            f"{len(modality_metadata.get('state', {}))} state entries and "
            f"{len(modality_metadata.get('action', {}))} action entries."
        )
    except Exception as err:
        print(f"WARNING: Failed to generate modality metadata: {err}")


def _compute_and_write_stats(
    dataset_path: Path,
    num_frames: int = 13,
    timestep_interval: int = 3,
    chunk_stride: int = 1,
):
    """
    Compute normalization statistics for relative actions and states,
    then write them to stats.json in the dataset's meta directory.

    IMPORTANT: Statistics are computed on PER-CHUNK relative actions to match
    the training pipeline. During training:
    1. Each sample is a chunk of `num_frames` frames sampled at `timestep_interval`
    2. RelativeActionTransform converts these to (num_frames-1) relative actions
    3. Each relative action is relative to the FIRST frame of its chunk

    This is different from computing relative actions over the entire episode,
    which would produce a very different distribution (larger variance, drifting mean).

    NOTE: chunk_stride=1 (default) samples ALL possible chunks like training does,
    giving representative statistics. This may take longer but ensures the stats
    match the actual training distribution.

    Args:
        dataset_path: Path to the dataset root directory
        num_frames: Number of frames per chunk (must match training, default=13 for dvrk)
        timestep_interval: Frame sampling interval (must match training, default=3 for dvrk)
        chunk_stride: Stride between chunk start positions. Default=1 for maximum
                      coverage (matching training sampling). Use larger values for faster
                      but potentially less accurate stats computation.
    """
    parquet_files = sorted(dataset_path.glob("data/*/*.parquet"))

    if not parquet_files:
        print(
            f"Warning: No parquet files found in {dataset_path / 'data'}, skipping stats computation"
        )
        return

    # Compute delta_indices matching training pipeline (groot_configs.py)
    # For dvrk: delta_indices = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]
    delta_indices = list(range(0, num_frames * timestep_interval, timestep_interval))
    max_delta = delta_indices[-1]  # Last frame offset needed

    print(f"Computing per-chunk stats from {len(parquet_files)} parquet files...")
    print(f"  num_frames={num_frames}, timestep_interval={timestep_interval}")
    print(f"  delta_indices={delta_indices}")
    print(f"  chunk_stride={chunk_stride}")

    all_rel_actions = []
    all_states = []
    total_chunks = 0
    skipped_short_episodes = 0

    for pf in tqdm(parquet_files, desc="Computing per-chunk stats"):
        df = pd.read_parquet(pf)

        # Extract all actions for this episode
        actions = np.vstack(df["action"].values)  # [T, 20]
        episode_len = len(actions)

        # Skip episodes too short for even one chunk
        if episode_len <= max_delta:
            skipped_short_episodes += 1
            continue

        # Iterate through chunks, sampling frames at delta_indices
        # This matches how training samples data
        for base_idx in range(0, episode_len - max_delta, chunk_stride):
            # Sample frames at delta_indices (like training does)
            chunk_indices = [base_idx + d for d in delta_indices]
            chunk_actions = actions[chunk_indices]  # [num_frames, 20]

            # Compute relative actions for this chunk
            # Output: [num_frames-1, 20] actions relative to chunk_actions[0]
            chunk_rel_actions = compute_rel_actions(chunk_actions)
            all_rel_actions.append(chunk_rel_actions)
            total_chunks += 1

        # Extract states (these don't need chunking - just collect all)
        if "observation.state" in df.columns:
            states = np.vstack(df["observation.state"].values)  # [T, 16]
            all_states.append(states)

    if not all_rel_actions:
        print("Error: No valid chunks found. All episodes may be too short.")
        print(f"  Minimum episode length needed: {max_delta + 1} frames")
        return

    # Stack all per-chunk relative actions
    all_rel_actions = np.vstack(all_rel_actions)
    print(f"Total chunks processed: {total_chunks}")
    print(f"Total per-chunk relative actions: {all_rel_actions.shape}")
    if skipped_short_episodes > 0:
        print(f"Skipped {skipped_short_episodes} episodes (too short for chunking)")

    # Compute stats on per-chunk relative actions
    stats = {
        "action": _compute_stats(all_rel_actions),
    }

    if all_states:
        all_states = np.vstack(all_states)
        print(f"Total states: {all_states.shape}")
        stats["observation.state"] = _compute_stats(all_states)

    # Save stats
    output_path = dataset_path / "meta" / "stats.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved stats to {output_path}")

    # Print summary of action stats for verification
    action_stats = stats["action"]
    print("\nAction statistics summary (should be used for normalization):")
    print(
        f"  Mean range: [{min(action_stats['mean']):.6f}, {max(action_stats['mean']):.6f}]"
    )
    print(
        f"  Std range:  [{min(action_stats['std']):.6f}, {max(action_stats['std']):.6f}]"
    )


def _discover_episodes(data_path: Path):
    episodes = []
    tissue_dirs = sorted(
        d for d in data_path.iterdir() if d.is_dir() and d.name.startswith("tissue_")
    )
    for tissue_dir in tissue_dirs:
        for subtask_name in sorted(os.listdir(tissue_dir)):
            subtask_dir = os.path.join(tissue_dir, subtask_name)
            if not os.path.isdir(subtask_dir):
                continue

            subtask_prompt = " ".join(subtask_name.split("_")[1:])
            is_recovery = subtask_prompt.endswith("recovery")
            if is_recovery:
                subtask_prompt = subtask_prompt[:-9].strip()

            for episode_name in sorted(os.listdir(subtask_dir)):
                episode_dir = os.path.join(subtask_dir, episode_name)
                if not os.path.isdir(episode_dir):
                    continue
                episodes.append((episode_dir, subtask_prompt))
    return episodes


def convert_data_to_lerobot(
    data_path: Path, repo_id: str, *, push_to_hub: bool = False
):
    """
    Converts a single Zarr store with episode boundaries to a LeRobotDataset.

    Args:
        data_path: The path to the source dataset directory.
        repo_id: The repository ID for the dataset on the Hugging Face Hub.
        push_to_hub: Whether to push the dataset to the Hub after conversion.

    Note:
        Output location is determined by HF_LEROBOT_HOME environment variable.
        Set it before running Python to customize the output path:
            export HF_LEROBOT_HOME=/your/custom/path
    """
    # Use h264 for broad compatibility (default libsvtav1 is AV1 which can cause
    # codec issues in some environments and is not needed for training).
    import functools

    import lerobot.datasets.lerobot_dataset as _lerobot_ds_mod
    import lerobot.datasets.video_utils as _lerobot_vid_mod

    _lerobot_vid_mod.encode_video_frames = functools.partial(
        _lerobot_vid_mod.encode_video_frames, vcodec="h264"
    )
    _lerobot_ds_mod.encode_video_frames = functools.partial(
        _lerobot_ds_mod.encode_video_frames, vcodec="h264"
    )

    final_output_path = os.path.join(HF_LEROBOT_HOME, repo_id)
    print(f"Output path: {final_output_path}")
    print("(To change output location, set HF_LEROBOT_HOME env var before running)")

    if os.path.exists(final_output_path):
        print(f"Removing existing dataset at {final_output_path}")
        shutil.rmtree(final_output_path)

    # Initialize a LeRobotDataset with the desired features.
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        use_videos=True,
        robot_type="dvrk",
        fps=30,
        features={
            "observation.images.main": {
                "dtype": "video",
                "shape": (540, 960, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(states_name),),
                "names": states_name,
            },
            "action": {
                "dtype": "float32",
                "shape": (len(ACTION_NAMES_6D),),
                "names": ACTION_NAMES_6D,
            },
            "observation.meta.tool.psm1": {
                "dtype": "string",
                "shape": (1,),
                "names": ["value"],
            },
            "observation.meta.tool.psm2": {
                "dtype": "string",
                "shape": (1,),
                "names": ["value"],
            },
            "instruction.text": {
                "dtype": "string",
                "shape": (1,),
                "description": "Natural language command for the robot",
            },
        },
        image_writer_processes=16,
        image_writer_threads=20,
        tolerance_s=1.0,
        batch_encoding_size=12,
        video_backend="torchcodec",  # Codec overridden to h264 above for compatibility
    )
    # measure time taken to complete the process
    start_time = time.time()
    episodes = _discover_episodes(data_path)
    if not episodes:
        print("Warning: No episodes found.")
        return

    for episode_dir, subtask_prompt in tqdm(episodes, desc="Processing episodes"):
        try:
            dataset = process_episode(
                dataset, episode_dir, states_name, actions_name, subtask_prompt
            )
            dataset.save_episode()
        except Exception as e:
            print(f"Error processing episode {episode_dir}: {e}")
            traceback.print_exc()
            dataset.clear_episode_buffer()
    print(f"Total episodes processed: {len(episodes)}")

    # Encode any remaining videos that didn't fill a complete batch.
    # With batch_encoding_size=12, episodes are encoded in groups of 12.
    # Any trailing episodes (count < 12) are left un-encoded and must be
    # flushed here, otherwise the videos/ directory won't be created.
    if dataset.episodes_since_last_encoding > 0:
        start_ep = dataset.num_episodes - dataset.episodes_since_last_encoding
        print(
            f"Encoding remaining {dataset.episodes_since_last_encoding} episodes "
            f"({start_ep}–{dataset.num_episodes - 1})..."
        )
        dataset.batch_encode_videos(start_ep, dataset.num_episodes)

    # Compute and write normalization stats for PER-CHUNK relative actions.
    # IMPORTANT: Stats must be computed on per-chunk deltas (not whole-episode deltas)
    # to match the training pipeline in groot_configs.py where RelativeActionTransform
    # is applied to each chunk independently.
    # Parameters: num_frames=13, timestep_interval=3 match dvrk training config.
    # chunk_stride=1 (default) samples ALL possible chunks like training does.
    _compute_and_write_stats(
        Path(final_output_path),
        num_frames=13,  # Must match training (groot_configs.py)
        timestep_interval=3,  # Must match training (dvrk timestep_interval)
        # chunk_stride=1 is default - samples all chunks like training
    )

    # Generate and write modality.json
    dataset_features = {
        "observation.images.main": {
            "dtype": "video",
            "shape": (540, 960, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(states_name),),
            "names": states_name,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(ACTION_NAMES_6D),),
            "names": ACTION_NAMES_6D,
        },
        "instruction.text": {
            "dtype": "string",
            "shape": (1,),
        },
    }
    _write_modality_metadata(Path(final_output_path), dataset_features, "dvrk")

    print(f"suturing processed successful, time taken: {time.time() - start_time}")


def main(
    data_path: Path = Path("/path/to/dataset"),
    repo_id: str = "suturebot_lerobot",
    *,
    push_to_hub: bool = False,
):
    """
    Main entry point for the conversion script.

    Args:
        data_path: The path to the source dataset.
        repo_id: The dataset name (subdirectory name for the output).
        push_to_hub: If True, uploads the dataset to the Hub after conversion.

    Note:
        To customize output location, set HF_LEROBOT_HOME before running:
            export HF_LEROBOT_HOME=/your/custom/path
            python convert_suturebot_to_lerobot_v3.py --data-path ...
    """
    if not data_path.exists():
        print(f"Error: The provided path does not exist: {data_path}")
        print("Please provide a valid path to your data.")
        return

    convert_data_to_lerobot(data_path, repo_id, push_to_hub=push_to_hub)


def recompute_stats(
    dataset_path: Path = Path("/path/to/lerobot/dataset"),
    num_frames: int = 13,
    timestep_interval: int = 3,
    chunk_stride: int = 1,
):
    """
    Standalone function to recompute stats.json for an existing LeRobot dataset.

    Use this when you need to fix statistics without re-running the full conversion.

    Args:
        dataset_path: Path to the LeRobot dataset (containing data/, meta/, videos/)
        num_frames: Number of frames per chunk (default=13 for dvrk)
        timestep_interval: Frame sampling interval (default=3 for dvrk)
        chunk_stride: Stride between chunks (default=1 for full coverage like training)

    Example:
        python convert_suturebot_to_lerobot_v3.py recompute-stats --dataset-path /SutureBot
    """
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        return

    if not (dataset_path / "data").exists():
        print(f"Error: No 'data' directory found in {dataset_path}")
        print("Expected LeRobot format with data/, meta/, videos/ directories.")
        return

    print(f"Recomputing stats for dataset: {dataset_path}")
    _compute_and_write_stats(
        dataset_path,
        num_frames=num_frames,
        timestep_interval=timestep_interval,
        chunk_stride=chunk_stride,
    )
    print("Done!")


if __name__ == "__main__":
    # Use tyro.extras.subcommand_cli to support both main conversion and stats recomputation
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "recompute-stats":
        # Remove 'recompute-stats' from argv so tyro can parse the rest
        sys.argv.pop(1)
        tyro.cli(recompute_stats)
    else:
        tyro.cli(main)
