#!/usr/bin/env python
"""
A script to convert DVRK (da Vinci Research Kit) robotics data into the LeRobot format (v2.1).

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
import shutil
import traceback
from pathlib import Path
from tqdm import tqdm
import tyro
import numpy as np
import os
import pandas as pd
import torch
from PIL import Image
import time
from scipy.spatial.transform import Rotation
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.constants import HF_LEROBOT_HOME

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

def compute_rel_actions_local(actions):
    """
    Computes relative actions for a dual-arm robot using SE(3) transformation.
    Both translation and rotation deltas are in the local (tooltip) frame.

    Follows UMI 'relative' mode: T_rel = T_base^(-1) @ T_action
    Reference: https://github.com/real-stanford/universal_manipulation_interface

    actions[0] is used as the base pose, actions[1:] are the targets.

    Input per-arm: [xyz (3), 6D_rotation (6), gripper (1)] = 10
    Dual-arm input: [n_actions, arm1 (10) + arm2 (10)] = [n_actions, 20]
    Output per-arm: [delta_xyz (3), delta_rot6d (6), gripper (1)] = 10
    Dual-arm output: [n_actions-1, arm1 (10) + arm2 (10)] = [n_actions-1, 20]
    """
    if isinstance(actions, torch.Tensor):
        actions = actions.numpy()

    base = actions[0]
    targets = actions[1:]
    n_targets = targets.shape[0]
    rel_actions = np.zeros((n_targets, 20))

    for arm in range(2):
        i = arm * 10  # Same stride for input and output

        # Build 4x4 base pose matrix
        T_base = np.eye(4)
        T_base[:3, :3] = rotation_6d_to_matrix(base[i + 3 : i + 9])
        T_base[:3, 3] = base[i : i + 3]

        # Build 4x4 target pose matrices
        T_targets = np.zeros((n_targets, 4, 4))
        T_targets[:, :3, :3] = rotation_6d_to_matrix(targets[:, i + 3 : i + 9])
        T_targets[:, :3, 3] = targets[:, i : i + 3]
        T_targets[:, 3, 3] = 1.0

        # SE(3) relative: T_rel = T_base^(-1) @ T_target
        T_base_inv = np.linalg.inv(T_base)
        T_rel = T_base_inv @ T_targets

        # Extract components
        rel_actions[:, i : i + 3] = T_rel[:, :3, 3]  # Local translation delta
        R_rel = T_rel[:, :3, :3]  # Relative rotation matrix
        rel_actions[:, i + 3 : i + 9] = R_rel[:, :2, :].reshape(n_targets, 6)  # 6D rotation (first 2 rows)
        rel_actions[:, i + 9] = targets[:, i + 9]  # Gripper (absolute)

    return rel_actions

def read_images(image_dir: str, file_pattern: str) -> np.ndarray:
    """Reads images from a directory into a NumPy array."""
    images = []
    ## count images in the dir
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


def generate_modality_metadata(features: dict, embodiment: str, description: str = None) -> dict:
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
        "description": description or "Auto-generated modality metadata derived from dataset features.",
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
            description="DVRK surgical robot dataset with dual-arm PSM control"
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


def _compute_and_write_stats(dataset_path: Path):
    """
    Compute normalization statistics for relative actions and states,
    then write them to stats.json in the dataset's meta directory.
    """
    parquet_files = sorted(dataset_path.glob("data/*/*.parquet"))

    if not parquet_files:
        print(f"Warning: No parquet files found in {dataset_path / 'data'}, skipping stats computation")
        return

    print(f"Computing stats from {len(parquet_files)} parquet files...")

    all_rel_actions = []
    all_states = []

    for pf in tqdm(parquet_files, desc="Computing stats"):
        df = pd.read_parquet(pf)

        # Extract actions and compute relative actions
        actions = np.vstack(df["action"].values)  # [T, 20]
        rel_actions = compute_rel_actions_local(actions)  # [T-1, 20]
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


def _discover_episodes(data_path: Path):
    episodes = []
    tissue_dirs = sorted(
        d
        for d in data_path.iterdir()
        if d.is_dir() and d.name.startswith("tissue_")
    )
    for tissue_dir in tissue_dirs:
        for subtask_name in sorted(os.listdir(tissue_dir)):
            subtask_dir = os.path.join(tissue_dir, subtask_name)
            if not os.path.isdir(subtask_dir):
                continue

            subtask_prompt = " ".join(subtask_name.split("_")[1:])
            is_recovery = subtask_prompt.endswith("recovery")
            if is_recovery:
                subtask_prompt = subtask_prompt[:-9]

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
    final_output_path = os.path.join(HF_LEROBOT_HOME, repo_id)
    print(f"Output path: {final_output_path}")
    print(f"(To change output location, set HF_LEROBOT_HOME env var before running)")
    
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
        video_backend="torchcodec",  # Uses AV1 codec
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

    # Compute and write normalization stats for relative actions and states
    _compute_and_write_stats(Path(final_output_path))

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


if __name__ == "__main__":
    tyro.cli(main)
