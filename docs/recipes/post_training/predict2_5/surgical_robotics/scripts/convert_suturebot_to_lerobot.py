#!/usr/bin/env python
"""
SutureBot dataset to LeRobot v2.1 format conversion script.

This script converts the SutureBot dataset (folder structure with CSVs and images)
to the LeRobot v2.1 format with:
- MP4 video compression for images (30-50x smaller)
- Parquet files for structured data (actions, states, metadata)
- Single parquet file per episode for cleaner organization

Dataset structure expected:
    input_path/
        tissue_X/
            phase_folder/  (e.g., 1_needle_pickup, 2_needle_throw, etc.)
                demo_folder/  (e.g., 20250117-120348-398226)
                    left_img_dir/
                    right_img_dir/
                    endo_psm1/
                    endo_psm2/
                    ee_csv.csv
"""

import shutil
import sys
import os
import multiprocessing
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Iterable
import json
import time
import random
from datetime import datetime

import tyro
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import cv2
from scipy.spatial.transform import Rotation as R


def _resize_with_padding(images: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """
    Resize a batch of images to target dimensions while preserving aspect ratio using padding.

    Args:
        images: Input images with shape (B, H, W, C) where B is batch size
        target_width: Desired width
        target_height: Desired height

    Returns:
        Resized and padded images with shape (B, target_height, target_width, C)
    """
    batch_size = images.shape[0]
    padded_images = np.zeros((batch_size, target_height, target_width, 3), dtype=np.uint8)

    for i in range(batch_size):
        h, w = images[i].shape[:2]
        target_aspect = target_width / target_height
        aspect = w / h

        if aspect > target_aspect:
            new_w = target_width
            new_h = int(new_w / aspect)
        else:
            new_h = target_height
            new_w = int(new_h * aspect)

        resized = cv2.resize(images[i], (new_w, new_h))

        pad_x = (target_width - new_w) // 2
        pad_y = (target_height - new_h) // 2

        padded_images[i, pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    return padded_images


def _collect_parquet_paths(dataset_root: Path) -> list[Path]:
    matches = sorted(dataset_root.glob("data/*/*.parquet"))
    if not matches:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")
    return matches


def _load_info(dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing {info_path}")
    with info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _vector_names(feature_meta: dict[str, Any]) -> list[str]:
    names = feature_meta.get("names")
    if not isinstance(names, list) or not names:
        raise ValueError(f"Expected `names` list in feature metadata, got {names}")
    return [str(name) for name in names]


def _stack_columns(
    frames: Iterable[pd.DataFrame],
    column_order: list[str],
) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for df in frames:
        if len(column_order) == 1:
            col_data = df[column_order[0]]
            if isinstance(col_data.iloc[0], (list, np.ndarray)):
                subset = np.stack(col_data.values).astype(np.float32)
            else:
                subset = df[column_order].to_numpy(dtype=np.float32, copy=False)
        else:
            subset = df[column_order].to_numpy(dtype=np.float32, copy=False)
        arrays.append(subset)
    return np.concatenate(arrays, axis=0)


def _compute_stats(array: np.ndarray) -> dict[str, list[float]]:
    return {
        "mean": np.mean(array, axis=0).tolist(),
        "std": np.std(array, axis=0, ddof=0).tolist(),
        "min": np.min(array, axis=0).tolist(),
        "max": np.max(array, axis=0).tolist(),
        "q01": np.quantile(array, 0.01, axis=0).tolist(),
        "q99": np.quantile(array, 0.99, axis=0).tolist(),
    }


def generate_stats(dataset_root: Path) -> dict[str, Any]:
    info = _load_info(dataset_root)
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("info.json is missing a `features` object.")

    parquet_paths = _collect_parquet_paths(dataset_root)
    parquet_frames = (
        pd.read_parquet(path)
        for path in tqdm(parquet_paths, desc="Loading parquet files")
    )

    cached_frames = [frame for frame in parquet_frames]

    stats: dict[str, Any] = {}

    if "observation.state" in features:
        state_columns = ["observation.state"]
        state_array = _stack_columns(cached_frames, state_columns)
        stats["observation.state"] = _compute_stats(state_array)

    if "action" in features:
        action_columns = ["action"]
        action_array = _stack_columns(cached_frames, action_columns)
        stats["action"] = _compute_stats(action_array)

    if not stats:
        raise ValueError("No state/action features detected; nothing to write.")

    return stats


def _vector_length(shape: list[int]) -> int:
    if not shape:
        raise ValueError("Shape is empty")
    if len(shape) > 1:
        raise ValueError(f"Expected 1D vector, got shape {shape}")
    return int(shape[0])


def _derive_state_entries(features: dict[str, Any]) -> dict[str, Any]:
    state_entries: dict[str, Any] = {}
    for key, meta in features.items():
        if not key.startswith("observation.state"):
            continue
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
    return state_entries


def _derive_action_entries(features: dict[str, Any]) -> dict[str, Any]:
    action_entries: dict[str, Any] = {}
    for key, meta in features.items():
        if key != "action" and not key.startswith("action."):
            continue
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
    return action_entries


def _derive_video_entries(features: dict[str, Any]) -> dict[str, Any]:
    video_entries: dict[str, Any] = {}
    for key, meta in features.items():
        if not key.startswith("observation.images"):
            continue
        video_entries[key] = {
            "original_key": key,
        }
    if not video_entries:
        raise ValueError(
            "No video features discovered. Expected keys like 'observation.images.main'."
        )
    return video_entries


def _derive_annotation_entries(features: dict[str, Any]) -> dict[str, Any] | None:
    annotation_entries: dict[str, Any] = {}
    for key in features:
        if key.startswith("annotation.") or key.startswith("language."):
            annotation_entries[key] = {
                "original_key": key,
            }
    return annotation_entries or None


def generate_modality_metadata(
    info: dict[str, Any], embodiment: str
) -> dict[str, Any]:
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("info.json is missing a `features` object.")

    state_entries = _derive_state_entries(features)
    if not state_entries:
        raise ValueError(
            "No state features detected. Expected keys starting with 'observation.state'."
        )
    action_entries = _derive_action_entries(features)
    if not action_entries:
        raise ValueError("No action features detected. Expected key 'action'.")
    video_entries = _derive_video_entries(features)
    annotation_entries = _derive_annotation_entries(features)

    return {
        "state": state_entries,
        "action": action_entries,
        "video": video_entries,
        "annotation": annotation_entries,
        "embodiment": embodiment,
        "description": info.get(
            "description",
            "Auto-generated modality metadata derived from info.json.",
        ),
        "version": info.get("codebase_version", "auto"),
    }


def save_video_mp4(frames: np.ndarray, output_path: Path, fps: int = 30, quality: int = 10):
    """
    Save frames as an MP4 video file using imageio and ffmpeg.
    """
    try:
        import imageio
        if frames.dtype != np.uint8:
            frames = frames.astype(np.uint8)
        
        writer = imageio.get_writer(
            output_path,
            fps=fps,
            quality=quality,
            codec='libx264',
            pixelformat='yuv420p',
            macro_block_size=1
        )
        
        for frame in frames:
            writer.append_data(frame)
        
        writer.close()
        return True
        
    except ImportError:
        print("Warning: imageio not installed. Falling back to OpenCV.")
        return save_video_opencv(frames, output_path, fps)


def save_video_opencv(frames: np.ndarray, output_path: Path, fps: int = 30):
    """Fallback video saving using OpenCV."""
    try:
        if frames.dtype != np.uint8:
            frames = frames.astype(np.uint8)
        
        height, width = frames.shape[1:3]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        return True
        
    except Exception as e:
        print(f"Error saving video with OpenCV: {e}")
        return False


def quat_to_6d_rotation(action: np.ndarray) -> np.ndarray:
    """
    Convert quaternion action to 6D rotation representation.
    
    Args:
        action: Array of shape (N, 8) representing [x, y, z, qx, qy, qz, qw, jaw]

    Returns:
        Array of shape (N, 10) representing [x, y, z, r11, r12, r13, r21, r22, r23, jaw]
    """
    quat_actions = action[:, 3:7]  # Shape: (N, 4) - [qx, qy, qz, qw]
    
    r_actions = R.from_quat(quat_actions)
    rot_matrices = r_actions.as_matrix()  # Shape: (N, 3, 3)
    # Extract first two columns of rotation matrix
    diff_6d = rot_matrices[:, :, :2]  # Shape: (N, 3, 2)
    diff_6d = diff_6d.transpose(0, 2, 1).reshape(-1, 6)  # Shape: (N, 6)
    
    # Build output array
    result = np.zeros((action.shape[0], 10), dtype=np.float32)
    result[:, 0:3] = action[:, 0:3]   # Position (x, y, z)
    result[:, 3:9] = diff_6d          # 6D rotation
    result[:, 9] = action[:, 7]       # Jaw
    
    return result


def save_episode_parquet_v21(
    parquet_path: Path,
    states: np.ndarray,
    actions: np.ndarray,
    timestamps: np.ndarray,
    instruction: str,
    episode_metadata: Dict[str, Any]
) -> bool:
    """
    Save episode data as a single Parquet file with all non-video data (v2.1 format).
    """
    try:
        num_timesteps = len(timestamps)
        
        data_dict = {
            'timestep': np.arange(num_timesteps),
            'timestamp': timestamps,
        }
        
        data_dict["observation.state"] = list(states)
        data_dict["action"] = list(actions)
        data_dict["episode_index"] = episode_metadata.get('episode_index', -1)

        df = pd.DataFrame(data_dict)
        table = pa.Table.from_pandas(df)
        
        metadata = {
            'instruction': instruction,
            'episode_index': str(episode_metadata.get('episode_index', -1)),
            'num_frames': str(num_timesteps),
            'fps': str(episode_metadata.get('fps', 30)),
            'video_format': episode_metadata.get('video_format', 'mp4'),
            'video_quality': str(episode_metadata.get('video_quality', 10)),
            'mono': str(episode_metadata.get('mono', False)),
            'episode_path': episode_metadata.get('episode_path', ''),
            'conversion_timestamp': datetime.now().isoformat()
        }
        
        metadata_bytes = {}
        for k, v in metadata.items():
            if v is None:
                v = ''
            metadata_bytes[k.encode()] = str(v).encode()
        
        table = table.replace_schema_metadata(metadata_bytes)
        
        pq.write_table(
            table,
            parquet_path,
            compression='snappy',
            use_dictionary=True,
            compression_level=None
        )
        
        return True
        
    except Exception as e:
        print(f"Error saving parquet file: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_episode_parquet(parquet_path: Path) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
    """Load episode data from a Parquet file."""
    try:
        table = pq.read_table(parquet_path)
        metadata = {}
        if table.schema.metadata:
            for key, value in table.schema.metadata.items():
                metadata[key.decode()] = value.decode()
        df = table.to_pandas()
        return df, metadata
    except Exception as e:
        print(f"Error loading parquet file: {e}")
        return None, None


def discover_episodes(input_path: Path) -> List[Tuple[str, str, str]]:
    """
    Discover all episodes in the SutureBot dataset structure.
    
    Returns:
        List of tuples: (episode_path, instruction, tissue_id)
    """
    episodes = []
    
    # Find all tissue folders
    tissue_folders = sorted([
        d for d in input_path.iterdir() 
        if d.is_dir() and d.name.startswith("tissue_")
    ])
    
    for tissue_folder in tissue_folders:
        tissue_id = tissue_folder.name
        
        # Find all phase folders within the tissue
        phase_folders = sorted([
            d for d in tissue_folder.iterdir() 
            if d.is_dir()
        ])
        
        for phase_folder in phase_folders:
            phase_name = phase_folder.name
            
            # Skip non-phase directories
            if phase_name in ["Corrections"]:
                continue
            
            # Extract instruction from phase name (remove leading number and underscore)
            parts = phase_name.split("_", 1)
            if len(parts) > 1:
                instruction = parts[1].replace("_", " ")
            else:
                instruction = phase_name
            
            # Find all demo folders within the phase
            demo_folders = sorted([
                d for d in phase_folder.iterdir() 
                if d.is_dir() and d.name != "Corrections"
            ])
            
            for demo_folder in demo_folders:
                # Check if this is a valid demo folder (has ee_csv.csv)
                csv_path = demo_folder / "ee_csv.csv"
                if csv_path.exists():
                    episodes.append((str(demo_folder), instruction, tissue_id))
    
    return episodes


# Header names for SutureBot CSV files
HEADER_NAMES = {
    'qpos_psm1': [
        "psm1_pose.position.x", "psm1_pose.position.y", "psm1_pose.position.z",
        "psm1_pose.orientation.x", "psm1_pose.orientation.y", 
        "psm1_pose.orientation.z", "psm1_pose.orientation.w",
        "psm1_jaw"
    ],
    'qpos_psm2': [
        "psm2_pose.position.x", "psm2_pose.position.y", "psm2_pose.position.z",
        "psm2_pose.orientation.x", "psm2_pose.orientation.y", 
        "psm2_pose.orientation.z", "psm2_pose.orientation.w",
        "psm2_jaw"
    ],
    'actions_psm1': [
        "psm1_sp.position.x", "psm1_sp.position.y", "psm1_sp.position.z",
        "psm1_sp.orientation.x", "psm1_sp.orientation.y", 
        "psm1_sp.orientation.z", "psm1_sp.orientation.w",
        "psm1_jaw_sp"
    ],
    'actions_psm2': [
        "psm2_sp.position.x", "psm2_sp.position.y", "psm2_sp.position.z",
        "psm2_sp.orientation.x", "psm2_sp.orientation.y", 
        "psm2_sp.orientation.z", "psm2_sp.orientation.w",
        "psm2_jaw_sp"
    ]
}

# Camera configurations
CAMERA_CONFIGS = {
    'left': {'subdir': 'left_img_dir', 'suffix': '_left.jpg'},
    'right': {'subdir': 'right_img_dir', 'suffix': '_right.jpg'},
    'wrist_left': {'subdir': 'endo_psm2', 'suffix': '_psm2.jpg'},
    'wrist_right': {'subdir': 'endo_psm1', 'suffix': '_psm1.jpg'},
}


def process_episode_worker_v21(args: Tuple) -> Optional[Dict[str, Any]]:
    """
    Worker function to process a single episode with LeRobot v2.1 format.
    """
    (episode_info, episode_idx, image_size, fps, output_dir, 
     mono, video_quality, chunk_size) = args
    
    try:
        import traceback
        
        output_dir = Path(output_dir)
        episode_path, instruction, tissue_id = episode_info
        episode_path = Path(episode_path)
        episode_name = episode_path.name
        
        debug_mode = os.environ.get('DEBUG_WORKERS', '0') == '1'
        worker_id = os.getpid()
        
        if debug_mode:
            print(f"[Worker PID={worker_id}] Processing {episode_name}", flush=True)
        
        start_time = time.time()
        
        # Load CSV data
        csv_path = episode_path / "ee_csv.csv"
        if not csv_path.exists():
            return {
                'status': 'skipped',
                'episode_path': str(episode_path),
                'reason': f'Missing CSV: {csv_path}'
            }
        
        csv_data = pd.read_csv(csv_path)
        episode_len = len(csv_data)
        
        if episode_len == 0:
            return {
                'status': 'skipped',
                'episode_path': str(episode_path),
                'reason': 'Empty episode'
            }
        
        # Check for required columns
        required_cols = (
            HEADER_NAMES['qpos_psm1'] + HEADER_NAMES['qpos_psm2'] +
            HEADER_NAMES['actions_psm1'] + HEADER_NAMES['actions_psm2']
        )
        missing_cols = [col for col in required_cols if col not in csv_data.columns]
        if missing_cols:
            return {
                'status': 'skipped',
                'episode_path': str(episode_path),
                'reason': f'Missing columns: {missing_cols[:5]}...'
            }
        
        # Check for image directories
        left_img_dir = episode_path / 'left_img_dir'
        if not left_img_dir.exists():
            return {
                'status': 'skipped',
                'episode_path': str(episode_path),
                'reason': 'Missing left_img_dir'
            }
        
        # Count available frames
        left_images = sorted(left_img_dir.glob("frame*_left.jpg"))
        num_image_frames = len(left_images)
        
        if num_image_frames == 0:
            return {
                'status': 'skipped',
                'episode_path': str(episode_path),
                'reason': 'No images found'
            }
        
        max_frame_idx = min(episode_len, num_image_frames)
        
        # Calculate chunk directory
        chunk_idx = episode_idx // chunk_size
        chunk_dir = output_dir / "data" / f"chunk-{chunk_idx:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        
        parquet_path = chunk_dir / f"episode_{episode_idx:06d}.parquet"
        
        # Check if already processed
        if parquet_path.exists():
            if debug_mode:
                print(f"[Worker {worker_id}] Skipping already processed: {episode_name}", flush=True)
            return {
                'status': 'skipped',
                'episode_path': str(episode_path),
                'reason': 'Already processed'
            }
        
        # Create video directories
        video_chunk_dir = output_dir / "videos" / f"chunk-{chunk_idx:03d}"
        video_main_dir = video_chunk_dir / "observation.images.main"
        video_main_dir.mkdir(parents=True, exist_ok=True)
        
        if not mono:
            video_left_dir = video_chunk_dir / "observation.images.wrist_left"
            video_right_dir = video_chunk_dir / "observation.images.wrist_right"
            video_left_dir.mkdir(parents=True, exist_ok=True)
            video_right_dir.mkdir(parents=True, exist_ok=True)
        else:
            video_left_dir = None
            video_right_dir = None
        
        # Process frames
        all_images = []
        all_states = []
        all_actions = []
        all_timestamps = []
        all_wrist_left = [] if not mono else None
        all_wrist_right = [] if not mono else None
        
        for frame_idx in range(max_frame_idx):
            # Load images
            left_img_path = episode_path / 'left_img_dir' / f"frame{frame_idx:06d}_left.jpg"
            if not left_img_path.exists():
                continue
            
            left_img = cv2.imread(str(left_img_path))
            if left_img is None:
                continue
            
            # Convert BGR to RGB
            left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
            
            # Resize if needed
            if left_img.shape[:2] != tuple(image_size):
                left_img = _resize_with_padding(
                    left_img[np.newaxis, ...], 
                    image_size[1], image_size[0]
                )[0]
            
            all_images.append(left_img)
            
            # Load wrist cameras if not in mono mode
            if not mono:
                wrist_left_path = episode_path / 'endo_psm2' / f"frame{frame_idx:06d}_psm2.jpg"
                wrist_right_path = episode_path / 'endo_psm1' / f"frame{frame_idx:06d}_psm1.jpg"
                
                wrist_left_img = cv2.imread(str(wrist_left_path))
                wrist_right_img = cv2.imread(str(wrist_right_path))
                
                if wrist_left_img is not None:
                    wrist_left_img = cv2.cvtColor(wrist_left_img, cv2.COLOR_BGR2RGB)
                    if wrist_left_img.shape[:2] != tuple(image_size):
                        wrist_left_img = _resize_with_padding(
                            wrist_left_img[np.newaxis, ...], 
                            image_size[1], image_size[0]
                        )[0]
                else:
                    wrist_left_img = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
                
                if wrist_right_img is not None:
                    wrist_right_img = cv2.cvtColor(wrist_right_img, cv2.COLOR_BGR2RGB)
                    if wrist_right_img.shape[:2] != tuple(image_size):
                        wrist_right_img = _resize_with_padding(
                            wrist_right_img[np.newaxis, ...], 
                            image_size[1], image_size[0]
                        )[0]
                else:
                    wrist_right_img = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
                
                all_wrist_left.append(wrist_left_img)
                all_wrist_right.append(wrist_right_img)
            
            # Get current robot state (qpos)
            qpos_psm1 = csv_data[HEADER_NAMES['qpos_psm1']].iloc[frame_idx].to_numpy()
            qpos_psm2 = csv_data[HEADER_NAMES['qpos_psm2']].iloc[frame_idx].to_numpy()
            state = np.concatenate([qpos_psm1, qpos_psm2])  # 16-dim
            all_states.append(state)
            
            # Get action setpoints and convert to 6D rotation representation
            action_psm1 = csv_data[HEADER_NAMES['actions_psm1']].iloc[frame_idx].to_numpy()
            action_psm2 = csv_data[HEADER_NAMES['actions_psm2']].iloc[frame_idx].to_numpy()
            
            # Convert quaternion actions to 6D rotation representation
            action_psm1_6d = quat_to_6d_rotation(action_psm1[np.newaxis, :])[0]  # 10-dim
            action_psm2_6d = quat_to_6d_rotation(action_psm2[np.newaxis, :])[0]  # 10-dim
            
            action = np.concatenate([action_psm1_6d, action_psm2_6d])  # 20-dim
            all_actions.append(action)
            
            all_timestamps.append(frame_idx / fps)
        
        num_frames_saved = len(all_images)
        if num_frames_saved == 0:
            return {
                'status': 'skipped',
                'episode_path': str(episode_path),
                'reason': 'No valid frames'
            }
        
        # Save videos
        video_path = video_main_dir / f"episode_{episode_idx:06d}.mp4"
        save_video_mp4(np.array(all_images), video_path, fps=fps, quality=video_quality)
        
        if not mono and all_wrist_left and all_wrist_right:
            save_video_mp4(
                np.array(all_wrist_left), 
                video_left_dir / f"episode_{episode_idx:06d}.mp4", 
                fps=fps, quality=video_quality
            )
            save_video_mp4(
                np.array(all_wrist_right), 
                video_right_dir / f"episode_{episode_idx:06d}.mp4", 
                fps=fps, quality=video_quality
            )
        
        # Prepare metadata
        episode_metadata = {
            'episode_path': str(episode_path),
            'instruction': instruction,
            'num_frames': num_frames_saved,
            'episode_index': episode_idx,
            'video_format': 'mp4',
            'fps': fps,
            'video_quality': video_quality,
            'mono': mono,
            'tissue_id': tissue_id
        }
        
        # Save parquet
        success = save_episode_parquet_v21(
            parquet_path,
            np.array(all_states, dtype=np.float32),
            np.array(all_actions, dtype=np.float32),
            np.array(all_timestamps, dtype=np.float32),
            instruction,
            episode_metadata
        )
        
        if not success:
            return {
                'status': 'error',
                'episode_path': str(episode_path),
                'reason': 'Failed to save parquet'
            }
        
        # Calculate sizes
        video_size_mb = video_path.stat().st_size / (1024 * 1024) if video_path.exists() else 0
        parquet_size_mb = parquet_path.stat().st_size / (1024 * 1024) if parquet_path.exists() else 0
        total_size_mb = parquet_size_mb + video_size_mb
        
        if not mono and video_left_dir and video_right_dir:
            left_video_path = video_left_dir / f"episode_{episode_idx:06d}.mp4"
            right_video_path = video_right_dir / f"episode_{episode_idx:06d}.mp4"
            if left_video_path.exists():
                total_size_mb += left_video_path.stat().st_size / (1024 * 1024)
            if right_video_path.exists():
                total_size_mb += right_video_path.stat().st_size / (1024 * 1024)
        
        if debug_mode:
            elapsed = time.time() - start_time
            print(f"[Worker {worker_id}] Completed {episode_name} in {elapsed:.2f}s "
                  f"(video: {video_size_mb:.1f}MB, parquet: {parquet_size_mb:.2f}MB)", 
                  flush=True)
        
        return {
            'status': 'success',
            'episode_path': str(episode_path),
            'instruction': instruction,
            'num_frames': num_frames_saved,
            'episode_index': episode_idx,
            'chunk_idx': chunk_idx,
            'parquet_path': str(parquet_path),
            'video_size_mb': video_size_mb,
            'parquet_size_mb': parquet_size_mb,
            'total_size_mb': total_size_mb
        }
        
    except Exception as e:
        error_traceback = traceback.format_exc()
        return {
            'status': 'error',
            'episode_path': str(episode_path) if 'episode_path' in dir() else 'unknown',
            'reason': str(e),
            'traceback': error_traceback
        }


def create_metadata_files_v21(
    output_path: Path,
    successful_episodes: List[Dict],
    all_episodes: List,
    fps: int,
    chunk_size: int
):
    """Create episodes.jsonl, tasks.jsonl, and episodes_stats.jsonl for v2.1 format"""
    meta_dir = output_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    episodes_file = meta_dir / "episodes.jsonl"
    tasks_file = meta_dir / "tasks.jsonl"
    stats_file = meta_dir / "episodes_stats.jsonl"
    
    tasks_map = {}
    task_id_counter = 0
    
    with open(episodes_file, 'w') as ep_f, open(stats_file, 'w') as stats_f:
        for ep_result in successful_episodes:
            if ep_result.get('status') != 'success':
                continue
            
            episode_idx = ep_result['episode_index']
            instruction = ep_result.get('instruction', 'unknown task')
            num_frames = ep_result['num_frames']
            chunk_idx = ep_result.get('chunk_idx', episode_idx // chunk_size)
            
            if instruction not in tasks_map:
                tasks_map[instruction] = task_id_counter
                task_id_counter += 1
            task_id = tasks_map[instruction]
            
            episode_data = {
                "episode_index": episode_idx,
                "task_index": task_id,
                "task": instruction,
                "chunk": chunk_idx,
                "length": num_frames
            }
            ep_f.write(json.dumps(episode_data) + '\n')
            
            stats_data = {
                "episode_index": episode_idx,
                "num_frames": num_frames,
                "fps": fps,
                "duration": num_frames / fps if fps > 0 else 0,
                "video_size_mb": ep_result.get('video_size_mb', 0),
                "parquet_size_mb": ep_result.get('parquet_size_mb', 0),
                "total_size_mb": ep_result.get('total_size_mb', 0)
            }
            stats_f.write(json.dumps(stats_data) + '\n')
    
    with open(tasks_file, 'w') as f:
        for task_name, task_id in sorted(tasks_map.items(), key=lambda x: x[1]):
            task_data = {
                "task_index": task_id,
                "task": task_name
            }
            f.write(json.dumps(task_data) + '\n')
    
    print(f"Created metadata files:")
    print(f"  - {episodes_file.relative_to(output_path)}")
    print(f"  - {tasks_file.relative_to(output_path)}")
    print(f"  - {stats_file.relative_to(output_path)}")
    print(f"  Tasks: {len(tasks_map)}")
    return tasks_map


def create_info_json_v21(
    output_path: Path,
    robot_type: str,
    fps: int,
    image_size: tuple,
    mono: bool,
    *,
    chunk_size: int,
    total_episode_count: int,
    test_episode_count: int,
    data_extension: str = "parquet",
    video_extension: str = "mp4",
) -> Dict:
    """Create info.json for LeRobot v2.1 format"""
    info = {
        "codebase_version": "v2.0",
        "robot_type": robot_type,
        "fps": fps,
        "video": True,
        "chunks_size": chunk_size,
        "data_path": f"data/chunk-{{episode_chunk:03d}}/episode_{{episode_index:06d}}.{data_extension}",
        "video_path": (
            f"videos/chunk-{{episode_chunk:03d}}/{{video_key}}/episode_{{episode_index:06d}}.{video_extension}"
        ),
        "encoding": {
            "vcodec": "libx264",
            "pix_fmt": "yuv420p",
            "g": fps,
            "crf": 10
        },
        "splits": {
            "train": f"{test_episode_count}:{total_episode_count}",
            "test": f"0:{test_episode_count}"
        },
        "features": {}
    }
    
    # State: 16-dim (8 per arm: 3 pos + 4 quat + 1 jaw)
    # Action: 20-dim (10 per arm: 3 pos + 6 rot (6D repr) + 1 jaw)
    
    if mono:
        info["features"] = {
            "observation.images.main": {
                "dtype": "video",
                "shape": list(image_size) + [3],
                "names": ["height", "width", "channel"],
                "video_info": {"video.fps": fps}
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [16]
            },
            "action": {
                "dtype": "float32",
                "shape": [20]
            }
        }
    else:
        info["features"] = {
            "observation.images.main": {
                "dtype": "video",
                "shape": list(image_size) + [3],
                "names": ["height", "width", "channel"],
                "video_info": {"video.fps": fps}
            },
            "observation.images.wrist_left": {
                "dtype": "video",
                "shape": list(image_size) + [3],
                "names": ["height", "width", "channel"],
                "video_info": {"video.fps": fps}
            },
            "observation.images.wrist_right": {
                "dtype": "video",
                "shape": list(image_size) + [3],
                "names": ["height", "width", "channel"],
                "video_info": {"video.fps": fps}
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [16],
                "names": [
                    "psm1_pos_x", "psm1_pos_y", "psm1_pos_z",
                    "psm1_quat_x", "psm1_quat_y", "psm1_quat_z", "psm1_quat_w",
                    "psm1_jaw",
                    "psm2_pos_x", "psm2_pos_y", "psm2_pos_z",
                    "psm2_quat_x", "psm2_quat_y", "psm2_quat_z", "psm2_quat_w",
                    "psm2_jaw"
                ]
            },
            "action": {
                "dtype": "float32",
                "shape": [20],
                "names": [
                    "psm1_pos_x", "psm1_pos_y", "psm1_pos_z",
                    "psm1_rot_r11", "psm1_rot_r12", "psm1_rot_r13",
                    "psm1_rot_r21", "psm1_rot_r22", "psm1_rot_r23",
                    "psm1_jaw",
                    "psm2_pos_x", "psm2_pos_y", "psm2_pos_z",
                    "psm2_rot_r11", "psm2_rot_r12", "psm2_rot_r13",
                    "psm2_rot_r21", "psm2_rot_r22", "psm2_rot_r23",
                    "psm2_jaw"
                ]
            }
        }
    
    info["features"]["timestamp"] = {
        "dtype": "float32",
        "shape": [1]
    }
    
    info["features"]["next.reward"] = {
        "dtype": "float32",
        "shape": [1]
    }
    
    info["features"]["next.done"] = {
        "dtype": "bool",
        "shape": [1]
    }

    info["features"]["index"] = {
        "dtype": "int64",
        "shape": [1]
    }

    info["features"]["episode_index"] = {
        "dtype": "int64",
        "shape": [1]
    }

    info["features"]["task_index"] = {
        "dtype": "int64",
        "shape": [1]
    }
    
    meta_dir = output_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    with open(meta_dir / "info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    return info


def convert_suturebot_to_lerobot_v21(
    input_path: Path, 
    output_path: Path,
    *,
    robot_type: str = "dvrk",
    fps: int = 30,
    image_size: tuple = (540, 960),
    num_workers: int = None,
    mono: bool = True,
    video_quality: int = 10,
    force_overwrite: bool = False,
    resume: bool = True,
    chunk_size: int = 1000
):
    """
    Converts SutureBot dataset to LeRobot v2.1 format with chunk structure.
    """
    
    if num_workers is None:
        available_cpus = multiprocessing.cpu_count()
        num_workers = min(64, max(1, available_cpus // 3))
    
    print(f"SutureBot to LeRobot v2.1 Format Conversion")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Using {num_workers} parallel workers")
    print(f"Chunk size: {chunk_size} episodes per chunk")
    print(f"Storage format: MP4 video + Parquet files")
    print(f"Video quality: {video_quality}/10")
    print(f"Image size: {image_size}")
    print(f"Mono mode: {mono}")
    
    # Handle directory creation/clearing
    if not resume and output_path.exists():
        contents = list(output_path.iterdir())
        if contents:
            print(f"Found {len(contents)} items in {output_path}")
            if not force_overwrite:
                response = input(f"Clear contents? (y/N): ")
                if response.lower() != 'y':
                    print("Aborting.")
                    sys.exit(1)
            
            print(f"Clearing contents...")
            for item in contents:
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                except Exception as e:
                    print(f"  Warning: Could not remove {item}: {e}")
    elif resume and output_path.exists():
        print(f"Resume mode enabled")
    else:
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Discover episodes
    print(f"\nDiscovering episodes...")
    all_episodes = discover_episodes(input_path)
    print(f"Found {len(all_episodes)} episodes")
    
    if len(all_episodes) == 0:
        print("No episodes found. Check input path and dataset structure.")
        return

    # shuffle episodes
    random.seed(42)
    random.shuffle(all_episodes)

    # Get number of unique instructions
    unique_instructions = set(episode[1] for episode in all_episodes)
    print(f"Found {len(unique_instructions)} unique instructions")
    
    test_episode = {instruction: [] for instruction in unique_instructions}
    total_episode_count = len(all_episodes)

    # Create train and test episodes, with 3 test episodes per instruction
    train_episodes = []
    for episode in all_episodes:
        instruction = episode[1]
        if len(test_episode[instruction]) < 3:
            test_episode[instruction].append(episode)
        else:
            train_episodes.append(episode)

    # Flatten test_episode
    test_episodes = [ep for episodes in test_episode.values() for ep in episodes]
    test_episode_count = len(test_episodes)
    print(f"Using {len(train_episodes)} train episodes")
    print(f"Using {test_episode_count} test episodes")

    # Re-merge episodes, we'll assign splits via info.json
    all_episodes = test_episodes + train_episodes
    
    # Debug mode
    max_episodes = os.environ.get('MAX_EPISODES')
    if max_episodes:
        try:
            max_episodes = int(max_episodes)
            if max_episodes > 0:
                all_episodes = all_episodes[:max_episodes]
                print(f"DEBUG: Processing only {max_episodes} episodes")
        except ValueError:
            pass
    
    # Create info.json
    info_json = create_info_json_v21(
        output_path,
        robot_type,
        fps,
        image_size,
        mono,
        chunk_size=chunk_size,
        total_episode_count=total_episode_count,
        test_episode_count=test_episode_count,
        data_extension="parquet",
        video_extension="mp4",
    )
    print(f"Created meta/info.json with v2.0 format")
    
    # Check for existing episodes if resume enabled
    complete_episodes = {}
    data_dir = output_path / "data"
    if resume and data_dir.exists():
        chunk_dirs = sorted([
            d for d in data_dir.iterdir() 
            if d.is_dir() and d.name.startswith("chunk-")
        ])
        if chunk_dirs:
            print(f"Found {len(chunk_dirs)} existing chunks, validating...")
            
            for chunk_dir in chunk_dirs:
                parquet_files = list(chunk_dir.glob("episode_*.parquet"))
                for parquet_file in parquet_files:
                    try:
                        ep_idx = int(parquet_file.stem.split('_')[1])
                        _, metadata = load_episode_parquet(parquet_file)
                        if metadata and 'episode_path' in metadata:
                            complete_episodes[metadata['episode_path']] = ep_idx
                    except Exception as e:
                        print(f"  Error checking {parquet_file.name}: {e}")
            
            print(f"  Found {len(complete_episodes)} complete episodes")
    
    # Prepare worker arguments
    worker_args = []
    for idx, episode_info in enumerate(all_episodes):
        episode_path = episode_info[0]
        
        if resume and episode_path in complete_episodes:
            continue
        
        worker_args.append((
            episode_info, idx, image_size, fps, str(output_path), 
            mono, video_quality, chunk_size
        ))
    
    if len(worker_args) == 0:
        print("\nAll episodes already complete!")
        return
    
    # Start timing
    conversion_start_time = time.time()
    
    # Process episodes
    print(f"\nProcessing {len(worker_args)} episodes...")
    
    episode_count = 0
    successful_episodes = []
    error_episodes = []
    total_video_size_mb = 0
    total_parquet_size_mb = 0
    total_size_mb = 0
    
    try:
        with multiprocessing.Pool(processes=num_workers, maxtasksperchild=5) as pool:
            chunksize = max(1, len(worker_args) // (num_workers * 4))
            
            with tqdm(total=len(worker_args), desc="Processing") as pbar:
                result_iterator = pool.imap_unordered(
                    process_episode_worker_v21, worker_args, chunksize=chunksize
                )
                
                for result in result_iterator:
                    if result is None:
                        pbar.update(1)
                        continue
                    
                    status = result.get('status', 'unknown')
                    
                    if status == 'success':
                        episode_count += 1
                        successful_episodes.append(result)
                        
                        video_size = result.get('video_size_mb', 0)
                        parquet_size = result.get('parquet_size_mb', 0)
                        total_ep_size = result.get('total_size_mb', 0)
                        
                        total_video_size_mb += video_size
                        total_parquet_size_mb += parquet_size
                        total_size_mb += total_ep_size
                        
                        if episode_count > 0:
                            avg_total = total_size_mb / episode_count
                            pbar.set_postfix({
                                'saved': episode_count,
                                'avg_mb': f"{avg_total:.1f}"
                            })
                    elif status == 'error':
                        error_episodes.append(result)
                    
                    pbar.update(1)
                    del result
                    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        pool.terminate()
        pool.join()
        raise
    
    # Calculate final statistics
    total_conversion_time = time.time() - conversion_start_time
    
    # Print summary
    print("\n" + "="*60)
    print(f"Conversion complete!")
    print(f"Successfully converted: {episode_count} episodes")
    print(f"Total episodes: {len(all_episodes)}")
    print(f"Output: {output_path}")
    
    print(f"\nTiming:")
    print(f"  Total time: {int(total_conversion_time//3600):02d}:{int((total_conversion_time%3600)//60):02d}:{int(total_conversion_time%60):02d}")
    if episode_count > 0:
        print(f"  Per episode: {total_conversion_time/episode_count:.2f}s")
    
    if episode_count > 0:
        print(f"\nStorage:")
        print(f"  Video total: {total_video_size_mb:.1f} MB")
        print(f"  Parquet total: {total_parquet_size_mb:.1f} MB")
        print(f"  Dataset total: {total_size_mb:.1f} MB")
        print(f"  Per episode: {total_size_mb/episode_count:.1f} MB")
    
    if error_episodes:
        print(f"\nErrors: {len(error_episodes)} episodes failed")
        for err in error_episodes[:5]:
            print(f"  - {err.get('episode_path', 'unknown')}: {err.get('reason', 'unknown')}")
    
    # Create v2.1 metadata files
    if successful_episodes:
        create_metadata_files_v21(
            output_path, 
            successful_episodes, 
            all_episodes, 
            fps, 
            chunk_size
        )

        # Save additional dataset metadata
        metadata_path = output_path / "dataset_metadata.json"
        
        dataset_metadata = {
            'format_version': '2.0',
            'storage_format': 'parquet+mp4',
            'conversion_timestamp': datetime.now().isoformat(),
            'conversion_time_seconds': total_conversion_time,
            'total_episodes': episode_count,
            'total_frames': sum(ep['num_frames'] for ep in successful_episodes),
            'fps': fps,
            'image_size': list(image_size),
            'mono': mono,
            'video_format': 'mp4',
            'video_quality': video_quality,
            'data_format': 'parquet',
            'compression': 'snappy',
            'num_workers': num_workers,
            'storage_stats': {
                'total_video_size_mb': total_video_size_mb,
                'total_parquet_size_mb': total_parquet_size_mb,
                'total_size_mb': total_size_mb,
                'avg_video_mb': total_video_size_mb / episode_count if episode_count > 0 else 0,
                'avg_parquet_mb': total_parquet_size_mb / episode_count if episode_count > 0 else 0,
                'avg_total_mb': total_size_mb / episode_count if episode_count > 0 else 0
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(dataset_metadata, f, indent=2)
        
        print(f"\nMetadata saved to: {metadata_path}")

    meta_dir = output_path / "meta"
    if meta_dir.exists():
        try:
            detailed_modality = generate_modality_metadata(info_json, embodiment=robot_type)
            modality_path = meta_dir / "modality.json"
            with modality_path.open("w", encoding="utf-8") as f:
                json.dump(detailed_modality, f, indent=2)
                f.write("\n")
            print(
                f"Rebuilt modality.json with "
                f"{len(detailed_modality.get('state', {}))} state entries and "
                f"{len(detailed_modality.get('action', {}))} action entries."
            )
        except Exception as err:
            print(f"WARNING: Failed to regenerate modality metadata: {err}")

        try:
            stats = generate_stats(output_path)
            stats_path = meta_dir / "stats.json"
            with stats_path.open("w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
                f.write("\n")
            print(f"Computed dataset statistics and wrote to {stats_path}.")
        except Exception as err:
            print(f"WARNING: Failed to compute dataset statistics: {err}")
    
    print("="*60)


def main(
    input_path: Path = tyro.MISSING,
    output_path: Path = tyro.MISSING,
    robot_type: str = "dvrk",
    fps: int = 30,
    image_height: int = 540,
    image_width: int = 960,
    num_workers: int = None,
    video_quality: int = 10,
    chunk_size: int = 1000,
    *,
    mono: bool = True,
    resume: bool = True,
    force_overwrite: bool = False,
):
    """
    Convert SutureBot dataset to LeRobot v2.1 format (Parquet + MP4).
    
    Args:
        input_path: Path to the SutureBot dataset root (containing tissue_X folders)
        output_path: Path to save the converted LeRobot dataset
        robot_type: Robot type identifier (default: dvrk)
        fps: Frames per second (default: 30)
        image_height: Target image height (default: 540)
        image_width: Target image width (default: 960)
        num_workers: Number of parallel workers (default: auto)
        video_quality: Video quality 1-10 (default: 10)
        chunk_size: Episodes per chunk (default: 1000)
        mono: Use only main camera (default: False)
        resume: Resume from previous run (default: True)
        force_overwrite: Force overwrite without prompt (default: False)
    
    Dataset structure expected:
        input_path/
            tissue_X/
                phase_folder/
                    demo_folder/
                        left_img_dir/
                        right_img_dir/ (optional)
                        endo_psm1/ (optional)
                        endo_psm2/ (optional)
                        ee_csv.csv
    
    Output features:
        - observation.state: 16-dim (8 per arm: pos, quat, jaw)
        - action: 20-dim (10 per arm: pos, 6D rotation, jaw)
        - observation.images.main: RGB video
        - observation.images.wrist_left: RGB video (if not mono)
        - observation.images.wrist_right: RGB video (if not mono)
    """
    
    if input_path is tyro.MISSING:
        print("Error: --input-path is required")
        print("Usage: python convert_suturebot_to_lerobot.py --input-path /path/to/SutureBot_v2 --output-path /path/to/output")
        return
    
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        return
    
    print(f"SutureBot to LeRobot v2.1 Conversion")
    print(f"Format: Parquet (structured data) + MP4 (video)")
    print()
    
    convert_suturebot_to_lerobot_v21(
        input_path=input_path,
        output_path=output_path,
        robot_type=robot_type,
        fps=fps,
        image_size=(image_height, image_width),
        num_workers=num_workers,
        mono=mono,
        video_quality=video_quality,
        chunk_size=chunk_size,
        resume=resume,
        force_overwrite=force_overwrite,
    )


if __name__ == "__main__":
    tyro.cli(main)
