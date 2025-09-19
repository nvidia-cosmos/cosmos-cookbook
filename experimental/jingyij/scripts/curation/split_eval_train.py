#!/usr/bin/env python3
"""
Script to split .mp4 files from input directory into evaluation and training sets
based on a given ratio, supporting both local and S3 paths.

Usage:
    python split_eval_train.py --input_dir /path/to/input --output_dir /path/to/output --ratio 1:9
    python split_eval_train.py --input_dir s3://bucket/input --output_dir s3://bucket/output --ratio 2:8
"""

import argparse
import logging
import os
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

from s3_utils import is_s3_path, sync_local_to_s3, sync_s3_to_local


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_video_files(directory: str) -> List[str]:
    """Find all .mp4 files in directory and subdirectories."""
    video_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".mp4"):
                video_files.append(os.path.join(root, file))
    return video_files


def parse_ratio(ratio_str: str) -> Tuple[int, int]:
    """
    Parse ratio string like '1:9' into tuple of integers.
    
    Args:
        ratio_str: String in format 'eval:train' like '1:9'
        
    Returns:
        Tuple of (eval_ratio, train_ratio)
    """
    try:
        eval_ratio, train_ratio = map(int, ratio_str.split(':'))
        if eval_ratio <= 0 or train_ratio <= 0:
            raise ValueError("Ratios must be positive integers")
        return eval_ratio, train_ratio
    except ValueError as e:
        raise ValueError(f"Invalid ratio format '{ratio_str}'. Expected format: 'eval:train' (e.g., '1:9')") from e


def find_s3_video_files(s3_path: str) -> List[str]:
    """
    Find all .mp4 files in an S3 directory using s5cmd.
    
    Args:
        s3_path: S3 directory path
        
    Returns:
        List of S3 paths to .mp4 files
    """
    # Ensure AWS profile is set
    if "AWS_PROFILE" not in os.environ:
        os.environ["AWS_PROFILE"] = "lha-share"
    
    # Use s5cmd to list files recursively
    s3_prefix = s3_path.rstrip('/') + '/'
    cmd = ["s5cmd", "ls", f"{s3_prefix}**/*.mp4"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        video_files = []
        
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                # s5cmd ls output format: "2023/01/01 00:00:00    1234 s3://bucket/path/file.mp4"
                # Sometimes the format might be different, so let's be more robust
                parts = line.strip().split()
                if len(parts) >= 1:
                    # Look for the part that contains .mp4 and starts with s3://
                    for part in parts:
                        if part.endswith('.mp4') and part.startswith('s3://'):
                            video_files.append(part)
                            break
                    else:
                        # If no s3:// found, the last part might be the relative path
                        # We need to construct the full S3 path
                        if parts[-1].endswith('.mp4'):
                            if not parts[-1].startswith('s3://'):
                                # Construct full S3 path
                                full_s3_path = f"{s3_prefix}{parts[-1]}"
                                video_files.append(full_s3_path)
                            else:
                                video_files.append(parts[-1])
        
        logger.info(f"Found {len(video_files)} .mp4 files in S3")
        return video_files
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to list S3 files: {e.stderr}")
        # Fallback: try non-recursive listing
        cmd_simple = ["s5cmd", "ls", f"{s3_prefix}*.mp4"]
        try:
            result = subprocess.run(cmd_simple, capture_output=True, text=True, check=True)
            video_files = []
            
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        for part in parts:
                            if part.endswith('.mp4') and part.startswith('s3://'):
                                video_files.append(part)
                                break
                        else:
                            if parts[-1].endswith('.mp4'):
                                if not parts[-1].startswith('s3://'):
                                    full_s3_path = f"{s3_prefix}{parts[-1]}"
                                    video_files.append(full_s3_path)
                                else:
                                    video_files.append(parts[-1])
            
            logger.info(f"Found {len(video_files)} .mp4 files in S3 (non-recursive)")
            return video_files
            
        except subprocess.CalledProcessError as e2:
            logger.error(f"Failed to list S3 files: {e2.stderr}")
            raise RuntimeError(f"Could not list files in S3 path: {s3_path}")


def split_files_by_ratio(video_files: List[str], eval_ratio: int, train_ratio: int) -> Tuple[List[str], List[str]]:
    """
    Split list of video files into eval and train sets based on ratio.
    
    Args:
        video_files: List of video file paths
        eval_ratio: Evaluation set ratio
        train_ratio: Training set ratio
        
    Returns:
        Tuple of (eval_files, train_files)
    """
    total_ratio = eval_ratio + train_ratio
    eval_count = int(len(video_files) * eval_ratio / total_ratio)
    
    # Shuffle files for random split
    shuffled_files = video_files.copy()
    random.shuffle(shuffled_files)
    
    eval_files = shuffled_files[:eval_count]
    train_files = shuffled_files[eval_count:]
    
    logger.info(f"Split {len(video_files)} files into {len(eval_files)} eval and {len(train_files)} train files")
    return eval_files, train_files


def copy_file(src_path: str, dst_path: str, is_src_s3: bool, is_dst_s3: bool) -> None:
    """
    Copy a single file between local/S3 locations.
    
    Args:
        src_path: Source file path
        dst_path: Destination file path
        is_src_s3: Whether source is S3 path
        is_dst_s3: Whether destination is S3 path
    """
    # Double-check S3 path detection
    actual_src_s3 = is_s3_path(src_path)
    actual_dst_s3 = is_s3_path(dst_path)
    
    if actual_src_s3 != is_src_s3:
        logger.warning(f"Source S3 detection mismatch: {src_path} - expected {is_src_s3}, actual {actual_src_s3}")
    if actual_dst_s3 != is_dst_s3:
        logger.warning(f"Destination S3 detection mismatch: {dst_path} - expected {is_dst_s3}, actual {actual_dst_s3}")
    
    # Use actual detection
    is_src_s3 = actual_src_s3
    is_dst_s3 = actual_dst_s3
    
    if is_src_s3 and is_dst_s3:
        # S3 to S3: download to temp, then upload
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_path = temp_file.name
        try:
            sync_s3_to_local(src_path, temp_path)
            sync_local_to_s3(temp_path, dst_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    elif is_src_s3 and not is_dst_s3:
        # S3 to local
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        sync_s3_to_local(src_path, dst_path)
    elif not is_src_s3 and is_dst_s3:
        # Local to S3
        sync_local_to_s3(src_path, dst_path)
    else:
        # Local to local
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(src_path, dst_path)


def process_split(
    input_dir: str,
    output_dir: str,
    video_files: List[str],
    split_name: str
) -> None:
    """
    Process a split (eval or train) by copying files to the appropriate output directory.
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        video_files: List of video files for this split
        split_name: Name of the split ('eval' or 'train')
    """
    is_input_s3 = is_s3_path(input_dir)
    is_output_s3 = is_s3_path(output_dir)
    
    split_output_dir = f"{output_dir.rstrip('/')}/{split_name}/"
    
    logger.info(f"Processing {split_name} split with {len(video_files)} files")
    
    for i, video_file in enumerate(video_files, 1):
        # Get relative path from input directory
        if is_input_s3:
            rel_path = video_file.replace(input_dir.rstrip('/') + '/', '')
        else:
            rel_path = os.path.relpath(video_file, input_dir)
        
        # Construct destination path
        dst_path = f"{split_output_dir}{rel_path}"
        
        logger.info(f"Copying {i}/{len(video_files)}: {rel_path}")
        
        try:
            copy_file(video_file, dst_path, is_input_s3, is_output_s3)
        except Exception as e:
            logger.error(f"Failed to copy {video_file}: {str(e)}")
            continue


def main():
    parser = argparse.ArgumentParser(description="Split video files into eval and train sets")
    parser.add_argument("--input_dir", required=True, help="Input directory (local or S3 path)")
    parser.add_argument("--output_dir", required=True, help="Output directory (local or S3 path)")
    parser.add_argument("--ratio", required=True, help="Split ratio in format 'eval:train' (e.g., '1:9')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits")
    
    args = parser.parse_args()
    
    # Set random seed for reproducible results
    random.seed(args.seed)
    
    # Parse ratio
    eval_ratio, train_ratio = parse_ratio(args.ratio)
    logger.info(f"Using split ratio - eval:{eval_ratio}, train:{train_ratio}")
    
    # Handle input directory and find video files
    input_dir = args.input_dir
    if is_s3_path(input_dir):
        logger.info(f"Input is S3 path: {input_dir}")
        logger.info("Searching for .mp4 files in S3...")
        video_files = find_s3_video_files(input_dir)
    else:
        logger.info(f"Input is local path: {input_dir}")
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        logger.info("Searching for .mp4 files...")
        video_files = find_video_files(input_dir)
    
    if not video_files:
        logger.warning("No .mp4 files found in input directory")
        return
    
    logger.info(f"Found {len(video_files)} .mp4 files")
    
    # Debug: Show first few files to verify paths
    if video_files:
        logger.info("Sample video files found:")
        for i, vf in enumerate(video_files[:3]):
            logger.info(f"  {i+1}: {vf} (is_s3: {is_s3_path(vf)})")
        if len(video_files) > 3:
            logger.info(f"  ... and {len(video_files) - 3} more")
    
    # Split files
    eval_files, train_files = split_files_by_ratio(video_files, eval_ratio, train_ratio)
    
    # Process eval split
    if eval_files:
        process_split(input_dir, args.output_dir, eval_files, "eval")
    
    # Process train split
    if train_files:
        process_split(input_dir, args.output_dir, train_files, "train")
    
    logger.info("Split completed successfully!")
    logger.info(f"Eval set: {len(eval_files)} files -> {args.output_dir}/eval/")
    logger.info(f"Train set: {len(train_files)} files -> {args.output_dir}/train/")


if __name__ == "__main__":
    main()
