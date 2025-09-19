import logging as logger
import os
import subprocess


def is_s3_path(path: str) -> bool:
    return path.startswith("s3://")


def sync_s3_to_local(s3_path: str, local_path: str) -> None:
    """Sync a single file from S3 to local path"""
    # Ensure AWS profile is set
    if "AWS_PROFILE" not in os.environ:
        os.environ["AWS_PROFILE"] = "lha-share"

    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Use cp instead of sync for single file
    cmd = ["s5cmd", "cp", s3_path, local_path]
    logger.info(f"Copying from S3: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"S3 copy error: {result.stderr}")
        raise RuntimeError(f"Failed to copy from {s3_path} to {local_path}")

    logger.info(f"S3 copy output: {result.stdout}")

    # Verify file was copied
    if not os.path.exists(local_path):
        raise RuntimeError(f"Local file {local_path} was not created")


def sync_local_to_s3(local_path: str, s3_path: str) -> None:
    """Sync a directory from local to S3"""
    # Ensure AWS profile is set
    if "AWS_PROFILE" not in os.environ:
        os.environ["AWS_PROFILE"] = "lha-share"

    # Ensure s3_path ends with a slash (prefix)
    if not s3_path.endswith("/"):
        s3_path += "/"

    cmd = ["s5cmd", "sync", local_path, s3_path]
    logger.info(f"Syncing to S3: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"S3 sync error: {result.stderr}")
        raise RuntimeError(f"Failed to sync from {local_path} to {s3_path}")

    logger.info(f"S3 sync output: {result.stdout}")
