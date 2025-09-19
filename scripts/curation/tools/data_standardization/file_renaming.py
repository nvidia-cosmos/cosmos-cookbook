import argparse
import os
import re
import shutil
import tempfile
from typing import NoReturn

from scripts.curation.tools.common.s3_utils import (
    is_s3_path,
    sync_local_to_s3,
    sync_s3_to_local,
)


def normalize_filename(filename: str) -> str:
    # Replace whitespace with underscore
    name = re.sub(r"\s+", "_", filename)
    # Remove special characters (anything not alphanumeric, dot, dash, or underscore)
    name = re.sub(r"[^A-Za-z0-9._-]", "", name)
    return name


def rename_files_in_dir(target_dir: str) -> None:
    for root, _, files in os.walk(target_dir):
        for file in files:
            new_name = normalize_filename(file)
            if new_name != file:
                src = os.path.join(root, file)
                dst = os.path.join(root, new_name)
                print(f"Renaming: {src} -> {dst}")
                os.rename(src, dst)


def main() -> NoReturn:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively rename files in a directory or S3 bucket to be processing-friendly "
            "(replace whitespaces and special characters with underscores)."
        )
    )
    parser.add_argument("input_dir", help="Input directory or S3 bucket to process")

    args = parser.parse_args()

    temp_dir = None
    try:
        if is_s3_path(args.input_dir):
            temp_dir = tempfile.mkdtemp()
            sync_s3_to_local(args.input_dir, temp_dir)
            work_dir = temp_dir
        else:
            work_dir = args.input_dir

        rename_files_in_dir(work_dir)

        if is_s3_path(args.input_dir):
            sync_local_to_s3(work_dir, args.input_dir)
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir)
    exit(0)


if __name__ == "__main__":
    main()
