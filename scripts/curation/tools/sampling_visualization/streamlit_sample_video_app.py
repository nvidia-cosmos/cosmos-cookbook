import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

import streamlit as st

from scripts.curation.tools.common.s3_utils import is_s3_path, sync_s3_to_local


def find_mp4_files(directory: str) -> List[Path]:
    dir_path = Path(directory)
    return list(dir_path.rglob("*.mp4"))


def main(input_dir: str, nsamples: int = 100) -> None:
    st.title("Sample Video Browser")

    temp_dir: Optional[str] = None
    try:
        # Handle S3 input
        if is_s3_path(input_dir):
            temp_dir = tempfile.mkdtemp()
            sync_s3_to_local(input_dir, temp_dir)
            work_dir = temp_dir
        else:
            work_dir = input_dir

        # Find and sample videos
        all_videos = find_mp4_files(work_dir)
        if not all_videos:
            st.warning("No .mp4 files found.")
            return

        if len(all_videos) < nsamples:
            st.warning(
                f"Only {len(all_videos)} videos found, showing all available videos."
            )
            sampled_videos = all_videos
        else:
            sampled_videos = random.sample(all_videos, nsamples)

        # Pagination
        videos_per_page = 12
        total_pages = (len(sampled_videos) + videos_per_page - 1) // videos_per_page
        page = st.number_input(
            "Page", min_value=1, max_value=total_pages, value=1, step=1
        )

        start_idx = (page - 1) * videos_per_page
        end_idx = min(start_idx + videos_per_page, len(sampled_videos))
        videos_to_show = sampled_videos[start_idx:end_idx]

        # Display videos in a 3x4 grid
        for row in range(3):
            cols = st.columns(4)
            for col in range(4):
                idx = row * 4 + col
                if idx < len(videos_to_show):
                    video_path = str(videos_to_show[idx])
                    with cols[col]:
                        st.video(video_path)
                        st.caption(os.path.basename(video_path))

    finally:
        if temp_dir:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sample Video Browser")
    parser.add_argument(
        "--input_dir", required=True, help="Input directory (local path or S3 bucket)"
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=100,
        help="Number of samples to display (default: 100)",
    )
    args = parser.parse_args()
    main(args.input_dir, args.nsamples)
