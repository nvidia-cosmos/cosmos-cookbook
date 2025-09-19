import json
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import streamlit as st
from scripts.curation.tools.common.s3_utils import is_s3_path, sync_s3_to_local


def find_video_files_with_metadata(directory: str, extensions: List[str] = None) -> List[Dict[str, Any]]:
    """Find video files in clips/ directory and match them with metadata from metas/v0/ directory."""
    if extensions is None:
        extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"]
    
    dir_path = Path(directory)
    clips_dir = dir_path / "clips"
    meta_dir = dir_path / "metas" / "v0"
    
    if not clips_dir.exists():
        st.error(f"❌ 'clips/' directory not found in {directory}")
        return []
    
    if not meta_dir.exists():
        st.error(f"❌ 'metas/v0/' directory not found in {directory}")
        return []
    
    video_files = []
    
    # Find all video files in clips/ directory (recursive)
    for ext in extensions:
        pattern = f"*{ext}"
        video_files.extend(clips_dir.rglob(pattern))
        # Also search for uppercase extensions
        pattern_upper = f"*{ext.upper()}"
        video_files.extend(clips_dir.rglob(pattern_upper))
    
    # Remove duplicates and sort
    video_files = sorted(list(set(video_files)))
    
    # Create list of video files with their metadata
    videos_with_metadata = []
    missing_metadata = []
    
    for video_path in video_files:
        # Get the video filename without extension
        video_stem = video_path.stem
        
        # Look for corresponding JSON file in metas/v0/
        json_path = meta_dir / f"{video_stem}.json"
        
        video_info = {
            "video_path": video_path,
            "video_filename": video_path.name,
            "relative_path": video_path.relative_to(clips_dir),
            "metadata_path": json_path if json_path.exists() else None,
            "qwen_caption": None,
            "metadata": None
        }
        
        # Try to load metadata if JSON exists
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    video_info["metadata"] = metadata
                    
                    # Extract first qwen_caption from windows
                    if "windows" in metadata and len(metadata["windows"]) > 0:
                        first_window = metadata["windows"][0]
                        if "qwen_caption" in first_window:
                            video_info["qwen_caption"] = first_window["qwen_caption"]
                            
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                st.warning(f"⚠️ Could not parse metadata for {video_path.name}: {str(e)}")
        else:
            missing_metadata.append(video_path.name)
        
        videos_with_metadata.append(video_info)
    
    # Report missing metadata
    if missing_metadata:
        st.warning(f"⚠️ Found {len(missing_metadata)} videos without corresponding metadata files")
        with st.expander("📋 Videos missing metadata"):
            for filename in missing_metadata[:20]:  # Show first 20
                st.text(f"❌ {filename}")
            if len(missing_metadata) > 20:
                st.text(f"... and {len(missing_metadata) - 20} more files")
    
    return videos_with_metadata


def find_video_files(directory: str, extensions: List[str] = None) -> List[Path]:
    """Find video files recursively in directory and subdirectories (legacy function)."""
    if extensions is None:
        extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"]
    
    dir_path = Path(directory)
    video_files = []
    
    # Use rglob to recursively find files with any of the video extensions
    for ext in extensions:
        pattern = f"*{ext}"
        video_files.extend(dir_path.rglob(pattern))
        # Also search for uppercase extensions
        pattern_upper = f"*{ext.upper()}"
        video_files.extend(dir_path.rglob(pattern_upper))
    
    # Remove duplicates and sort
    video_files = sorted(list(set(video_files)))
    return video_files


def get_directory_structure_info(directory: str) -> Tuple[int, List[str]]:
    """Get information about the directory structure."""
    dir_path = Path(directory)
    subdirs = []
    file_count = 0
    
    for item in dir_path.rglob("*"):
        if item.is_file():
            file_count += 1
        elif item.is_dir():
            # Get relative path from root directory
            rel_path = item.relative_to(dir_path)
            if str(rel_path) != ".":
                subdirs.append(str(rel_path))
    
    return file_count, sorted(subdirs)


def main(input_dir: str, nsamples: int = 100) -> None:
    st.title("Video Browser with Captions")
    st.markdown(f"**Input Directory:** `{input_dir}`")
    st.info("📁 Expected structure: `clips/` directory for videos and `metas/v0/` directory for JSON metadata")

    temp_dir: Optional[str] = None
    try:
        # Handle S3 input
        if is_s3_path(input_dir):
            st.info("🔄 Syncing S3 data to local directory...")
            temp_dir = tempfile.mkdtemp()
            
            with st.spinner("Downloading from S3..."):
                sync_s3_to_local(input_dir, temp_dir)
            
            work_dir = temp_dir
            st.success("✅ S3 sync completed!")
        else:
            work_dir = input_dir

        # Check for expected directory structure
        clips_dir = Path(work_dir) / "clips"
        meta_dir = Path(work_dir) / "metas" / "v0"
        
        # Display directory structure status
        col1, col2, col3 = st.columns(3)
        with col1:
            if clips_dir.exists():
                st.success("✅ clips/ directory found")
            else:
                st.error("❌ clips/ directory missing")
        
        with col2:
            if meta_dir.exists():
                st.success("✅ metas/v0/ directory found")
            else:
                st.error("❌ metas/v0/ directory missing")
        
        with col3:
            if clips_dir.exists() and meta_dir.exists():
                st.success("✅ Structure valid")
            else:
                st.error("❌ Invalid structure")

        # Video file extension selector
        available_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"]
        selected_extensions = st.multiselect(
            "Select video file extensions to include:",
            available_extensions,
            default=[".mp4"],
            help="Choose which video file types to search for in clips/ directory"
        )

        if not selected_extensions:
            st.warning("Please select at least one file extension.")
            return

        # Find videos with metadata
        with st.spinner("Searching for videos and matching metadata..."):
            videos_with_metadata = find_video_files_with_metadata(work_dir, selected_extensions)
        
        if not videos_with_metadata:
            st.warning(f"No video files found with extensions: {', '.join(selected_extensions)}")
            return

        # Display statistics
        total_videos = len(videos_with_metadata)
        videos_with_captions = len([v for v in videos_with_metadata if v["qwen_caption"]])
        videos_with_metadata_files = len([v for v in videos_with_metadata if v["metadata_path"]])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Videos", total_videos)
        with col2:
            st.metric("With Metadata", videos_with_metadata_files)
        with col3:
            st.metric("With Captions", videos_with_captions)

        # Display some example paths to show the structure
        if len(videos_with_metadata) > 0:
            with st.expander("📋 Sample video-metadata pairs"):
                sample_videos = videos_with_metadata[:5]
                for video_info in sample_videos:
                    st.text(f"📹 Video: {video_info['relative_path']}")
                    if video_info['metadata_path']:
                        st.text(f"📄 Metadata: ✅ Found")
                        if video_info['qwen_caption']:
                            caption_preview = video_info['qwen_caption'][:100] + "..." if len(video_info['qwen_caption']) > 100 else video_info['qwen_caption']
                            st.text(f"💬 Caption: {caption_preview}")
                        else:
                            st.text(f"💬 Caption: ❌ Missing")
                    else:
                        st.text(f"📄 Metadata: ❌ Missing")
                    st.text("---")

        # Filter options
        show_only_with_captions = st.checkbox(
            "Show only videos with captions", 
            value=False,
            help="Filter to show only videos that have qwen_caption in their metadata"
        )
        
        if show_only_with_captions:
            filtered_videos = [v for v in videos_with_metadata if v["qwen_caption"]]
            st.info(f"Filtered to {len(filtered_videos)} videos with captions")
        else:
            filtered_videos = videos_with_metadata

        if not filtered_videos:
            st.warning("No videos match the current filters.")
            return

        # Sampling controls
        actual_nsamples = min(nsamples, len(filtered_videos))
        nsamples = st.slider(
            "Number of videos to sample:",
            min_value=1,
            max_value=min(len(filtered_videos), 200),  # Cap at 200 for performance
            value=actual_nsamples,
            help="Randomly sample this many videos from filtered videos"
        )

        if len(filtered_videos) <= nsamples:
            st.info(f"Showing all {len(filtered_videos)} filtered videos.")
            sampled_videos = filtered_videos
        else:
            # Add random seed control for reproducible sampling
            random_seed = st.number_input("Random seed (for reproducible sampling):", value=42, min_value=0)
            random.seed(random_seed)
            sampled_videos = random.sample(filtered_videos, nsamples)

        # Pagination
        videos_per_page = st.selectbox("Videos per page:", [3, 6, 9, 12], index=1)
        total_pages = (len(sampled_videos) + videos_per_page - 1) // videos_per_page
        
        if total_pages > 1:
            page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
        else:
            page = 1

        start_idx = (page - 1) * videos_per_page
        end_idx = min(start_idx + videos_per_page, len(sampled_videos))
        videos_to_show = sampled_videos[start_idx:end_idx]

        st.markdown(f"**Showing videos {start_idx + 1}-{end_idx} of {len(sampled_videos)} sampled videos**")

        # Display videos with captions in a vertical layout
        for i, video_info in enumerate(videos_to_show):
            st.markdown("---")
            
            # Create two columns: video on left, metadata on right
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Display video
                video_path = str(video_info["video_path"])
                st.video(video_path)
                
                # Display filename
                st.markdown(f"**📁 Filename:** `{video_info['video_filename']}`")
                st.markdown(f"**📂 Path:** `{video_info['relative_path']}`")
            
            with col2:
                # Display metadata information
                if video_info["metadata_path"]:
                    st.markdown("**📄 Metadata Status:** ✅ Found")
                    
                    # Display caption
                    if video_info["qwen_caption"]:
                        st.markdown("**💬 Qwen Caption:**")
                        st.text_area(
                            f"Caption for video {start_idx + i + 1}",
                            value=video_info["qwen_caption"],
                            height=150,
                            disabled=True,
                            label_visibility="collapsed"
                        )
                    else:
                        st.markdown("**💬 Qwen Caption:** ❌ Not found in metadata")
                    
                    # Show additional metadata if available
                    if video_info["metadata"]:
                        with st.expander("🔍 View full metadata"):
                            # Show key metadata fields
                            metadata = video_info["metadata"]
                            if "duration_span" in metadata:
                                st.text(f"Duration: {metadata['duration_span']}")
                            if "width" in metadata and "height" in metadata:
                                st.text(f"Resolution: {metadata['width']}x{metadata['height']}")
                            if "framerate" in metadata:
                                st.text(f"Framerate: {metadata['framerate']:.2f} fps")
                            if "num_frames" in metadata:
                                st.text(f"Frames: {metadata['num_frames']}")
                            
                            # Show raw JSON
                            st.json(metadata)
                else:
                    st.markdown("**📄 Metadata Status:** ❌ No corresponding JSON file found")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Video Browser with Captions - Displays videos with their corresponding qwen_caption metadata"
    )
    parser.add_argument(
        "--input_dir", 
        required=True, 
        help="Input directory (local path or S3 bucket). Must contain 'clips/' and 'metas/v0/' subdirectories."
    )
    parser.add_argument(
        "--nsamples", 
        type=int, 
        default=100, 
        help="Initial number of samples to display (default: 100). Can be adjusted in the UI."
    )
    args = parser.parse_args()
    
    main(args.input_dir, args.nsamples)
