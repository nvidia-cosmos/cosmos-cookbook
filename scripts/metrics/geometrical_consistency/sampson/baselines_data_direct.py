"""
Direct baselines_data module for running CSE/TSE evaluation on any video directory.
This version doesn't depend on eval_list.json and works with any video path.
Completely independent version - all dependencies are local.
"""
from pathlib import Path
from dataclasses import dataclass

# Use local mvbench modules
from mvbench.data.base import BaseData, CameraView
from mvbench.data.generated import GeneratedStackedData


@dataclass
class SimpleClip:
    """Simple clip representation that doesn't depend on eval_list.json."""
    clip_id: str
    video_path: Path
    category: str = "custom"  # Default category for videos not in eval_list
    chunk_id: str = ""  # Optional, extracted from filename if available

    @classmethod
    def from_video_path(cls, video_path: Path):
        """Create a SimpleClip from a video file path.

        Supports formats:
        - chunk-{chunk_id}-{clip_id}.mp4
        - {clip_id}.mp4
        - any other .mp4 file
        """
        stem = video_path.stem

        # Try to parse chunk-XX-UUID format
        if stem.startswith("chunk-"):
            parts = stem.split('-', 2)
            if len(parts) == 3:
                return cls(
                    clip_id=parts[2],
                    chunk_id=parts[1],
                    video_path=video_path,
                    category="custom"
                )

        # Otherwise use the whole stem as clip_id
        return cls(
            clip_id=stem,
            chunk_id="",
            video_path=video_path,
            category="custom"
        )


def get_data_direct(video_path: Path) -> BaseData:
    """
    Load video data directly from a file path.

    Args:
        video_path: Path to the video file (should be a 2x3 grid video)

    Returns:
        BaseData object for evaluation
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found at {video_path}")

    # Assume standard Cosmos 2x3 grid layout:
    # [LEFT  FRONT  RIGHT]
    # [REAR_L REAR  REAR_R]
    return GeneratedStackedData(
        video_path,
        [
            [CameraView.CROSS_LEFT, CameraView.FRONT, CameraView.CROSS_RIGHT],
            [CameraView.REAR_LEFT, CameraView.REAR_TELE, CameraView.REAR_RIGHT],
        ],
    )


def get_all_videos(input_dir: Path, pattern: str = "*.mp4") -> list[Path]:
    """
    Get all video files from a directory.

    Args:
        input_dir: Directory containing video files
        pattern: Glob pattern for video files (default: "*.mp4")

    Returns:
        List of video file paths
    """
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")

    videos = sorted(input_dir.glob(pattern))

    if not videos:
        raise ValueError(f"No videos found in {input_dir} matching pattern {pattern}")

    return videos
