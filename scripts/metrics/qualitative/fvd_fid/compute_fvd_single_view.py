import argparse
import gc
import json
import os
from glob import glob
from typing import Any, List, Tuple

import decord
import numpy as np
import scipy
import torch
from cdfvd import fvd
from einops import rearrange
from torch.nn.functional import interpolate
from tqdm import tqdm

# pip install cd-fvd decord einops

decord.bridge.set_bridge("torch")
device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def compute_fvd(videos_fake: List[torch.Tensor], videos_real: List[torch.Tensor],
                batch_size: int = 8) -> float:
    """
    Compute FVD score between predicted and ground truth videos.

    Args:
        videos_fake: List of predicted video tensors
        videos_real: List of ground truth video tensors
        batch_size: Batch size for processing

    Returns:
        FVD score as float
    """
    n_batches = (len(videos_fake) - 1) // batch_size + 1

    videos_real_batched = [
        {"video": torch.stack(videos_real[i * batch_size : (i + 1) * batch_size])}
        for i in range(n_batches) if i * batch_size < len(videos_real)
    ]
    videos_fake_batched = [
        {"video": torch.stack(videos_fake[i * batch_size : (i + 1) * batch_size])}
        for i in range(n_batches) if i * batch_size < len(videos_fake)
    ]

    evaluator = fvd.cdfvd("i3d", n_real="full", n_fake="full", ckpt_path=None)
    evaluator.compute_real_stats(videos_real_batched)
    evaluator.compute_fake_stats(videos_fake_batched)

    return evaluator.compute_fvd_from_stats()


def load_videos_for_fvd(video_paths: List[str], num_frames: int = None,
                        target_size: Tuple[int, int] = (224, 224)) -> List[torch.Tensor]:
    """
    Load videos and prepare them for FVD computation.

    Args:
        video_paths: List of paths to video files
        num_frames: Number of frames to extract (None for all frames)
        target_size: Target size for resizing frames (height, width)

    Returns:
        List of video tensors in shape (c, t, h, w)
    """
    videos = []

    for video_path in tqdm(video_paths, desc="Loading videos"):
        vr = decord.VideoReader(video_path)
        frame_idxs = np.arange(0, len(vr))
        raw_video = vr.get_batch(frame_idxs)

        # Limit frames if specified
        if num_frames is not None:
            video = raw_video[:num_frames]
        else:
            video = raw_video

        # Convert to torch tensor and move to device
        # Shape: (t, h, w, c) -> (t, c, h, w)
        video = video.permute(0, 3, 1, 2).float().to(device)

        # Resize to target size
        video = interpolate(video, target_size, mode='bilinear', align_corners=False)

        # Convert to shape expected by FVD: (c, t, h, w) and normalize to [0, 1]
        video_fvd = rearrange(video.cpu(), "t c h w -> c t h w") / 255.0
        videos.append(video_fvd)

        del video, vr
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return videos


def main():
    parser = argparse.ArgumentParser(description="Compute FVD score for single-view videos")
    parser.add_argument(
        "--pred_video_paths",
        type=str,
        required=True,
        help="Path pattern for predicted videos (supports glob patterns, e.g., './path/*.mp4')"
    )
    parser.add_argument(
        "--gt_video_paths",
        type=str,
        required=True,
        help="Path pattern for ground truth videos (supports glob patterns, e.g., './path/*.mp4')"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="Number of frames to use from each video (default: all frames)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for FVD computation (default: 8)"
    )
    parser.add_argument(
        "--target_size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Target size for resizing frames as height width (default: 224 224)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="fvd_results.json",
        help="Output JSON file for results (default: fvd_results.json)"
    )

    args = parser.parse_args()

    # Get video paths
    pred_video_paths = sorted(glob(args.pred_video_paths))
    gt_video_paths = sorted(glob(args.gt_video_paths))

    # Validate paths
    if len(pred_video_paths) == 0:
        raise ValueError(f"No predicted videos found at: {args.pred_video_paths}")
    if len(gt_video_paths) == 0:
        raise ValueError(f"No ground truth videos found at: {args.gt_video_paths}")

    print(f"Found {len(pred_video_paths)} predicted videos")
    print(f"Found {len(gt_video_paths)} ground truth videos")

    assert len(gt_video_paths) == len(pred_video_paths), (
        f"Number of videos mismatch: {len(gt_video_paths)} GT vs {len(pred_video_paths)} predicted"
    )

    # Load ground truth videos
    print("\nLoading ground truth videos...")
    videos_real = load_videos_for_fvd(gt_video_paths, args.num_frames, tuple(args.target_size))
    print(f"Loaded {len(videos_real)} GT videos")

    # Load predicted videos
    print("\nLoading predicted videos...")
    videos_fake = load_videos_for_fvd(pred_video_paths, args.num_frames, tuple(args.target_size))
    print(f"Loaded {len(videos_fake)} predicted videos")

    # Compute FVD score
    print("\nComputing FVD score...")
    fvd_score = compute_fvd(videos_fake, videos_real, args.batch_size)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"FVD Score: {fvd_score:.2f}")

    # Save results to JSON
    results = {
        "FVD": fvd_score,
        "num_pred_videos": len(pred_video_paths),
        "num_gt_videos": len(gt_video_paths),
        "num_frames_per_video": args.num_frames if args.num_frames else "all",
        "batch_size": args.batch_size,
        "target_size": args.target_size,
        "pred_video_pattern": args.pred_video_paths,
        "gt_video_pattern": args.gt_video_paths
    }

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()
