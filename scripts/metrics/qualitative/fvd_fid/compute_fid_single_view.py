import argparse
import gc
import json
import os
from glob import glob
from typing import Any, List

import decord
import numpy as np
import torch
from torch.nn.functional import interpolate
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

# pip install decord torchmetrics[image]

decord.bridge.set_bridge("torch")
device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def compute_fid(
    pred_images: List[torch.Tensor], gt_images: List[torch.Tensor]
) -> float:
    """
    Compute FID score between predicted and ground truth images.

    Args:
        pred_images: List of predicted image tensors
        gt_images: List of ground truth image tensors

    Returns:
        FID score as float
    """
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    fid.reset()

    trunk_size = 512
    for i in range(0, len(pred_images), trunk_size):
        fid.update(torch.cat(pred_images[i : i + trunk_size]), real=False)
        fid.update(torch.cat(gt_images[i : i + trunk_size]), real=True)

    return fid.compute().item()


def load_videos_for_fid(
    video_paths: List[str], num_frames: int = None
) -> List[torch.Tensor]:
    """
    Load videos and extract frames for FID computation.

    Args:
        video_paths: List of paths to video files
        num_frames: Number of frames to extract (None for all frames)

    Returns:
        List of frame tensors
    """
    all_images = []

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

        # Resize to 224x224 for Inception network
        video = interpolate(video, (224, 224), mode="bilinear", align_corners=False)

        # Extract individual frames
        video_images = [frame.unsqueeze(0) for frame in video]
        all_images.extend(video_images)

        del video, vr
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_images


def main():
    parser = argparse.ArgumentParser(
        description="Compute FID score for single-view videos"
    )
    parser.add_argument(
        "--pred_video_paths",
        type=str,
        required=True,
        help="Path pattern for predicted videos (supports glob patterns, e.g., './path/*.mp4')",
    )
    parser.add_argument(
        "--gt_video_paths",
        type=str,
        required=True,
        help="Path pattern for ground truth videos (supports glob patterns, e.g., './path/*.mp4')",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="Number of frames to use from each video (default: all frames)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="fid_results.json",
        help="Output JSON file for results (default: fid_results.json)",
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

    assert len(gt_video_paths) == len(
        pred_video_paths
    ), f"Number of videos mismatch: {len(gt_video_paths)} GT vs {len(pred_video_paths)} predicted"

    # Load ground truth videos
    print("\nLoading ground truth videos...")
    gt_images = load_videos_for_fid(gt_video_paths, args.num_frames)
    print(f"Loaded {len(gt_images)} frames from GT videos")

    # Load predicted videos
    print("\nLoading predicted videos...")
    pred_images = load_videos_for_fid(pred_video_paths, args.num_frames)
    print(f"Loaded {len(pred_images)} frames from predicted videos")

    # Compute FID score
    print("\nComputing FID score...")
    fid_score = compute_fid(pred_images, gt_images)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"FID Score: {fid_score:.2f}")

    # Save results to JSON
    results = {
        "FID": fid_score,
        "num_pred_videos": len(pred_video_paths),
        "num_gt_videos": len(gt_video_paths),
        "num_frames_per_video": args.num_frames if args.num_frames else "all",
        "total_pred_frames": len(pred_images),
        "total_gt_frames": len(gt_images),
        "pred_video_pattern": args.pred_video_paths,
        "gt_video_pattern": args.gt_video_paths,
    }

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()
