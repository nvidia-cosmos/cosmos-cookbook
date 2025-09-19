#!/usr/bin/env python3
"""
Batch inference script for video anomaly detection.
Processes all videos in a directory and generates text output files.
"""

import argparse
import os
import sys
import shutil
from pathlib import Path
from typing import List, Tuple

def find_mp4_files(directory: str) -> List[str]:
    """Find all .mp4 files in a directory."""
    mp4_files = []
    if os.path.exists(directory):
        for file in os.listdir(directory):
            if file.endswith('.mp4'):
                mp4_files.append(os.path.join(directory, file))
    return sorted(mp4_files)

def run_inference(video_path: str, checkpoint_path: str) -> Tuple[str, float]:
    """Run inference on a single video and return prediction and score."""
    try:
        # Import the inference module
        sys.path.append('cosmos-reason1-reward-7b')
        from inference import predict_video_anomaly, load_model_and_processor

        # Load model once if not already loaded
        if not hasattr(run_inference, 'model'):
            print("Loading model...")
            run_inference.model, run_inference.processor = load_model_and_processor(checkpoint_path)

        prediction, score = predict_video_anomaly(video_path, run_inference.model, run_inference.processor)
        return prediction, score
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return "Error", 0.0

def main():
    parser = argparse.ArgumentParser(description="Batch video inference with text output generation")
    parser.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--video-dir", required=True, help="Directory containing video files")
    parser.add_argument("--output-dir", help="Output directory for text files (default: same as video directory)")

    args = parser.parse_args()

    # Validate video directory exists
    if not os.path.exists(args.video_dir):
        print(f"Error: Video directory not found: {args.video_dir}")
        return 1

    # Find all video files
    videos = find_mp4_files(args.video_dir)

    if not videos:
        print(f"No MP4 files found in {args.video_dir}")
        return 1

    print(f"Found {len(videos)} videos in directory: {args.video_dir}")

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.video_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each video
    print("\n=== Processing videos ===")
    processed_count = 0
    error_count = 0

    for i, video_path in enumerate(videos, 1):
        video_name = os.path.basename(video_path)
        video_stem = Path(video_path).stem  # filename without extension

        print(f"[{i}/{len(videos)}] Processing {video_name}")

        # Run inference
        prediction, score = run_inference(video_path, args.checkpoint)

        # Generate output text file
        output_file = output_dir / f"{video_stem}.txt"
        output_video = output_dir / video_name

        try:
            # Write text file with results
            with open(output_file, 'w') as f:
                f.write(f"Video: {video_name}\n")
                f.write(f"Physical accuracy: {prediction}\n")
                f.write(f"Score (high is good): {score:.4f}\n")
                f.write(f"Checkpoint: {args.checkpoint}\n")

            # Copy video file to output directory (only if different from input directory)
            if output_dir != Path(args.video_dir):
                shutil.copy2(video_path, output_video)
                print(f"  Video copied to: {output_video}")

            print(f"  Prediction: {prediction}, Score: {score:.4f}")
            print(f"  Text output saved to: {output_file}")

            if prediction != "Error":
                processed_count += 1
            else:
                error_count += 1

        except Exception as e:
            print(f"  Error writing files: {e}")
            error_count += 1

    # Print summary
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Total videos found: {len(videos)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Errors encountered: {error_count}")
    print(f"Output directory: {output_dir}")
    if output_dir != Path(args.video_dir):
        print(f"Videos and text files saved to output directory")
    else:
        print(f"Text files saved alongside original videos")

    return 0

if __name__ == "__main__":
    sys.exit(main())
