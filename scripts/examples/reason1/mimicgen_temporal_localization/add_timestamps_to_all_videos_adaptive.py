#!/usr/bin/env python3
"""
Script to add timestamps to all video files in a folder at 30 fps with 2 decimal places.

This script processes all MP4 files in the input directory and adds timestamps
with a single centered timestamp at the bottom of each frame.
"""

import os
import sys
import glob
import argparse
from pathlib import Path
import subprocess

# Add the cosmos_reason1_utils to the path
sys.path.insert(0, str(Path.home() / "repos" / "cosmos-reason1" / "cosmos_reason1_utils" / "src"))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm
import functools

# Import the overlay functionality
from cosmos_reason1_utils.vision import OverlayConfig

# FPS_COEFFS = {4: 1.07, 8: 1.5, 12: 1.85, 16: 2.13}
FPS_COEFFS = {8: 1.5}


@functools.cache
def _get_overlay_font_path(family: str) -> str:
    """Return the path to the font for overlaying text on images."""
    return fm.findfont(fm.FontProperties(family=family))

def convert_to_h264(input_path: str, output_path: str = None) -> str:
    """Convert video to H.264 codec using ffmpeg."""
    if output_path is None:
        # Create output path with _h264 suffix
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_h264.mp4"
    
    try:
        # FFmpeg command to convert to H.264
        cmd = [
            'ffmpeg', '-i', input_path,
            '-c:v', 'libx264',           # Use H.264 codec
            '-preset', 'medium',          # Encoding preset (fast, medium, slow)
            '-crf', '23',                 # Constant Rate Factor (18-28 is good)
            '-y',                         # Overwrite output file
            output_path
        ]
        
        print(f"🔄 Converting to H.264: {os.path.basename(input_path)} → {os.path.basename(output_path)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Successfully converted to H.264: {output_path}")
            return output_path
        else:
            print(f"❌ FFmpeg conversion failed:")
            print(f"   Error: {result.stderr}")
            return None
            
    except FileNotFoundError:
        print("❌ FFmpeg not found. Please install ffmpeg to convert to H.264.")
        return None
    except Exception as e:
        print(f"❌ Error during FFmpeg conversion: {e}")
        return None


def add_timestamps_to_video(input_video_path: str, output_video_path: str, fps: float = 30.0, coeff: float = 1.0):
    """
    Add timestamps to video frames with single centered timestamp.
    
    Args:
        input_video_path: Path to input video
        output_video_path: Path to output video
        fps: Frames per second (default: 30.0)
        coeff: Coefficient to multiply border_height and font_size (default: 1.0)
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    
    # Configuration for overlay text - single centered timestamp
    # Apply coefficient to border_height and font_size
    config = OverlayConfig(
        border_height=int(28 * coeff),  # Height of black border multiplied by coefficient
        temporal_path_size=1,  # Single position (centered)
        font_family="DejaVu Sans Mono",  # Font family
        font_size=int(20 * coeff),  # Font size multiplied by coefficient
        font_color="white"  # Text color
    )
    
    # Load font
    font = ImageFont.truetype(_get_overlay_font_path(config.font_family), config.font_size)
    
    # Open video
    cap = cv2.VideoCapture(input_video_path)
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  Input video: {width}x{height}, {original_fps} fps, {total_frames} frames")
    print(f"  Using coefficient: {coeff} (border_height: {config.border_height}, font_size: {config.font_size})")
    
    # Calculate new dimensions with border
    new_height = height + config.border_height
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, new_height))
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Create new image with black border at the bottom
        new_image = Image.new("RGB", (width, new_height), color="black")
        
        # Paste original image at the top
        new_image.paste(pil_image, (0, 0))
        
        # Draw text on the black border
        draw = ImageDraw.Draw(new_image)
        
        # Calculate timestamp for current frame (seconds since start)
        total_seconds = frame_count / fps
        text = f"{total_seconds:.2f}s"
        
        # Get text dimensions
        try:
            # Get text bounding box
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            # Fallback for older PIL versions
            text_width, text_height = draw.textsize(text, font=font)
        
        # Define available positions (cycling through horizontal positions)
        position_idx = frame_count % config.temporal_path_size
        section_width = width // config.temporal_path_size
        
        # Calculate x position based on cycling position
        section_center_x = position_idx * section_width + section_width // 2
        text_x = section_center_x - text_width // 2
        
        # Ensure text doesn't go outside bounds
        text_x = max(0, min(text_x, width - text_width))
        
        # Center vertically in the border
        text_y = height + (config.border_height - text_height) // 2
        
        # Draw the timestamp
        draw.text((text_x, text_y), text, fill=config.font_color, font=font)
        
        # Convert back to BGR for OpenCV
        new_image_bgr = cv2.cvtColor(np.array(new_image), cv2.COLOR_RGB2BGR)
        
        # Write frame
        out.write(new_image_bgr)
        
        frame_count += 1
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"  ✓ Processed {frame_count} frames with timestamps at {fps} fps")


def process_all_videos(input_dir: str, output_dir: str):
    """
    Process all MP4 files in the input directory for each FPS coefficient.
    
    Args:
        input_dir: Directory containing input videos
        output_dir: Directory to save processed videos
    """
    # Find all MP4 files
    video_pattern = os.path.join(input_dir, "*.mp4")
    video_files = glob.glob(video_pattern)
    
    if not video_files:
        print(f"No MP4 files found in {input_dir}")
        return
    
    print(f"Found {len(video_files)} video files to process")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"FPS coefficients: {FPS_COEFFS}")
    print("=" * 60)
    
    total_videos = len(video_files) * len(FPS_COEFFS)
    video_count = 0
    
    # Process each video for each FPS coefficient
    for fps_key, coeff in FPS_COEFFS.items():
        print(f"\n=== Processing videos for FPS {fps_key} (coefficient: {coeff}) ===")
        
        for i, input_video in enumerate(sorted(video_files), 1):
            if "h264" in input_video:
                continue
            filename = os.path.basename(input_video)
            name_without_ext = os.path.splitext(filename)[0]
            # output_video = os.path.join(output_dir, f"{name_without_ext}_fps{fps_key}.mp4")
            output_video = os.path.join(output_dir, filename)
            
            video_count += 1
            print(f"[{video_count}/{total_videos}] Processing: {filename} (FPS {fps_key})")
            
            try:
                add_timestamps_to_video(input_video, output_video, fps=30.0, coeff=coeff)
                h264_output = convert_to_h264(output_video)
                
                # Delete the intermediate mp4v file if H.264 conversion was successful
                if h264_output and os.path.exists(h264_output):
                    try:
                        os.remove(output_video)
                        os.rename(h264_output, output_video)
                        print(f"  🗑️  Removed intermediate file: {os.path.basename(output_video)}")
                        print(f"  ✓ Final output: {h264_output}")
                    except Exception as e:
                        print(f"  ⚠️  Could not remove intermediate file: {e}")
                else:
                    print(f"  ⚠️  H.264 conversion failed, keeping original: {output_video}")
            except Exception as e:
                print(f"  ✗ Error processing {filename} for FPS {fps_key}: {e}")
            
            print("-" * 40)
    
    print("All videos processed!")


def main():
    """Main function to process all videos."""
    parser = argparse.ArgumentParser(
        description="Add timestamps to all video files in a folder at 30 fps with 2 decimal places.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-i", "--input-dir",
        required=True,
        help="Input directory containing MP4 video files"
    )
    parser.add_argument(
        "-o", "--output-dir",
        required=True,
        help="Output directory to save processed videos with timestamps"
    )
    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    print("Adding timestamps to all videos...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target FPS: 30.0")
    print(f"Timestamp style: Single centered timestamp")
    print(f"FPS coefficients: {FPS_COEFFS}")
    print("=" * 60)
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return
    
    try:
        process_all_videos(input_dir, output_dir)
    except Exception as e:
        print(f"Error processing videos: {e}")
        return


if __name__ == "__main__":
    main()
