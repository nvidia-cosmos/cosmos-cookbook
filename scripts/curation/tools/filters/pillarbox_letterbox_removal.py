import argparse
import logging as logger
import os
import re
import subprocess
from collections import Counter


# Recursively find all mp4 files in the given directory
def find_mp4_files(root_dir: str) -> list[str]:
    mp4_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".mp4") and not filename.startswith(
                "cropped_"
            ):
                mp4_files.append(os.path.join(dirpath, filename))
    return mp4_files


# Use ffmpeg cropdetect to detect black padding, return crop parameters
def get_crop_params(video_path: str) -> str | None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        video_path,
        "-vf",
        "format=gray,cropdetect=80:2:0",  # More aggressive: grayscale + higher limit
        "-frames:v",
        "500",  # Analyze the first 500 frames
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(
            cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True, check=True
        )
    except subprocess.CalledProcessError as e:
        output = e.stderr
    else:
        output = result.stderr
    # Match crop=xxx:xxx:xxx:xxx
    crops = re.findall(r"crop=([0-9:]+)", output)
    if not crops:
        logger.warning("    [WARN] No crop parameter detected by cropdetect!")
        return None
    # Use the most common crop parameter
    crop_param = Counter(crops).most_common(1)[0][0]
    logger.info(f"    Detected crop parameter: {crop_param}")
    return crop_param


# Adjust crop parameters to preserve original aspect ratio
MIN_CROP_SIZE = 16


def adjust_crop_to_aspect(crop_param: str, orig_w: int, orig_h: int) -> str | None:
    parts = crop_param.split(":")
    if len(parts) != 4 or not all(p.isdigit() for p in parts):
        logger.error(
            f"    [ERROR] Invalid crop parameter: {crop_param}, skipping this file."
        )
        return None
    crop_w, crop_h, crop_x, crop_y = map(int, parts)
    if crop_w < MIN_CROP_SIZE or crop_h < MIN_CROP_SIZE:
        logger.error(
            f"    [ERROR] Crop width/height too small: {crop_param}, skipping this file."
        )
        return None
    orig_aspect = orig_w / orig_h
    crop_aspect = crop_w / crop_h
    # If aspect ratio difference > 1%, adjust
    if abs(crop_aspect - orig_aspect) / orig_aspect > 0.01:
        logger.info(
            f"    [INFO] Adjusting crop to preserve aspect ratio ({orig_aspect:.4f})"
        )
        # Prefer to reduce crop_h (letterbox) or crop_w (pillarbox) as needed
        if crop_aspect > orig_aspect:
            # Too wide, reduce crop_w
            new_crop_w = int(round(crop_h * orig_aspect))
            crop_x = crop_x + (crop_w - new_crop_w) // 2
            crop_w = new_crop_w
        else:
            # Too tall, reduce crop_h
            new_crop_h = int(round(crop_w / orig_aspect))
            crop_y = crop_y + (crop_h - new_crop_h) // 2
            crop_h = new_crop_h
        logger.info(f"    [INFO] Adjusted crop: {crop_w}:{crop_h}:{crop_x}:{crop_y}")
    return f"{crop_w}:{crop_h}:{crop_x}:{crop_y}"


# Crop the video using ffmpeg and the detected crop parameters
def crop_video(input_path: str, crop_param: str, output_path: str) -> None:
    out_dir = os.path.dirname(output_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        input_path,
        "-vf",
        f"crop={crop_param}",
        "-c:a",
        "copy",
        output_path,
    ]
    subprocess.run(cmd, check=True)


# Multi-pass cropping until no more black padding is detected or max passes reached
def multi_pass_crop(video_path: str, output_path: str, max_passes: int = 3) -> bool:
    if not output_path.lower().endswith(".mp4"):
        output_path += ".mp4"
    temp_input = video_path
    last_output = None
    for i in range(max_passes):
        logger.info(f"  Pass {i + 1}: Running cropdetect...")
        crop_param = get_crop_params(temp_input)
        if crop_param is None:
            logger.info(f"  Pass {i + 1}: No crop parameter detected, stopping.")
            break
        # Get original resolution
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0",
                temp_input,
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        width, height = probe.stdout.strip().split(",")
        width, height = int(width), int(height)
        # Check crop_param format before parsing
        adj_crop_param = adjust_crop_to_aspect(crop_param, width, height)
        if adj_crop_param is None:
            logger.info(f"  Pass {i + 1}: Invalid crop parameter, skipping this file.")
            return False
        adj_crop_w, adj_crop_h, _, _ = map(int, adj_crop_param.split(":"))
        if abs(adj_crop_w - width) <= 1 and abs(adj_crop_h - height) <= 1:
            logger.info(f"  Pass {i + 1}: Crop size nearly equals original, stopping.")
            break
        # For the last pass, output to the final output_path
        if i == max_passes - 1:
            next_output = output_path
        else:
            next_output = f"{output_path}.tmp{i + 1}.mp4"
        crop_video(temp_input, adj_crop_param, next_output)
        logger.info(f"  Pass {i + 1}: Cropped with {adj_crop_param}")
        temp_input = next_output
        last_output = next_output
    # After loop, rename last temp file if needed
    if last_output and last_output != output_path and os.path.exists(last_output):
        os.rename(last_output, output_path)
    # Clean up all other temp files except output_path
    for j in range(1, max_passes):
        tmp_file = f"{output_path}.tmp{j}.mp4"
        if os.path.exists(tmp_file) and tmp_file != output_path:
            os.remove(tmp_file)
    return os.path.exists(output_path)


# Main function to process all mp4 files
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove pillarbox or letterbox with text banners from videos and save to output directory."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=".",
        help="Input directory to search for mp4 files",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save cropped videos"
    )
    parser.add_argument(
        "--max_passes",
        type=int,
        default=3,
        help="Maximum number of passes for cropping",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    mp4_files = find_mp4_files(input_dir)
    for video_path in mp4_files:
        rel_path = os.path.relpath(video_path, input_dir)
        dir_name, base_name = os.path.split(rel_path)
        output_subdir = os.path.join(output_dir, dir_name)
        output_path = os.path.join(output_subdir, base_name)
        if os.path.exists(output_path):
            logger.info(f"Skipping {video_path}: already exists in output_dir.")
            continue
        logger.info(f"Processing: {video_path}")
        if not output_path.lower().endswith(".mp4"):
            output_path += ".mp4"
        success = multi_pass_crop(video_path, output_path, max_passes=args.max_passes)
        if success:
            logger.info(f"  Saved: {output_path}")


if __name__ == "__main__":
    main()
