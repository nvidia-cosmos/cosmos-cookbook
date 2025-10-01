import argparse
import gc
import importlib.util
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_utils
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Configuration
NUM_CHUNKS = int(os.environ.get("NUM_CHUNKS", 1))
START_CHUNK = int(os.environ.get("START_CHUNK", 0))
END_CHUNK = int(os.environ.get("END_CHUNK", NUM_CHUNKS))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", END_CHUNK - START_CHUNK))
FONT_SIZE = int(os.environ.get("FONT_SIZE", 50))
TARGET_WIDTH = int(os.environ.get("TARGET_WIDTH", 1920))
TARGET_HEIGHT = int(os.environ.get("TARGET_HEIGHT", 1080))
TARGET_SIZE = (TARGET_WIDTH, TARGET_HEIGHT)


def get_annotations(annotations_file):
    with open(annotations_file, "r") as f:
        annotations = json.load(f)

    # Check if annotations is a list or a dict with nested structure
    if isinstance(annotations, dict):
        # If it's a dict, look for common keys that might contain the list
        if "annotations" in annotations:
            return annotations["annotations"]
        elif "data" in annotations:
            return annotations["data"]
        elif "items" in annotations:
            return annotations["items"]
        else:
            # If it's a dict but we don't know the structure, return as is
            print(
                f"Warning: Annotations file contains a dict with keys: {list(annotations.keys())}"
            )
            return annotations

    return annotations


def split_list_to_nested_list(lst):
    split_by_chunk = np.array_split(lst, NUM_CHUNKS)[START_CHUNK:END_CHUNK]
    return split_by_chunk


# prompt style:
# <image>\n<image>\nThe first image is the original, and the second is an overlay. Bright numeric IDs are labeled at the center of certain visual objects in the second image. Is the car with numeric ID 2 in lane 4? Answer with 'yes' or 'no'. If no, please provide the correct lane.


def som_overlay_and_save_iccv_show(
    image_path, annotation, save_path, font_path=None, font_size=None
):
    image = Image.open(image_path).convert("RGBA")
    # Resize image to target size
    image = image.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
    masks = annotation["rle"]
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    if font_size is None:
        font_size = FONT_SIZE
    if font_path is None:
        # Use font file in the same directory as this script
        script_dir = Path(__file__).parent
        font_path = script_dir / "DejaVuSans-Bold.ttf"
    font = ImageFont.truetype(font_path, font_size)

    text_infos = []

    if save_path.exists():
        return annotation

    # Process masks
    for i, mask in enumerate(masks):
        mask = mask_utils.decode(mask)
        # Resize mask to match the resized image
        mask = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
        mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            128,
        )
        colored_mask = Image.new("RGBA", image.size, color)
        overlay.paste(colored_mask, (0, 0), mask_image)

        draw = ImageDraw.Draw(overlay)
        text = str(i)
        text_size = draw.textbbox((0, 0), text, font=font)
        text_width = text_size[2] - text_size[0]
        text_height = text_size[3] - text_size[1]

        mask_indices = np.argwhere(mask)
        if mask_indices.size > 0:
            min_y, min_x = mask_indices.min(axis=0)
            max_y, max_x = mask_indices.max(axis=0)
            center_x = (min_x + max_x) // 2
            center_y = (min_y + max_y) // 2
            text_position = (center_x - text_width // 2, center_y - text_height // 2)
            text_infos.append((text, text_position))

    draw = ImageDraw.Draw(overlay)
    for text, text_position in text_infos:
        draw.text(text_position, text, fill=(255, 255, 255, 255), font=font)

    # Blend RGB image with mask overlay
    blended_image = Image.alpha_composite(image, overlay)
    som_image = blended_image.convert("RGB")
    som_image.save(save_path)
    return True


def som_overlay_and_save_show(image_path, annotation, save_path, visualization):
    mask_rles = annotation["rle"]

    if save_path.exists():
        return True

    image = Image.open(image_path).convert("RGBA")
    # Resize image to target size
    image = image.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

    masks_np = [
        cv2.resize(
            mask_utils.decode(mask), TARGET_SIZE, interpolation=cv2.INTER_NEAREST
        )
        for mask in mask_rles
    ]

    visualizer = visualization.Visualizer(np.array(image))
    for i, mask_np in enumerate(masks_np):
        demo = visualizer.draw_binary_mask_with_number(
            mask_np > 0,
            color=None,
            text=str(i),
            text_color=None,
            text_bg_color=None,
            label_mode="1",
            alpha=0.2,
            anno_mode=["Mask", "Mark", "Contour"],
        )
    som_image = Image.fromarray(demo.get_image())
    som_image.save(save_path)
    return True


def som_overlay(
    image_path,
    annotation,
    save_path,
    method="show",
    font_path=None,
    visualization=None,
    font_size=None,
):
    func_map = {
        "iccv_show": som_overlay_and_save_iccv_show,
        "show": som_overlay_and_save_show,
    }
    if method == "iccv_show":
        return func_map[method](image_path, annotation, save_path, font_path, font_size)
    else:
        return func_map[method](image_path, annotation, save_path, visualization)


def load_visualization_module(visualization_tool_package):
    """Load the visualization module dynamically"""
    spec = importlib.util.spec_from_file_location(
        "visualization", visualization_tool_package
    )
    visualization = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(visualization)
    return visualization


def process_annotation_list(
    local_chunk_id,
    annotation_list,
    args,
    annotation_media_dir,
    visualization_tool_package,
    visualization=None,
):
    """Process a list of annotations and generate SOM overlays"""
    # Load visualization module within the worker process if not provided
    if visualization is None:
        visualization = load_visualization_module(visualization_tool_package)

    processed_count = 0
    for i, annotation in enumerate(tqdm(annotation_list)):
        if i % 100 == 0:
            print(f"[chunk {local_chunk_id}] Processing {i}/{len(annotation_list)}")

        # Debug: Print annotation info
        if args.debug:
            print(f"Processing annotation ID: {annotation.get('id', 'unknown')}")
            print(f"Image field: {annotation.get('image', 'missing')}")

        # Check if image field exists and is valid
        if "image" not in annotation:
            print(
                f"Warning: No 'image' field in annotation {annotation.get('id', 'unknown')}"
            )
            continue

        image_filename = annotation["image"]

        # Check if the image filename looks like a hash (indicating a blob path)
        if len(image_filename) > 50 and not image_filename.endswith(
            (".png", ".jpg", ".jpeg")
        ):
            print(f"Warning: Image field appears to be a blob path: {image_filename}")
            print(f"Annotation keys: {list(annotation.keys())}")
            continue

        image_path = annotation_media_dir / image_filename

        # Check if the image file actually exists
        if not image_path.exists():
            print(f"Warning: Image file does not exist: {image_path}")
            # Check if it's a broken symlink
            if image_path.is_symlink():
                target = Path(os.readlink(image_path))
                print(f"  This is a broken symlink pointing to: {target}")
                print(f"  The target file exists: {target.exists()}")
                print("  This usually means Git LFS files weren't downloaded properly.")
                print(
                    "  Please run: git lfs pull or re-download the dataset with proper LFS support."
                )
            continue

        save_path = Path(
            str(image_path)
            .replace("images", "som_images")
            .replace(".png", f'.{annotation["id"]}.png')
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)

        success = som_overlay(
            image_path,
            annotation,
            save_path,
            method=args.som_method,
            font_path=args.font_path,
            visualization=visualization,
            font_size=args.font_size,
        )
        if success and save_path.exists():
            processed_count += 1

        gc.collect()

    print(
        f"[chunk {local_chunk_id}] Successfully processed {processed_count}/{len(annotation_list)} images"
    )
    return processed_count


def main():
    parser = argparse.ArgumentParser(
        description="Generate SOM overlays for spatial VQA dataset"
    )
    parser.add_argument(
        "--original_data_root_dir",
        type=str,
        default=".",
        help="Root directory containing the original dataset",
    )
    parser.add_argument(
        "--save_root_dir",
        type=str,
        default="som/som_all",
        help="Root directory to save processed SOM images",
    )
    parser.add_argument(
        "--annotation_names",
        nargs="+",
        default=["train"],
        help="List of annotation names to process (e.g., train val test)",
    )
    parser.add_argument(
        "--visualization_tool_package",
        type=str,
        default="toolbox/visualization.py",
        help="Path to the visualization tool package",
    )
    parser.add_argument(
        "--font_path",
        type=str,
        default=None,
        help="Path to the font file for text rendering (defaults to DejaVuSans-Bold.ttf in the same directory as this script)",
    )
    parser.add_argument(
        "--font_size",
        type=int,
        default=None,
        help="Font size for text rendering (defaults to FONT_SIZE environment variable or 30)",
    )
    parser.add_argument(
        "--som_method",
        type=str,
        default="show",
        choices=["iccv_show", "show"],
        help="SOM overlay method to use",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: process only 1 sample sequentially without parallelization",
    )

    args = parser.parse_args()

    # Set default font path if not provided
    if args.font_path is None:
        script_dir = Path(__file__).parent
        args.font_path = script_dir / "DejaVuSans-Bold.ttf"

    # Load visualization module
    visualization = load_visualization_module(args.visualization_tool_package)

    # Make save directory
    os.makedirs(args.save_root_dir, exist_ok=True)

    for annotation_name in args.annotation_names:
        annotations_file = f"{args.original_data_root_dir}/{annotation_name}.json"
        annotation_media_dir = (
            Path(args.original_data_root_dir) / annotation_name / "images"
        )
        annotations_list = get_annotations(annotations_file)

        # Debug: Print first annotation to understand structure
        if annotations_list:
            print(f"First annotation keys: {list(annotations_list[0].keys())}")
            print(
                f"First annotation image field: {annotations_list[0].get('image', 'missing')}"
            )
            print(f"Annotation media dir: {annotation_media_dir}")
            print(f"Annotation media dir exists: {annotation_media_dir.exists()}")

        if args.debug:
            # Debug mode: process only 1 sample sequentially
            print("Debug mode: processing only 1 sample sequentially")
            annotations_list = annotations_list[:1]  # Take only the first sample
            annotations_nested_list = [annotations_list]  # Single chunk
        else:
            annotations_nested_list = split_list_to_nested_list(annotations_list)
        gc.collect()

        total_processed = 0
        if args.debug:
            # Sequential processing for debug mode
            for local_chunk_id, annotation_list in enumerate(annotations_nested_list):
                print(
                    f"Processing chunk {local_chunk_id} with {len(annotation_list)} annotations"
                )
                processed_count = process_annotation_list(
                    local_chunk_id=local_chunk_id,
                    annotation_list=annotation_list,
                    args=args,
                    annotation_media_dir=annotation_media_dir,
                    visualization_tool_package=args.visualization_tool_package,
                    visualization=visualization,
                )
                total_processed += processed_count
        else:
            # Parallel processing (original logic)
            with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
                futures = {
                    executor.submit(
                        process_annotation_list,
                        local_chunk_id=local_chunk_id,
                        annotation_list=annotation_list,
                        args=args,
                        annotation_media_dir=annotation_media_dir,
                        visualization_tool_package=args.visualization_tool_package,
                    ): (local_chunk_id, annotation_list)
                    for local_chunk_id, annotation_list in enumerate(
                        annotations_nested_list
                    )
                }
                for future in futures:
                    try:
                        processed_count = future.result()
                        total_processed += processed_count
                    except Exception as e:
                        import traceback

                        print(f"Error processing chunk: {e}")
                        traceback.print_exc()

        print(f"Total processed images for {annotation_name}: {total_processed}")


if __name__ == "__main__":
    main()
