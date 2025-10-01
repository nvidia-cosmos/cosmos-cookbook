import argparse
import logging as logger

from data_curation.tools.common.video_classifier_utils import process_directory

CAMERA_TYPE_PROMPT = """
You are an expert in video analysis and camera classification.
Your task is to analyze the input video in a concise manner and
classify it into one of these three categories:

1) Static CCTV-style camera:
   - Fixed position with no movement
   - Typically mounted high on walls or ceilings
   - Wide-angle view covering a large area
   - Often used for surveillance in buildings, streets, or public spaces
   - No camera motion, only subjects move within the frame

2) Static dash-camera (parked):
   - Mounted on a stationary vehicle
   - Fixed position but at street/road level
   - Typically shows traffic intersection or road view
   - May capture passing vehicles and pedestrians
   - No camera motion, but perspective is from vehicle height

3) Moving dash-camera (in-motion):
   - Mounted on a moving vehicle
   - Shows forward motion and changing scenery
   - Captures road ahead with moving perspective
   - May show vehicle dashboard or hood
   - Continuous camera motion matching vehicle movement

Please analyze the video carefully and:
1. Consider the camera's position, movement, and perspective
2. Note any visible mounting points or vehicle parts
3. Observe the motion characteristics
4. Consider the typical use case and context

Provide your analysis in the following format:
<think>
[Your detailed reasoning about why the video belongs to a specific category]
[Key observations about camera position, movement, and context]
[Comparison with other categories and why they don't fit]
</think>

<answer>
[The category number: 1, 2, or 3]
</answer>
"""


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process videos for camera type classification."
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Input directory containing videos"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="camera_classification_results.json",
        help="Output JSON file name (default: camera_classification_results.json)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["reason", "qwen7b"],
        default="reason",
        help="Model to use for inference (default: reason)",
    )
    return parser.parse_args()


def main() -> None:
    """Main function to process videos for camera type classification."""
    args = parse_arguments()
    try:
        process_directory(
            input_dir=args.input_dir,
            prompt=CAMERA_TYPE_PROMPT,
            model_name=args.model_name,
            output_filename=args.output_file,
        )
        logger.info("\nProcessing complete!")
    except Exception as e:  # noqa
        logger.error(f"Failed to process directory: {str(e)}")


if __name__ == "__main__":
    main()
