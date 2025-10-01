import argparse
import logging as logger

from scripts.curation.tools.common.video_classifier_utils import process_directory

CONTENT_QUALITY_PROMPT = """
You are an expert in video content quality analysis.
Your task is to classify the input video as either GOOD or BAD content
based on the following definitions:

GOOD content (ALL conditions must be met):
- Must be a SINGLE, continuous real-life traffic recording from ONE camera angle only
(no split-screens, multi-views, or transitions)
- Must show clear, high-quality footage of actual vehicles and road scenes (no static screens, logos, or solid colors)
- Must be unedited direct camera footage (no overlays, effects, text, or processed content)
- Must be properly exposed and focused, with easily recognizable traffic-related content
- Must show continuous movement (not frozen frames or still images)

BAD content (classify as BAD if ANY of these conditions are met):
- ANY multi-view or view-switching content:
  * Split-screen layouts (even if only for a portion of the video)
  * Multiple camera angles shown simultaneously
  * Camera view transitions or switches during the video
  * Security camera arrays or grid layouts
  * Picture-in-picture displays
  * Any combination of different camera perspectives
- ANY non-camera footage:
  * Static screens, logos, warnings
  * Text overlays, transitions
  * Artificial or computer-generated content
- ANY poor quality footage:
  * Blurry, dark, noisy content
  * Blocked or unrecognizable scenes
- ANY edited content:
  * Effects, transitions, cuts
  * Processed or modified footage
- ANY non-traffic content:
  * Empty frames, solid backgrounds
  * Unrelated scenes or content

CRITICAL INSTRUCTIONS:
1. FIRST describe what you see in the video in detail
2. Then check for split-screens or multiple views - these are ALWAYS BAD
3. Check for any camera transitions during the video - these are ALWAYS BAD
4. Check for non-traffic content (solid colors, logos, static screens)
5. Check video quality and editing
6. ONLY classify as GOOD if it's a single, continuous camera view of actual traffic

Please analyze the video and provide your answer in the following format:
<think>
[FIRST: Detailed description of what you see in the video]
[SECOND: Analysis of camera views - are there multiple views or transitions?]
[THIRD: Analysis of content type - is it real traffic footage?]
[FOURTH: Analysis of quality and editing]
[FINALLY: Your conclusion based on the above analysis]
</think>
<answer>
[good or bad]
</answer>
"""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify videos as good or bad content."
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Input directory containing videos"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="reason",
        help="Model to use for inference (default: reason)",
    )
    parser.add_argument(
        "--inference_endpoint",
        type=str,
        help="Custom inference endpoint URL (e.g. http://localhost:8008/v1)",
    )
    return parser.parse_args()


def main() -> None:
    """Main function to process videos for content quality classification."""
    args = parse_arguments()
    try:
        process_directory(
            input_dir=args.input_dir,
            prompt=CONTENT_QUALITY_PROMPT,
            model_name=args.model_name,
            inference_endpoint=args.inference_endpoint,
            output_filename="bad_content_classification_results.json",
        )
        logger.info("\nProcessing complete!")
    except Exception as e:  # noqa
        logger.error(f"Error: {e}")


if __name__ == "__main__":
    main()
