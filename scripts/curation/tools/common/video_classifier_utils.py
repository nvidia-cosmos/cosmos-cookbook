import json
import logging as logger
import os
from typing import Dict, List, Optional

from openai import OpenAI

from scripts.curation.tools.common.mediaprocessor import MediaProcessor

# Default inference endpoints
DEFAULT_ENDPOINTS = {
    "qwen7b": "http://10.31.92.100:8006/v1",
    "reason": "http://10.31.92.100:8007/v1",
}

# Default model names
DEFAULT_MODELS = {
    "qwen7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "reason": "/config/models/qwen2p5_vl_7B_instruct",
}


def find_video_files(directory: str) -> List[str]:
    """Find all .mp4 files in directory and subdirectories."""
    video_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".mp4"):
                video_files.append(os.path.join(root, file))
    return video_files


def get_openai_client(
    model_name: str, inference_endpoint: Optional[str] = None
) -> OpenAI:
    """Get OpenAI client with appropriate endpoint."""
    url = (
        inference_endpoint
        if inference_endpoint
        else DEFAULT_ENDPOINTS[model_name.lower()]
    )
    return OpenAI(base_url=url, api_key=os.getenv("NVCF_API_KEY", ""))


def process_video(
    video_path: str,
    prompt: str,
    model_name: str = "reason",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    stream: bool = False,
    inference_endpoint: Optional[str] = None,
) -> Optional[str]:
    """
    Process a single video and return the classification result.

    Args:
        video_path: Path to the video file
        prompt: The prompt for the model
        model_name: Model to use (qwen7b or reason)
        temperature: Temperature for sampling
        max_tokens: Maximum tokens to generate
        stream: Whether to stream the response
        inference_endpoint: Optional custom inference endpoint
    """
    client = get_openai_client(model_name, inference_endpoint)
    model = DEFAULT_MODELS[model_name.lower()]

    try:
        processor = MediaProcessor("tmp")
        # Verify video exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Process video
        media_data = processor.process_video(video_path)
        if not media_data:
            raise RuntimeError("Failed to process video")

        media_content = {"type": "video_url", "video_url": {"url": media_data}}
        instruction = (
            "\nPlease answer the question in the following format: "
            "<think> your reasoning </think> "
            "<answer> your answer </answer>."
        )
        messages = [{"role": "user", "content": [media_content, {"type": "text", "text": prompt + instruction}]}]  # type: ignore

        completion = client.chat.completions.create(  # type: ignore
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=1,
            max_tokens=max_tokens,
            stream=stream,
        )
        return completion.choices[0].message.content
    except Exception as e:  # noqa
        logger.error(f"Error processing video {video_path}: {str(e)}")
        return None


def extract_answer(result: str) -> str:
    """Extract answer from model response."""
    try:
        return result.split("<answer>")[1].split("</answer>")[0].strip().lower()
    except Exception as e:  # noqa
        logger.error(f"Could not parse result: {e}")
        return "error"


def process_directory(
    input_dir: str,
    prompt: str,
    model_name: str = "reason",
    inference_endpoint: Optional[str] = None,
    output_filename: str = "classification_results.json",
) -> Dict[str, Dict[str, str]]:
    """
    Process all videos in directory and save results to output file.

    Args:
        input_dir: Directory containing videos to process
        prompt: The prompt for the model
        model_name: Model to use (qwen7b or reason)
        inference_endpoint: Optional custom inference endpoint
        output_filename: Name of the output JSON file
    """
    video_files = find_video_files(input_dir)
    logger.info(f"Found {len(video_files)} video files to process")
    logger.info(
        f"Using inference endpoint: "
        f"{inference_endpoint if inference_endpoint else DEFAULT_ENDPOINTS[model_name.lower()]}"
    )
    results = {}
    for video_path in video_files:
        logger.info(f"\nProcessing: {video_path}")
        result = process_video(
            video_path,
            prompt,
            model_name=model_name,
            inference_endpoint=inference_endpoint,
        )
        rel_path = os.path.relpath(video_path, input_dir)

        if result:
            answer = extract_answer(result)
            results[rel_path] = {"category": answer, "full_response": result}
            logger.info(f"Classified as {answer.upper()}: {rel_path}")
        else:
            results[rel_path] = {"category": "error", "full_response": ""}
            logger.error(f"Could not process {rel_path}")

    # Write results to JSON file
    output_file = os.path.join(input_dir, output_filename)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"\nResults saved to {output_file}")
    return results
