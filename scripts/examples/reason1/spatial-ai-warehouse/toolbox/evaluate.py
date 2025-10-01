# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Simplified evaluation script for video language models.
Follows CustomDataset logic without response for evaluation.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Any

from tqdm import tqdm

# Import required modules with fallbacks
try:
    import qwen_vl_utils
    import transformers
    import torch
    from cosmos_reason1_utils.text import create_conversation
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please install required dependencies:")
    print("pip install transformers qwen-vl-utils torch cosmos-reason1-utils")
    exit(1)

# Configuration macros
DEFAULT_FPS = 1
DEFAULT_MAX_PIXELS = 655360
DEFAULT_RESIZED_WIDTH = 960
DEFAULT_RESIZED_HEIGHT = 540
DEFAULT_MAX_TOKENS = 1024


def load_evaluation_data(annotation_path: str, media_path: str = "", limit: int = -1) -> List[Dict[str, Any]]:
    """Load evaluation data following CustomDataset logic."""
    print(f"Loading evaluation data from {annotation_path}...")

    with open(annotation_path, 'r') as f:
        data = json.load(f)

    if limit > 0:
        data = data[:limit]

    # Process data following CustomDataset.__getitem__ logic
    processed_data = []
    for idx, sample in enumerate(data):
        user_prompt = sample["conversations"][0]["value"]
        response = sample["conversations"][1]["value"]
        images = sample.get("image", None) or sample.get("images", None)
        if images and isinstance(images, str):
            images = [images]
        videos = sample.get("video", None)
        if videos and isinstance(videos, str):
            videos = [videos]

        # If media_path is not empty, join it with each image/video path
        if media_path != "":
            if images:
                images = [os.path.join(media_path, img) for img in images]
            if videos:
                videos = [os.path.join(media_path, vid) for vid in videos]

        # Remove image and video tags from user prompt
        user_prompt = re.sub(r"(\n)?</?(image|video)>(\n)?", "", user_prompt)

        processed_sample = {
            "id": sample.get("id", f"sample_{idx}"),
            "user_prompt": user_prompt,
            "response": response,
            "images": images,
            "videos": videos
        }
        processed_data.append(processed_sample)

    print(f"Loaded {len(processed_data)} evaluation tasks")
    return processed_data


def load_model_and_processor(model_name: str):
    """Load model and processor using the simple pattern from inference_sample.py."""
    print(f"Loading model from {model_name}...")

    # Use transformers directly like in inference_sample.py
    model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    processor = transformers.AutoProcessor.from_pretrained(model_name)

    return model, processor




def process_single_task(
    entry: Dict[str, Any],
    model,
    processor,
    task_id: int,
    vision_kwargs: Dict[str, Any],
    max_tokens: int = DEFAULT_MAX_TOKENS
) -> Dict[str, Any]:
    """Process a single evaluation task following CustomDataset logic without response."""
    annotation_id = entry.get('id', f'entry_{task_id}')
    correct_answer = entry.get('response', '')

    # Create conversation without response (for evaluation)
    conversation = create_conversation(
        user_prompt=entry.get('user_prompt', ''),
        response="",  # No response for evaluation
        images=entry.get('images', []),
        videos=entry.get('videos', []),
        vision_kwargs=vision_kwargs,
    )

    # Process inputs
    text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = qwen_vl_utils.process_vision_info(conversation)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    # Parse answer - handle different types of answers (letters, words, numbers)
    predicted_answer = ""

    # First, try to extract a single uppercase letter (A-Z) for letter-based answers
    SINGLE_LETTER_PATTERN = re.compile(r"[A-Z]")
    letter_match = SINGLE_LETTER_PATTERN.search(output_text)
    if letter_match:
        predicted_answer = letter_match.group(0)
    else:
        # If no single letter found, try to extract the first meaningful word/number
        # Remove common prefixes and extract the first word/number
        cleaned_text = output_text.strip()
        # Remove common prefixes like "The answer is", "Answer:", etc.
        cleaned_text = re.sub(r"^(the\s+)?answer\s*(is)?\s*:?\s*", "", cleaned_text, flags=re.IGNORECASE)
        # Extract first word or number
        first_word_match = re.search(r'\b(\w+)\b', cleaned_text)
        if first_word_match:
            predicted_answer = first_word_match.group(1)
        else:
            # Fallback: take the first non-whitespace token
            tokens = cleaned_text.split()
            if tokens:
                predicted_answer = tokens[0]

    # Check if correct - handle both string and numeric correct answers
    correct_answer_str = str(correct_answer).lower().strip()
    predicted_answer_str = predicted_answer.lower().strip()
    is_correct = predicted_answer_str == correct_answer_str

    return {
        "datasource": "warehouse",
        "video_id": annotation_id,
        "prompt": None,
        "correct_answer": correct_answer,
        "reasoning": "",
        "answer": predicted_answer,
        "full_response": output_text,
        "is_correct": is_correct
    }


def save_single_result(result: Dict[str, Any], output_json_fname: str):
    """Save a single result to its own JSON file."""
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_json_fname)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save as list with single item (matching original format)
    with open(output_json_fname, 'w') as f:
        json.dump([result], f, indent=4)


def save_results(results: List[Dict[str, Any]], output_dir: str):
    """Save evaluation results - one file per entry."""
    print(f"Saving {len(results)} individual result files...")

    # Calculate accuracy
    correct_count = sum(1 for r in results if r["is_correct"])
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0.0

    # Save each result to its own file
    for result in results:
        video_id = result["video_id"]
        output_json_fname = os.path.join(output_dir, f"{video_id}.json")
        save_single_result(result, output_json_fname)

    print(f"Results saved to {output_dir}")
    print(f"Accuracy: {accuracy:.3f} ({correct_count}/{total_count})")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Simplified video language model evaluation")
    parser.add_argument("--annotation_path", type=str, required=True, help="Path to annotation JSON file")
    parser.add_argument("--media_path", type=str, default="", help="Path to media files directory")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
    parser.add_argument("--results_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--limit", type=int, default=-1, help="Limit number of tasks to evaluate")

    args = parser.parse_args()

    # Build vision kwargs from default macros
    vision_kwargs = {
        "fps": DEFAULT_FPS,
        "max_pixels": DEFAULT_MAX_PIXELS, # Note: not used if resized_width and resized_height are provided
        "resized_height": DEFAULT_RESIZED_HEIGHT,
        "resized_width": DEFAULT_RESIZED_WIDTH,
    }

    # Set up results directory structure
    results_output_dir = os.path.join(
        args.results_dir,
        os.path.basename(args.model_name.rstrip("/")),
        "naive"  # Default answer type
    )

    print(f"Annotation path: {args.annotation_path}")
    print(f"Media path: {args.media_path}")
    print(f"Model: {args.model_name}")
    print(f"Results dir: {results_output_dir}")
    print("Answer type: naive")
    print(f"Vision settings: {vision_kwargs}")
    print(f"Max tokens: {DEFAULT_MAX_TOKENS}")
    print(f"Limit: {args.limit}")
    print("-" * 50)

    # Load model
    start_time = time.time()
    model, processor = load_model_and_processor(args.model_name)
    print(f"Model loaded in {time.time() - start_time:.1f}s")

    # Load evaluation data
    start_time = time.time()
    evaluation_data = load_evaluation_data(args.annotation_path, args.media_path, args.limit)
    print(f"Data loaded in {time.time() - start_time:.1f}s")

    # Process tasks
    print(f"\nProcessing {len(evaluation_data)} tasks...")
    results = []

    for i, entry in enumerate(tqdm(evaluation_data, desc="Evaluating")):
        result = process_single_task(entry, model, processor, i, vision_kwargs, DEFAULT_MAX_TOKENS)
        results.append(result)

    # Save results
    start_time = time.time()
    save_results(results, results_output_dir)
    print(f"Results saved in {time.time() - start_time:.1f}s")

    print("\nEvaluation completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
