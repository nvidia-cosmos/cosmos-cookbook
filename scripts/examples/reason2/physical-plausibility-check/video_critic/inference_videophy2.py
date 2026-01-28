#!/usr/bin/env -S uv run --script
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

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "cosmos-reason2-utils[inference]",
#   "datasets",
#   "vllm",
# ]
# [tool.uv.sources]
# cosmos-reason2-utils = { path = "../../cosmos_reason2_utils", editable = true }
# ///

"""Run inference on videophy2 dataset using offline mode.

This script processes video datasets by loading metadata from HuggingFace.
It runs inference on videos using URLs directly (no downloading) and saves
results as JSON files.

Example usage:
    uv run examples/video_critic/inference_videophy2.py --model nvidia/Cosmos-Reason2-2B
"""

from cosmos_reason2_utils.init import init_script

init_script()

import argparse
import json
import os
import re
import traceback
from pathlib import Path

import datasets
import qwen_vl_utils
import transformers
import vllm
import yaml

from cosmos_reason2_utils.text import create_conversation, SYSTEM_PROMPT
from cosmos_reason2_utils.script.inference import Offline
from cosmos_reason2_utils.vision import PIXELS_PER_TOKEN, VisionConfig

ROOT = Path(__file__).resolve().parent.parent.parent


def get_video_data(dataset_name: str, split: str = "train"):
    """Load dataset and return video URLs with ground truth scores."""
    print(f"Loading dataset: {dataset_name}, split: {split}")
    dataset = datasets.load_dataset(dataset_name)
    dataset_split = dataset[split]

    print(f"Dataset loaded successfully. {split} split has {len(dataset_split)} examples.")

    video_data = []
    for example in dataset_split:
        video_data.append(
            {
                "video_url": example["video_url"],
                "ground_truth": example["pc"],
            }
        )

    if video_data:
        print(f"Sample data: {video_data[0]}")
    return video_data


def parse_answer_from_text(text: str) -> float | None:
    """Parse a numeric answer from model output text.
    
    The prompt expects a score between 1 and 5. The model outputs:
    - A number on its own line: "3" or "4"
    - Sometimes with template text on previous line: "[Score between 1 and 5.]\n\n3"
    - Sometimes followed by explanation: "3\n\nOkay, let's see..."
    
    This function looks for numbers (1-5) that appear on their own line.
    """
    # Split text into lines
    lines = text.strip().split('\n')
    
    # Look for a number (1-5) that appears on its own line
    for line in lines:
        line = line.strip()
        # Match a single integer between 1-5 on its own line
        match = re.match(r'^([1-5])\.?\s*$', line)
        if match:
            try:
                value = float(match.group(1))
                return value
            except ValueError:
                continue
    
    return None


def load_prompt_config(prompt_path: str) -> tuple[str, str]:
    """Load prompt configuration from YAML file."""
    if not os.path.isabs(prompt_path):
        prompt_path = os.path.join(ROOT, prompt_path)
    
    with open(prompt_path, 'r') as f:
        config = yaml.safe_load(f)
    
    system_prompt = config.get('system_prompt', SYSTEM_PROMPT)
    user_prompt = config.get('user_prompt', '')
    
    if not user_prompt:
        raise ValueError(f"No user_prompt found in {prompt_path}")
    
    return system_prompt, user_prompt


def run_inference_for_video(
    llm: vllm.LLM,
    processor: transformers.Qwen3VLProcessor,
    video_url: str,
    system_prompt: str,
    user_prompt: str,
    vision_kwargs: dict | None,
    sampling_params: vllm.SamplingParams,
) -> str:
    """Run inference for a single video.
    
    This follows the same pattern as offline_inference but reuses the provided model.
    """
    # Create conversation
    conversation = create_conversation(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        videos=[video_url],
        vision_kwargs=vision_kwargs,
    )
    
    # Process inputs (matching offline_inference pattern)
    # add_vision_ids is True when there are multiple media items (images + videos > 1)
    # In our case, we have 1 video, so add_vision_ids = False
    add_vision_ids = False
    prompt = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        add_vision_ids=add_vision_ids,
    )
    
    image_inputs, video_inputs, video_kwargs = qwen_vl_utils.process_vision_info(
        conversation,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    
    # Run inference (matching offline_inference pattern)
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    
    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }
    
    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    
    # Extract output text
    output_text = outputs[0].outputs[0].text.strip()
    return output_text


def run_inference_for_dataset(args):
    """Run inference on videos for a dataset."""
    # Load video data
    print(f"Loading videos from HuggingFace dataset: {args.dataset}")
    video_data = get_video_data(args.dataset, args.split)

    print(f"\nFound {len(video_data)} videos to process")

    if not video_data:
        print("❌ No videos to process!")
        return

    # Use provided output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nUsing output directory: {output_dir}")

    # Load prompt configuration
    prompt_path = args.input_file
    system_prompt, user_prompt = load_prompt_config(prompt_path)

    # Create Offline args with defaults (will be used for vision_kwargs and sampling_params)
    offline_args = Offline(
        model=args.model,
        revision=args.revision,
        input_file=args.input_file,
        videos=[],  # Will be set per video
        images=[],
    )

    # Calculate and validate vision kwargs (matching offline_inference pattern)
    vision_kwargs = offline_args.vision.model_dump(exclude_none=True)
    assert offline_args.sampling_params.max_tokens
    if offline_args.max_model_len < offline_args.sampling_params.max_tokens:
        raise ValueError("Max model length must be greater than max tokens.")
    max_seq_len = offline_args.max_model_len - offline_args.sampling_params.max_tokens
    total_pixels = int(max_seq_len * PIXELS_PER_TOKEN * 0.9)
    if "total_pixels" in vision_kwargs:
        if vision_kwargs["total_pixels"] > total_pixels:
            raise ValueError(
                f"Total pixels {vision_kwargs['total_pixels']} exceeds limit {total_pixels}."
            )
    else:
        vision_kwargs["total_pixels"] = total_pixels
    VisionConfig.model_validate(vision_kwargs)

    # Initialize model and processor once (reused across all videos)
    print(f"\nInitializing vLLM model: {offline_args.model}")
    llm = vllm.LLM(
        model=offline_args.model,
        revision=offline_args.revision,
        max_model_len=offline_args.max_model_len,
        limit_mm_per_prompt={"video": 1},
        enforce_eager=True,
    )
    print("✓ Model loaded successfully")

    print("Loading processor...")
    processor: transformers.Qwen3VLProcessor = (
        transformers.AutoProcessor.from_pretrained(offline_args.model)
    )
    print("✓ Processor loaded successfully")

    # Create sampling params (matching offline_inference pattern)
    # Override seed to 1 for reproducibility
    sampling_kwargs = dict(offline_args.sampling_kwargs)
    sampling_kwargs["seed"] = 1
    sampling_params = vllm.SamplingParams(**sampling_kwargs)

    # Process each video
    for i, video_item in enumerate(video_data, 1):
        video_url = video_item["video_url"]
        ground_truth = video_item["ground_truth"]

        json_path = os.path.join(output_dir, f"{i}.json")

        if os.path.exists(json_path):
            print(f"\n[{i}/{len(video_data)}] 📋 Results already exist: {os.path.basename(json_path)}. Skipping...")
            continue

        print(f"\n[{i}/{len(video_data)}] Processing: {video_url}")

        try:
            # Run inference (reusing the same model)
            output_text = run_inference_for_video(
                llm=llm,
                processor=processor,
                video_url=video_url,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                vision_kwargs=vision_kwargs,
                sampling_params=sampling_params,
            )

            # Parse answer
            score = parse_answer_from_text(output_text)

            # Save results to JSON
            result_entry = {
                "video_url": video_url,
                "ground_truth": ground_truth,
                "output_text": output_text,
                "pred_score": score,
            }

            with open(json_path, "w") as f:
                json.dump(result_entry, f, indent=2)

            if score is not None:
                print(f"✅ Saved results (score: {score}) to {os.path.basename(json_path)}")
            else:
                print(f"✅ Saved results to {os.path.basename(json_path)}")
                print(f"   Output: {output_text[:200]}...")

        except Exception as e:
            print(f"❌ Error processing video: {str(e)}")
            traceback.print_exc()

    print(f"\n✅ Batch processing completed. Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="videophysics/videophy2_test",
                        help='Dataset name (default: "videophysics/videophy2_test")')
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split (default: train)")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="nvidia/Cosmos-Reason2-2B",
                        help="Model name or path (default: nvidia/Cosmos-Reason2-2B)")
    parser.add_argument("--revision", type=str, default=None,
                        help="Model revision")
    
    # Prompt arguments
    parser.add_argument("--input-file", type=str, default="prompts/video_reward.yaml",
                        help="Path to input yaml file")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="outputs/videophy2_test",
                        help="Output directory for JSON results")
    
    args = parser.parse_args()
    run_inference_for_dataset(args)


if __name__ == "__main__":
    main()
