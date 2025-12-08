#!/usr/bin/env -S uv run --script
#
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

#
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "accelerate",
#   "cosmos-reason1-utils",
#   "pydantic",
#   "pyyaml",
#   "qwen-vl-utils",
#   "torch",
#   "torchcodec",
#   "torchvision",
#   "transformers>=4.51.3",
#   "vllm",
# ]
# [tool.uv]
# exclude-newer = "2025-07-31T00:00:00Z"
# [tool.uv.sources]
# cosmos-reason1-utils = { path = "../../cosmos_reason1_utils", editable = true }
# ///

"""Example script for using Cosmos Reason 1 as a video reward model.

Example:

```shell
# Print results to terminal only
./examples/video_critic/video_reward.py --video_path assets/sample.mp4

# Print results to terminal and save as both HTML and JSON (can enable one or both)
./examples/video_critic/video_reward.py --video_path assets/sample.mp4 --output_html --output_json
```
"""
# ruff: noqa: E402

from cosmos_reason1_utils.script import init_script

init_script()

import argparse
import base64
import json
import os
import pathlib
import re
import xml.etree.ElementTree as ET

import pydantic
import yaml
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

ROOT = pathlib.Path(__file__).parents[2].resolve()


class Prompt(pydantic.BaseModel):
    """Config for prompt."""

    model_config = pydantic.ConfigDict(extra="forbid")

    system_prompt: str = pydantic.Field(default="", description="System prompt")
    user_prompt: str = pydantic.Field(default="", description="User prompt")


def parse_response(response):
    """Parse response to extract integer from <answer></answer> tags.

    Returns:
        dict with 'answer' (int or None) and 'raw' (str) keys, or None if parsing failed
    """
    try:
        # Try XML parsing first
        wrapped = f"<root>{response.strip()}</root>"
        root = ET.fromstring(wrapped)
        answer_element = root.find("answer")

        if answer_element is not None and answer_element.text:
            answer_text = answer_element.text.strip()
            try:
                answer_int = int(answer_text)
                return {"answer": answer_int, "raw": response}
            except ValueError:
                # If not a valid integer, return None
                return {"answer": None, "raw": response}

        return {"answer": None, "raw": response}
    except Exception:
        # If XML parsing fails, try regex as fallback
        match = re.search(r"<answer>\s*(\d+)\s*</answer>", response)
        if match:
            try:
                answer_int = int(match.group(1))
                return {"answer": answer_int, "raw": response}
            except ValueError:
                pass
        return None


def video_to_base64(video_path):
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")


def build_html_report(video_path, responses):
    # Convert video to base64
    video_base64 = video_to_base64(video_path)
    mime_type = "video/mp4"

    # Parse responses
    parsed_responses = [parse_response(response) for response in responses]
    valid_responses = [
        r for r in parsed_responses if r is not None and r.get("answer") is not None
    ]

    # Calculate statistics
    if valid_responses:
        answers = [r["answer"] for r in valid_responses]
        avg_score = sum(answers) / len(answers)
        min_score = min(answers)
        max_score = max(answers)
    else:
        avg_score = min_score = max_score = None

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Cosmos Reason 1 Video Reward Report - {os.path.basename(video_path)}</title>
    <style>
        body {{ font-family: sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }}
        video {{ width: 100%; max-width: 600px; }}
        .result {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; }}
        .raw-output {{ background-color: #f5f5f5; padding: 10px; margin: 10px 0; white-space: pre-wrap; font-family: monospace; font-size: 12px; }}
        .answer {{ background-color: #e3f2fd; padding: 10px; margin: 10px 0; font-weight: bold; font-size: 18px; }}
        .failed {{ background-color: #ffebee; color: #c62828; padding: 10px; margin: 10px 0; }}
        .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
        .stat {{ text-align: center; padding: 15px; border: 1px solid #ddd; flex: 1; }}
    </style>
</head>
<body>
    <h1>Cosmos Reason 1 Video Reward Report</h1>
    <p>File: {os.path.basename(video_path)}</p>

    <h2>Video</h2>
    <video controls>
        <source src="data:{mime_type};base64,{video_base64}" type="{mime_type}">
    </video>

    <h2>Summary</h2>
    <div class="stats">"""

    if avg_score is not None:
        html += f"""
        <div class="stat">
            <div style="font-size: 24px; font-weight: bold;">{avg_score:.2f}</div>
            <div>Average Score</div>
        </div>
        <div class="stat">
            <div style="font-size: 24px; font-weight: bold;">{min_score}</div>
            <div>Min Score</div>
        </div>
        <div class="stat">
            <div style="font-size: 24px; font-weight: bold;">{max_score}</div>
            <div>Max Score</div>
        </div>"""
    else:
        html += """
        <div class="stat failed">
            <div>No valid scores parsed</div>
        </div>"""

    html += f"""
        <div class="stat">
            <div style="font-size: 24px; font-weight: bold;">{len(valid_responses)}/{len(responses)}</div>
            <div>Valid Responses</div>
        </div>
    </div>

    <h2>Detailed Results ({len(responses)} trials)</h2>"""

    for i, (response, parsed) in enumerate(
        zip(responses, parsed_responses, strict=False), 1
    ):
        if parsed is None or parsed.get("answer") is None:
            html += f"""
    <div class="result">
        <h3>Trial {i}</h3>
        <div class="failed">Failed to parse answer</div>
        <h4>Raw Response:</h4>
        <div class="raw-output">{response}</div>
    </div>"""
        else:
            html += f"""
    <div class="result">
        <h3>Trial {i}</h3>
        <div class="answer">Score: {parsed["answer"]}</div>
        <h4>Raw Response:</h4>
        <div class="raw-output">{parsed["raw"]}</div>
    </div>"""

    html += """
</body>
</html>"""

    return html


def build_json_report(video_path, responses):
    """Build minimal JSON report from responses."""
    # Parse responses
    parsed_responses = [parse_response(response) for response in responses]

    # Extract scores (None if parsing failed)
    scores = [
        parsed["answer"] if parsed and parsed.get("answer") is not None else None
        for parsed in parsed_responses
    ]

    # Build minimal JSON structure
    report = {
        "video_path": video_path,
        "scores": scores,
        "raw_responses": responses,
    }

    return report


def run_reward_model(llm, args):
    prompt_path = f"{ROOT}/{args.prompt_path}"
    prompt_config = Prompt.model_validate(yaml.safe_load(open(prompt_path, "rb")))

    sampling_params = SamplingParams(
        n=args.num_trials,
        temperature=0.6,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.05,
        max_tokens=4096,
        seed=1,  # for reproducibility
        logprobs=10,  # Get more logprobs to ensure we capture YES/NO tokens
    )

    messages = [
        {"role": "system", "content": prompt_config.system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": args.video_path,
                    # Recommended settings for video critic:
                    "fps": 16,
                    "total_pixels": 8192 * 28 * 28,
                },
                {"type": "text", "text": prompt_config.user_prompt},
            ],
        },
    ]
    processor = AutoProcessor.from_pretrained(args.model)
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )

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
    generated_text = [output.text for output in outputs[0].outputs]

    return generated_text


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run video reward model inference and optionally save reports"
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to input video for reward model",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=1,
        help="Number of trials for each video",
    )
    parser.add_argument(
        "--model", type=str, default="nvidia/Cosmos-Reason1-7B", help="Model path"
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="prompts/video_reward.yaml",
        help="Path to prompt config file",
    )
    parser.add_argument(
        "--output_html",
        action="store_true",
        help="Save results as HTML file",
    )
    parser.add_argument(
        "--output_json",
        action="store_true",
        help="Save results as JSON file",
    )
    return parser.parse_args()


def print_terminal_results(video_path, responses):
    """Print minimal results to terminal (scores only)."""
    # Parse responses
    parsed_responses = [parse_response(response) for response in responses]

    # Extract scores
    scores = [
        parsed["answer"] if parsed and parsed.get("answer") is not None else None
        for parsed in parsed_responses
    ]

    print(f"\nScores: {scores}")


def main():
    args = parse_args()

    llm = LLM(
        model=args.model,
        limit_mm_per_prompt={"image": 0, "video": 1},
        enforce_eager=True,
    )

    generated_text = run_reward_model(llm, args)

    # Always print results to terminal
    print_terminal_results(args.video_path, generated_text)

    # Optionally save HTML
    if args.output_html:
        html_content = build_html_report(args.video_path, generated_text)
        html_path = os.path.splitext(args.video_path)[0] + ".html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"✓ Generated HTML report: {html_path}")

    # Optionally save JSON
    if args.output_json:
        json_report = build_json_report(args.video_path, generated_text)
        json_path = os.path.splitext(args.video_path)[0] + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_report, f, indent=2)
        print(f"✓ Generated JSON report: {json_path}")


if __name__ == "__main__":
    main()
