#!/usr/bin/env python3
# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""
Cosmos Reason 2 Prompt Tests

This script demonstrates various Cosmos Reason 2 capabilities through a series of
prompt tests covering vision-language tasks like image/video understanding,
temporal localization, robotics reasoning, and synthetic data critique.

Usage:
    python cosmos_reason2_tests.py --host <IP> --port <PORT> --api-key <KEY>

    # Using environment variables:
    export VLLM_ENDPOINT="IP:PORT"
    export VLLM_API_KEY="your-api-key"
    python cosmos_reason2_tests.py

For more details on prompting patterns, see the Cosmos Reason 2 Prompt Guide:
https://github.com/nvidia-cosmos/cosmos-cookbook/blob/main/docs/core_concepts/prompt_guide/reason_guide.md
"""

import argparse
import json
import os
import sys
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not found. Install it with: pip install openai")
    sys.exit(1)


# =============================================================================
# Test Definitions
# =============================================================================

TESTS = {
    "1_basic_image": {
        "name": "Basic Image Understanding",
        "description": "Simple image description without reasoning",
        "media_type": "image_url",
        "media_url": "https://assets.ngc.nvidia.com/products/api-catalog/cosmos-reason1-7b/critic_rejection_sampling.jpg",
        "prompt": "What is in this image? Describe the objects and their positions.",
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.8,
    },
    "2_basic_video": {
        "name": "Basic Video Understanding",
        "description": "Simple video description without reasoning",
        "media_type": "video_url",
        "media_url": "https://assets.ngc.nvidia.com/products/api-catalog/cosmos-reason1-7b/car_curb.mp4",
        "prompt": "Caption the video in detail.",
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.8,
        "fps": 4,
    },
    "3_temporal_localization": {
        "name": "Temporal Localization (Timestamps)",
        "description": "Video description with timestamps in mm:ss format",
        "media_type": "video_url",
        "media_url": "https://assets.ngc.nvidia.com/products/api-catalog/cosmos-reason1-7b/car_curb.mp4",
        "prompt": """Describe the video. Add timestamps in mm:ss format.

Answer the question using the following format:

<think>
Your reasoning.
</think>

Write your final answer immediately after the </think> tag and include the timestamps.""",
        "max_tokens": 2048,
        "temperature": 0.6,
        "top_p": 0.95,
        "fps": 4,
    },
    "4_temporal_json": {
        "name": "Temporal Localization (JSON Output)",
        "description": "Video events with timestamps in JSON format",
        "media_type": "video_url",
        "media_url": "https://assets.ngc.nvidia.com/products/api-catalog/cosmos-reason1-7b/car_curb.mp4",
        "prompt": """Describe the notable events in the provided video. Provide the result in json format with 'mm:ss.ff' format for time depiction for each event. Use keywords 'start', 'end' and 'caption' in the json output.

Answer the question using the following format:

<think>
Your reasoning.
</think>

Write your final answer immediately after the </think> tag and include the timestamps.""",
        "max_tokens": 2048,
        "temperature": 0.6,
        "top_p": 0.95,
        "fps": 4,
    },
    "5_robotics_next_action": {
        "name": "Robotics Next Action Prediction",
        "description": "Predict the next action for a robotic system",
        "media_type": "image_url",
        "media_url": "https://assets.ngc.nvidia.com/products/api-catalog/cosmos-reason1-7b/critic_rejection_sampling.jpg",
        "prompt": """What can be the next immediate action for a robot to organize these objects?

Answer the question using the following format:

<think>
Your reasoning.
</think>

Write your final answer immediately after the </think> tag.""",
        "max_tokens": 1024,
        "temperature": 0.6,
        "top_p": 0.95,
    },
    "6_2d_trajectory": {
        "name": "2D Trajectory Creation",
        "description": "Generate end-effector trajectory in pixel coordinates",
        "media_type": "image_url",
        "media_url": "https://assets.ngc.nvidia.com/products/api-catalog/cosmos-reason1-7b/critic_rejection_sampling.jpg",
        "prompt": """You are given the task "Move the left bottle to far right". Specify the 2D trajectory your end effector should follow in pixel space. Return the trajectory coordinates in JSON format like this: {"point_2d": [x, y], "label": "gripper trajectory"}.

Answer the question using the following format:

<think>
Your reasoning.
</think>

Write your final answer immediately after the </think> tag.""",
        "max_tokens": 2048,
        "temperature": 0.3,
        "top_p": 0.3,
    },
    "7_sdg_critic": {
        "name": "Synthetic Data Generation Critic",
        "description": "Evaluate video for physics adherence and anomalies",
        "media_type": "video_url",
        "media_url": "https://assets.ngc.nvidia.com/products/api-catalog/cosmos-reason1-7b/car_curb.mp4",
        "prompt": """Approve or reject this generated video for inclusion in a dataset for physical world model AI training. It must perfectly adhere to physics, object permanence, and have no anomalies. Any issue or concern causes rejection.

Answer the question using the following format:

<think>
Your reasoning.
</think>

Write your final answer immediately after the </think> tag. Answer with Approve or Reject only.""",
        "max_tokens": 2048,
        "temperature": 0.3,
        "top_p": 0.3,
        "fps": 4,
    },
    "8_2d_grounding": {
        "name": "2D Object Grounding (Bounding Box)",
        "description": "Locate objects and return bounding box coordinates",
        "media_type": "image_url",
        "media_url": "https://assets.ngc.nvidia.com/products/api-catalog/cosmos-reason1-7b/critic_rejection_sampling.jpg",
        "prompt": """Locate all objects in the image. Return the bounding boxes in JSON format with the following structure:
{
  "objects": [
    {"label": "object_name", "box_2d": [x_min, y_min, x_max, y_max]}
  ]
}

Answer the question using the following format:

<think>
Your reasoning.
</think>

Write your final answer immediately after the </think> tag.""",
        "max_tokens": 2048,
        "temperature": 0.3,
        "top_p": 0.3,
    },
}


# =============================================================================
# Helper Functions
# =============================================================================


def parse_args():
    """Parse command-line arguments with environment variable fallbacks."""
    parser = argparse.ArgumentParser(
        description="Run Cosmos Reason 2 prompt tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Direct arguments:
    python cosmos_reason2_tests.py --host 89.169.115.247 --port 8000 --api-key mykey

    # Using environment variables:
    export VLLM_ENDPOINT="89.169.115.247:8000"
    export VLLM_API_KEY="mykey"
    python cosmos_reason2_tests.py

    # Run specific tests:
    python cosmos_reason2_tests.py --tests 1_basic_image 3_temporal_localization

    # List available tests:
    python cosmos_reason2_tests.py --list
        """,
    )

    # Connection settings
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("VLLM_HOST"),
        help="Host IP address (or set VLLM_HOST env var)",
    )
    parser.add_argument(
        "--port",
        type=str,
        default=os.environ.get("VLLM_PORT", "8000"),
        help="Port number (default: 8000, or set VLLM_PORT env var)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("VLLM_API_KEY", "not-used"),
        help="API key (or set VLLM_API_KEY env var)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/Cosmos-Reason2-2B",
        help="Model name (default: nvidia/Cosmos-Reason2-2B)",
    )

    # Test selection
    parser.add_argument(
        "--tests",
        nargs="+",
        choices=list(TESTS.keys()) + ["all"],
        default=["all"],
        help="Which tests to run (default: all)",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available tests and exit"
    )

    args = parser.parse_args()

    # Handle VLLM_ENDPOINT environment variable (IP:PORT format)
    vllm_endpoint = os.environ.get("VLLM_ENDPOINT")
    if vllm_endpoint and not args.host:
        if ":" in vllm_endpoint:
            args.host, args.port = vllm_endpoint.rsplit(":", 1)
        else:
            args.host = vllm_endpoint

    return args


def list_tests():
    """Print available tests and their descriptions."""
    print("\n" + "=" * 70)
    print("Available Cosmos Reason 2 Tests")
    print("=" * 70)
    for test_id, test in TESTS.items():
        print(f"\n  {test_id}")
        print(f"    Name: {test['name']}")
        print(f"    Description: {test['description']}")
        print(f"    Media: {test['media_type']}")
    print("\n" + "=" * 70)
    print("Run with: python cosmos_reason2_tests.py --tests <test_id> [<test_id> ...]")
    print("Run all:  python cosmos_reason2_tests.py --tests all")
    print("=" * 70 + "\n")


def create_client(host: str, port: str, api_key: str) -> OpenAI:
    """Create and return an OpenAI client configured for vLLM."""
    base_url = f"http://{host}:{port}/v1"
    print(f"Connecting to: {base_url}")
    return OpenAI(base_url=base_url, api_key=api_key)


def run_test(
    client: OpenAI, model: str, test_id: str, test_config: dict
) -> Optional[str]:
    """Run a single test and return the result."""
    print(f"\n{'=' * 70}")
    print(f"Test: {test_config['name']}")
    print(f"Description: {test_config['description']}")
    print(f"{'=' * 70}")

    # Build message content (media first, then text - per prompt guide)
    content = []

    # Add media (image or video)
    media_type = test_config["media_type"]
    media_url = test_config["media_url"]

    if media_type == "image_url":
        content.append({"type": "image_url", "image_url": {"url": media_url}})
    elif media_type == "video_url":
        content.append({"type": "video_url", "video_url": {"url": media_url}})

    # Add text prompt
    content.append({"type": "text", "text": test_config["prompt"]})

    # Build messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content},
    ]

    # Build extra_body for video FPS if specified
    extra_body = {}
    if "fps" in test_config:
        extra_body["media_io_kwargs"] = {"video": {"fps": test_config["fps"]}}

    print(f"\nMedia URL: {media_url}")
    print(
        f"Prompt: {test_config['prompt'][:100]}..."
        if len(test_config["prompt"]) > 100
        else f"Prompt: {test_config['prompt']}"
    )
    print("\nWaiting for response...")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=test_config.get("max_tokens", 1024),
            temperature=test_config.get("temperature", 0.7),
            top_p=test_config.get("top_p", 0.8),
            extra_body=extra_body if extra_body else None,
        )

        result = response.choices[0].message.content

        print(f"\n{'-' * 70}")
        print("Response:")
        print(f"{'-' * 70}")
        print(result)

        # Print usage stats
        if response.usage:
            print(
                f"\n[Tokens - Prompt: {response.usage.prompt_tokens}, "
                f"Completion: {response.usage.completion_tokens}, "
                f"Total: {response.usage.total_tokens}]"
            )

        return result

    except Exception as e:
        print(f"\nError running test: {e}")
        return None


def main():
    """Main entry point."""
    args = parse_args()

    # List tests if requested
    if args.list:
        list_tests()
        return

    # Validate connection settings
    if not args.host:
        print("Error: Host not specified.")
        print("Use --host <IP> or set VLLM_ENDPOINT or VLLM_HOST environment variable.")
        print("\nExample:")
        print("  export VLLM_ENDPOINT='89.169.115.247:8000'")
        print("  python cosmos_reason2_tests.py")
        sys.exit(1)

    # Create client
    client = create_client(args.host, args.port, args.api_key)

    # Determine which tests to run
    if "all" in args.tests:
        tests_to_run = list(TESTS.keys())
    else:
        tests_to_run = args.tests

    print(f"\n{'#' * 70}")
    print(f"# Cosmos Reason 2 Prompt Tests")
    print(f"# Model: {args.model}")
    print(f"# Tests to run: {len(tests_to_run)}")
    print(f"{'#' * 70}")

    # Run tests
    results = {}
    for test_id in tests_to_run:
        if test_id in TESTS:
            result = run_test(client, args.model, test_id, TESTS[test_id])
            results[test_id] = {
                "name": TESTS[test_id]["name"],
                "success": result is not None,
                "response": result,
            }
        else:
            print(f"\nWarning: Unknown test '{test_id}', skipping.")

    # Print summary
    print(f"\n{'#' * 70}")
    print("# Summary")
    print(f"{'#' * 70}")

    successful = sum(1 for r in results.values() if r["success"])
    failed = len(results) - successful

    print(f"\nTotal tests: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("\nFailed tests:")
        for test_id, result in results.items():
            if not result["success"]:
                print(f"  - {test_id}: {result['name']}")

    print(f"\n{'#' * 70}")
    print("# Tests completed")
    print(f"{'#' * 70}\n")


if __name__ == "__main__":
    main()
