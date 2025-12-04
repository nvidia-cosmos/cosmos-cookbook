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

# https://docs.astral.sh/uv/guides/scripts/#using-a-shebang-to-create-an-executable-file
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "accelerate>=1.10.1",
#   "torchvision>=0.23.0",
#   "torchcodec>=0.6.0",
#   "qwen-vl-utils>=0.0.14",
#   "torch>=2.7.1",
#   "transformers>=4.57.0",
# ]
# [tool.uv.sources]
# torch = [
#   { index = "pytorch-cu128"},
# ]
# torchvision = [
#   { index = "pytorch-cu128"},
# ]
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///

"""Evaluation script for Cosmos-Reason1 using configuration file.

Example:

```shell
./avha_caption.py
./avha_caption.py --output-dir ./output_dir
./avha_caption.py --video_dir ./videos --output-dir ./output_dir
```
"""

import argparse
from pathlib import Path
from typing import Any, Optional

from misc_utils import (
    get_list_of_files,
    iterate_with_timing_info,
    read_text_file,
    run_sharded_computation,
    write_text_file,
)
from model_qwen3 import LocalModelQwen3
from model_reason import LocalModel

SCRIPT_DIR = Path(__file__).parent
SEPARATOR = "-" * 20


def open_model(model_path: str, gpu_id):
    """Return an object that encapsulates the model."""
    print(f"Loading model from path {model_path}.")
    if model_path.startswith("Qwen/Qwen3"):
        return LocalModelQwen3(model_path, gpu_id=gpu_id)
    else:
        return LocalModel(model_path, gpu_id=gpu_id)


def process_videos(
    video_list: list[str],
    model_path: str,
    video_dir: Path,
    output_dir: Path,
    system_prompt: str,
    user_prompt: str,
    force_reprocess: bool = False,
    gpu_id: Optional[int] = None,
):
    """Process a list of videos, and record the results.

    Args:
        model_path: A string which identifies the model.
        video_dir: Directory where videos are stored.
        output_dir: Directory to write outputs.
        system_prompt: System prompt for the model.
        user_prompt: User prompt for the model.
        force_reprocess: If true, overwrite previously computed results.
        gpu_id: The GPU on which this process is running.

    Returns:
       The number of processed videos.
    """

    prefix_str = "" if gpu_id is None else f"s{gpu_id}: "

    model = open_model(model_path, gpu_id)
    model.set_system_prompt(system_prompt)

    def process_video_fn(video_filename: str) -> bool:
        # Process a single video.
        video_path = video_dir / video_filename
        output_path = output_dir / (video_filename + ".json")

        # Skip if already processed (unless force flag is set)
        if output_path.exists():
            if not force_reprocess:
                print(f"{prefix_str}Skipping {video_filename} (already processed).")
                return False
            else:
                print(f"{prefix_str}Force re-processing {video_filename}.")
        else:
            print(f"{prefix_str}Processing {video_filename}...")

        result = model.generate(user_prompt, video_path=video_path)
        print(f"{prefix_str}Writing result to {output_path}.")
        write_text_file(result, output_path)
        return True

    totalp = iterate_with_timing_info(
        video_list, process_video_fn, prefix_str=f"GPU {gpu_id}: "
    )
    return totalp


def process_fn(video_list: list[str], other_args: list[Any], shard_id: int) -> int:
    """Pickleable function which wraps process_videos."""
    (
        model_path,
        video_dir,
        output_dir,
        system_prompt,
        user_prompt,
        force_reprocess,
    ) = other_args

    return process_videos(
        video_list,
        model_path=model_path,
        video_dir=video_dir,
        output_dir=output_dir,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        force_reprocess=force_reprocess,
        gpu_id=shard_id,
    )


def join_totals_fn(results: list[Optional[int]]) -> int:
    """Return the sum of non-None items in the list."""

    def to_int(r: Any) -> int:
        if r is None:
            return 0
        elif isinstance(r, int):
            return r
        else:
            return 0

    nps = [to_int(ri) for ri in results]
    return sum(nps)


def main():
    """Main script."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate Cosmos-Reason1 model using configuration file"
    )

    parser.add_argument(
        "--model-path",
        "-m",
        default="nvidia/Cosmos-Reason1-7B",
        # Also try:
        #   "Qwen/Qwen3-VL-8B-Instruct",
        #   "Qwen/Qwen3-VL-30B-A3B-Instruct",
        help=(
            "Path to the model.  This can be a huggingface model name, "
            "or a file path to the pretrained weights in safetensor format."
        ),
    )

    parser.add_argument(
        "--video-dir",
        "-v",
        default="./eval/videos",
        help="Directory where the input videos are located.",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        default="./output/baseline",
        help="Output directory for results.",
    )

    parser.add_argument(
        "--system-prompt",
        default="./prompts/system_prompt.txt",
        help="Text file containing the system prompt.",
    )

    parser.add_argument(
        "--user-prompt",
        default="./prompts/user_prompt.txt",
        help="Text file containing the user prompt.",
    )

    parser.add_argument(
        "--dryrun", action="store_true", help="Do a dry run (not running the model)."
    )

    parser.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        help="Number of GPUs to use for parallel processing.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of videos even if output files exist.",
    )

    args = parser.parse_args()
    print(f"Script arguments: {args}")

    # Resolve file path to video directory.
    video_dir = Path(args.video_dir)
    if not video_dir.is_absolute():
        video_dir = SCRIPT_DIR / video_dir
    print(f"Video directory set to: {video_dir}")

    # Resolve file path to output directory.
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = SCRIPT_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory set to: {output_dir}")

    # Load system prompt.
    system_prompt_fname = Path(args.system_prompt)
    if not system_prompt_fname.is_absolute():
        system_prompt_fname = SCRIPT_DIR / system_prompt_fname
    system_prompt = read_text_file(system_prompt_fname)
    print(f"System prompt:\n {system_prompt}")

    # Load user prompt.
    user_prompt_fname = Path(args.user_prompt)
    if not user_prompt_fname.is_absolute():
        user_prompt_fname = SCRIPT_DIR / user_prompt_fname
    user_prompt = read_text_file(user_prompt_fname)
    print(f"User prompt:\n {user_prompt}")

    # Handle dry run.
    if args.dryrun:
        print("Dry run -- no model.")
        model_path = None
    else:
        model_path = args.model_path
        print(f"Using model {model_path}.")

    # Parallelism
    num_gpus = args.num_gpus
    # Overwrite previous results
    force_reprocess = args.force

    video_list = get_list_of_files(video_dir)
    print(f"Found {len(video_list)} videos.")

    other_args = (
        model_path,
        video_dir,
        output_dir,
        system_prompt,
        user_prompt,
        force_reprocess,
    )
    num_processed_results = run_sharded_computation(
        process_fn,
        join_totals_fn,
        input_data=video_list,
        other_args=other_args,
        num_shards=num_gpus,
    )
    print("Done.")
    print(f"Processed {num_processed_results} videos.")


if __name__ == "__main__":
    main()
