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

"""Miscellaneous utility functions."""

from typing import Any, Optional

import json
import os
from pathlib import Path
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


SCRIPT_DIR = Path(__file__).parent
JSON_FAIL_DIR = SCRIPT_DIR / "json_failures"


def get_base_filename(filename: str) -> str:
    """Get the base filename from a URL."""
    # E.g. get_base_filename("s3://lha-datasets/uber/02c9398b-3eee-528d-b0e1-3a54e9bc8d6a.mp4")
    # Should return "02c9398b-3eee-528d-b0e1-3a54e9bc8d6a.mp4"
    return filename.split('/')[-1]


def get_filename_prefix(filename: str) -> str:
    """Strip anything after the first '.' in a filename."""
    # E.g. get_filename_prefix("foo.label.json") = "foo"
    return filename.split('.')[0]


def read_text_file(filename: Path) -> str:
    """Read a text file."""
    with open(filename, 'r') as f:
        return f.read()


def write_text_file(text: str, filename: Path) -> str:
    """Read a text file."""
    with open(filename, 'w') as f:
        f.write(text)


def read_json_file(filename: Path):
    """Load evaluation configuration from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def write_json_file(json_data, filename: Path):
    """Save evaluation configuration back to JSON file."""
    with open(filename, 'w') as f:
        json.dump(json_data, f, indent=4)


def read_json_output_file(filename: Path, fix_json=True):
    """Read a file containing json output."""

    with open(filename, 'r') as f:
        orig_content = f.read()

    def parse_json_content(f_content):
        # The model may have output something other than pure JSON,
        # e.g. "Here's my answer: ```json { ... }```."
        # We attempt to detect and fix that here.

        # Find the first '{' and the last '}'
        first_brace = f_content.find('{')
        last_brace = f_content.rfind('}')
        if first_brace == -1 or last_brace == -1 or first_brace >= last_brace:
            raise ValueError("Failed to find JSON dictionary.")

        # Extract the JSON content between the braces (including the braces)
        json_content = f_content[first_brace:last_brace + 1]

        # Parse the JSON.
        result = json.loads(json_content)
        return result

    # Try to parse the json.
    # JSON parse failures will be written to a separate directory to examine later.
    # These may need to be fixed by hand.
    try:
        result = parse_json_content(orig_content)
        return result
    except Exception as e:
        print(f"Failed to parse JSON from file {filename}: {e}")
        fail_path = JSON_FAIL_DIR / get_base_filename(str(filename))
        write_text_file(orig_content, fail_path)
        return None


def get_list_of_files(directory_name: str) -> list[str]:
    """Returns a list of all files in the given directory."""
    if not os.path.exists(directory_name):
        raise FileNotFoundError(f"Directory '{directory_name}' not found")

    if not os.path.isdir(directory_name):
        raise ValueError(f"'{directory_name}' is not a directory")

    files = []
    for item in os.listdir(directory_name):
        item_path = os.path.join(directory_name, item)
        if os.path.isfile(item_path):
            files.append(item)

    return sorted(files)


def iterate_with_timing_info(item_list, process_fn, prefix_str: str = "") -> int:
    """Iterate over the items in item_list, while tracking elapsed and predicted times.

    Args:
        item_list: A list of items to process.

        process_fn: A function to run on each item.  f(item) -> bool
            Returns true if the item was processed successfully;
            false if the item was skipped.  Skipped items will not
            be used to calculate timing info.

        prefix_str: A string to prefix log messages.

    Returns:
        The number of items processed, ignoring skipped items.
    """

    total_num = len(item_list)
    total_processed = 0   # number of videos that we've run the model on
    total_elapsed_time = 0.0

    for (i, item) in enumerate(item_list):
        start_time = time.time()

        print(f"{prefix_str}Processing {i}/{total_num}.")
        success = process_fn(item)
        if not success:
            continue   # Don't record timing for failed or skipped items.

        # Calculate and display processing times
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_processed += 1
        total_elapsed_time += elapsed_time
        average_elapsed_time = total_elapsed_time / total_processed
        print(f"{prefix_str}Elapsed time: {elapsed_time:.2f} secs;"
              f" total time: {total_elapsed_time:.2f} secs;"
              f" average time: {average_elapsed_time:.2f} secs/item")

        # Calculate remaining time.
        remaining_items = total_num - i - 1
        time_remaining = average_elapsed_time * remaining_items / 3600
        print(f"{prefix_str}Remaining: {remaining_items}; remaining time: {time_remaining:.2f} hours.")

    print(f"{prefix_str}Processed {total_processed} items.")
    return total_processed


def extract_tagged_text(text: str, key: str, fallback: str = "") -> dict[str, str]:
    """Extract text between <key> and </key> tags."""
    match = re.search(f"<{key}>(.*?)</{key}>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    elif fallback:
        # Look for a fallback expression at the end of the response, e.g. "[1-5]" or "[A-D]"
        fallback_match = re.search(r'\b({fallback})\b(?!.*\b{fallback}\b)', text)
        if fallback_match:
            return fallback_match.group(1)
    return None


def string_to_int(s: str) -> Optional[int]:
    """Return an integer, or None."""
    if not s or not isinstance(s, str):
        return None
    # Strip whitespace
    s = s.strip()
    try:
        return int(s)
    except ValueError:
        return None


def split_data_into_shards(input_data: list, num_shards: int) -> list[list]:
    """Split a long list of items into num_shards smaller lists."""

    print(f"Total number of items: {len(input_data)}")

    buckets = [[] for i in range(0, num_shards)]
    for (i, val) in enumerate(input_data):
        bi = i % num_shards
        buckets[bi].append(val)

    bsizes = [len(b) for b in buckets]
    print(f"Shard sizes: {bsizes}")
    return buckets


def run_sharded_computation(
    computation_fn,
    result_join_fn,
    input_data: list[Any],
    other_args: list[Any],
    num_shards: int
) -> Any:
    """Run multiple copies of computation_fn in parallel, and join the results.

    Args:
        computation_fn:  A function which takes a list of inputs, and returns a result.
            f(input_list, other_args, shard_id: int) -> result.

        result_join_fn:  A function to combine results.
            f(list_of_results) -> result

        input_data: A list of inputs, which will be sharded into smaller lists.
        other_args: Additional arguments to pass as 'other_args' to computation_fn.
        num_shards: The number of parallel processes.

    Returns:
        The result of running computations in parallel, and joining the results.
    """

    if num_shards == 1:
        result = computation_fn(input_data, other_args, shard_id=0)
        return result

    # Otherwise divide work across multiple shards
    sharded_inputs = split_data_into_shards(input_data, num_shards)
    shard_results = [None for i in range(0, num_shards)]

    with ProcessPoolExecutor(max_workers=num_shards) as executor:
        future_to_shard_id = {
            executor.submit(computation_fn, sh_inputs, other_args, shard_id=i): i
            for (i, sh_inputs) in enumerate(sharded_inputs)
        }
        for future in as_completed(future_to_shard_id):
            shard_id = future_to_shard_id[future]
            try:
                sh_result = future.result()
                print(f"Shard {shard_id} completed processing of outputs.")
                shard_results[shard_id] = sh_result
            except Exception as e:
                print(f"Shard {shard_id} encountered error: {e}")

    print("All shards have finished.  Joining results:")
    final_result = result_join_fn(shard_results)
    return final_result
