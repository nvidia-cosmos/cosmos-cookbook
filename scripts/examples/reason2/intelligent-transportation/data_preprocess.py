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

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm


def parse_arguments() -> str:
    """
    Parse command line arguments for data path.

    Returns:
        str: Path to the directory containing annotation JSONs
    """
    parser = argparse.ArgumentParser(
        description="Format WTS annotations for best view training data generation"
    )
    parser.add_argument(
        "--data_path", type=Path, required=True, help="Directory with annotation JSONs"
    )
    args = parser.parse_args()
    return str(args.data_path)


def process_question(row: Dict[str, Any]) -> str:
    """
    Process a question row into a formatted prompt with multiple choice options.

    Args:
        row: Question data containing question text and options (a, b, c, d)

    Returns:
        Formatted question prompt with video tag and multiple choice options
    """
    prompt = f"<video> \n {row['question']} \n "

    # Add multiple choice options if they exist
    for option in ["a", "b", "c", "d"]:
        if option in row:
            prompt += f"{option.upper()}: {row[option]} \n "

    return prompt


def format_training_data_mcq_llava(
    id: str,
    video_file: str,
    question: str,
    answer: str,
    qtype: str,
    phase: str,
    wts_id: str,
) -> Dict[str, Any]:
    """
    Format training data for MCQ tasks in LLaVA conversation format.

    Args:
        id: Unique identifier for the training sample
        video_file: Path to the video file
        question: Question text with multiple choice options
        answer: Correct answer (A, B, C, or D)
        qtype: Question type identifier
        phase: Traffic scene phase
        wts_id: WTS scene identifier

    Returns:
        Formatted training data dictionary in LLaVA format
    """
    item = {
        "id": id,
        "wts_id": wts_id,
        "video": video_file,
        "type": qtype,
        "phase": phase,
        "conversations": [
            {
                "from": "human",
                "value": question
                + "\nAnswer with the option's letter from the given choices directly.",
            },
            {"from": "gpt", "value": answer.upper()},
        ],
    }
    return item


def process_wts_environment_mcq(root_dir: str, split: str) -> List[Dict]:
    """
    Process WTS environment view MCQ data.

    Args:
        root_dir: Base path for input files
        split: Split name

    Returns:
        List of formatted MCQ
    """

    mcq_env_dataset = []
    root_dir = os.path.join(root_dir)

    for name in tqdm(os.listdir(root_dir)):
        if "normal_trimmed" in name:
            continue

        env_file = os.path.join(root_dir, name, "environment", name + ".json")
        if not os.path.exists(env_file):
            print(f"Environment file not found for {name}")
            continue

        with open(env_file, "r") as e:
            data = json.load(e)

        wts_id = data[0]["id"]

        # Process environment questions for each video
        for vid in data[0]["overhead_videos"]:
            fir = vid[:-4]
            cnt = 0
            vid2 = os.path.join(name, "overhead_view", vid)
            if "normal_trimmed" in root_dir:
                vid2 = "normal_trimmed/" + vid2
            # print(vid2)

            for row in data[0]["environment"]:
                lab = fir + "_" + str(cnt)
                question = process_question(row)
                item = format_training_data_mcq_llava(
                    lab,
                    vid2,
                    question,
                    row["correct"],
                    "environment",
                    "full_video",
                    wts_id,
                )
                mcq_env_dataset.append(item)
                cnt += 1

    return mcq_env_dataset


def main():
    """
    Main execution function that orchestrates the complete data processing pipeline.

    This function:
    1. Parses command line arguments
    2. Loads best view mappings
    3. Processes all datasets based on configuration flags
    4. Merges and saves datasets in various combinations
    5. Generates formatted training data in LLaVA format
    """
    # Parse arguments and setup
    user_path = parse_arguments()
    os.makedirs(user_path, exist_ok=True)

    print("Starting WTS annotations processing...")

    # Process WTS MCQ datasets
    train_mcq_env_dataset = process_wts_environment_mcq(
        os.path.join(user_path, "annotations", "vqa", "train"), "train"
    )
    train_mcq_env_dataset += process_wts_environment_mcq(
        os.path.join(user_path, "annotations", "vqa", "train", "normal_trimmed"),
        "train",
    )

    output_dir = user_path
    os.makedirs(output_dir, exist_ok=True)
    # Save dataset
    with open(os.path.join(output_dir, "environment_mcq_llava_train.json"), "w") as f:
        json.dump(train_mcq_env_dataset, f, indent=4)

    val_mcq_env_dataset = process_wts_environment_mcq(
        os.path.join(user_path, "annotations", "vqa", "val"), "val"
    )
    val_mcq_env_dataset += process_wts_environment_mcq(
        os.path.join(user_path, "annotations", "vqa", "val", "normal_trimmed"), "val"
    )

    # Save dataset
    with open(os.path.join(output_dir, "environment_mcq_llava_val.json"), "w") as f:
        json.dump(val_mcq_env_dataset, f, indent=4)

    print("WTS annotations processing completed!")
    print(f"Train dataset length: {len(train_mcq_env_dataset)}")
    print(f"Val dataset length: {len(val_mcq_env_dataset)}")


if __name__ == "__main__":
    main()
