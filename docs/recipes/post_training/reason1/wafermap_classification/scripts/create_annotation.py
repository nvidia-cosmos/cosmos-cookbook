# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
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

"""Generate annotation JSON files for LLaVA model training using wafer map images."""

import argparse
import json
import os
import random

DEFECT_TYPES = [
    "loc",
    "edge-loc",
    "center",
    "edge-ring",
    "scratch",
    "near-full",
    "donut",
    "random",
]
DEFECT_TO_INDEX = {d: i for i, d in enumerate(DEFECT_TYPES)}
IMAGE_PROMPT = "<image>\nThis is a image of a wafer map, the yellow pattern in the circle refers to the defect pattern. "
TEMPLATE_DIR = "./annotation_template"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate JSON annotation file for LLaVA"
    )
    parser.add_argument(
        "-r",
        "--root-path",
        default="../WM811K_data/train/",
        help="Root path to image dataset",
    )
    parser.add_argument(
        "-o", "--output-path", default="./annotation/", help="Output path for JSON file"
    )
    return parser.parse_args()


def load_template(filename):
    """Load a JSON template file."""
    with open(os.path.join(TEMPLATE_DIR, filename), "r") as f:
        return json.load(f)


def create_entry(image_path, human_text, gpt_text):
    """Create a QA entry with image and conversations."""
    return {
        "conversations": [
            {"from": "human", "value": IMAGE_PROMPT + human_text},
            {"from": "gpt", "value": gpt_text},
        ],
        "image": image_path,
    }


def generate_qa_entries(image_path, label, templates):
    """Generate all QA types for a single image."""
    golden, binary, multi = templates
    entries = []

    # Golden QA (direct question)
    entries.append(
        create_entry(
            image_path,
            golden["golden_human_list"][0],
            golden["golden_ans_list"][0].replace("[label]", label),
        )
    )

    # Binary QA - Yes (correct guess)
    entries.append(
        create_entry(
            image_path,
            binary["binary_human_list"][0].replace("[label]", label),
            binary["binary_ans_list"][0].replace("[label]", label),
        )
    )

    # Binary QA - No (wrong guess)
    wrong_idx = random.choice(
        [i for i in range(len(DEFECT_TYPES)) if i != DEFECT_TO_INDEX.get(label, -1)]
    )
    wrong_defect = DEFECT_TYPES[wrong_idx]
    entries.append(
        create_entry(
            image_path,
            binary["binary_human_list"][0].replace("[label]", wrong_defect),
            binary["binary_ans_list"][1].replace("[label]", label),
        )
    )

    # Multiple choice QA
    entries.append(
        create_entry(
            image_path,
            multi["multiple_human_list"][0],
            multi["multiple_ans_list"][DEFECT_TO_INDEX.get(label, 0)].replace(
                "[label]", label
            ),
        )
    )

    return entries


def main():
    args = parse_args()
    print(f"Processing images from: {args.root_path}")

    # Load templates
    templates = (
        load_template("golden_qa_template.json"),
        load_template("binary_qa_template.json"),
        load_template("multi_qa_template.json"),
    )

    # Process all images
    qa_data = []
    for folder in sorted(os.listdir(args.root_path)):
        folder_path = os.path.join(args.root_path, folder)
        if not os.path.isdir(folder_path):
            continue
        print(f"  Processing: {folder}")

        for image in os.listdir(folder_path):
            image_path = os.path.join(folder, image)
            qa_data.extend(generate_qa_entries(image_path, folder, templates))

    # Shuffle and assign IDs
    random.shuffle(qa_data)
    for idx, entry in enumerate(qa_data):
        entry["id"] = idx

    # Save output
    os.makedirs(args.output_path, exist_ok=True)
    output_file = os.path.join(args.output_path, "WM811K_train.json")
    with open(output_file, "w") as f:
        json.dump(qa_data, f, indent=4)

    print(f"\nDone! Generated {len(qa_data)} QA entries -> {output_file}")


if __name__ == "__main__":
    main()
