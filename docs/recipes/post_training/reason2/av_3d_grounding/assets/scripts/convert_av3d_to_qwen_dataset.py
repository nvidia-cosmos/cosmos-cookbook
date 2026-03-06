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
Convert AV 3D grounding evaluation dataset to Qwen VL annotations.json format.

Source format (e.g. cosmos-reason1 or cosmos-reason2 eval/train dataset):
  - meta.json: list of {"id", "media": "images/<frame>.jpg", "text": "text/<frame>.json"}
  - text/<frame>.json: {"annotations": [{"label": "...", "bbox_3d": [...]}, ...], ...}

Output format (Qwen-Finetune eval/train dataset):
  - annotations.json: list of {"image": "images/<frame>.jpg", "conversations": [human, gpt]}
  - Optionally copy/symlink images to output_dir/images/

Example (source dir = cosmos-reason1/.../dataset/eval/ or cosmos-reason2/.../dataset/train/):
  python path/to/convert_av3d_to_qwen_dataset.py \\
    --source_dir /path/to/cosmos-reason1/.../dataset/eval/ \\
    --output_dir ./qwenvl/data/eval \\
    [--copy_images]
"""

import argparse
import json
import shutil
from pathlib import Path

USER_PROMPT = (
    "Find all vehicles in this image. For each vehicle, provide its 3D bounding box "
    "coordinates including x, y, z, x_size, y_size, z_size, roll, pitch, yaw and the label "
    "of the vehicle. The output format required is JSON: "
    '`[{"bbox_3d":[x, y, z, x_size, y_size, z_size, roll, pitch, yaw],"label":"category"}]`.'
)


def convert_entry(source_dir: Path, meta_entry: dict) -> dict | None:
    """Convert one meta entry to Qwen format."""
    media_path = meta_entry.get("media") or meta_entry.get("image")
    text_path = meta_entry.get("text") or meta_entry.get("conversation")
    if not media_path or not text_path:
        return None

    text_file = source_dir / text_path
    if not text_file.exists():
        return None
    with open(text_file) as f:
        data = json.load(f)

    annotations = data.get("annotations", [])
    # Qwen format uses bbox_3d then label per object
    bbox_list = [{"bbox_3d": a["bbox_3d"], "label": a["label"]} for a in annotations]
    gpt_value = json.dumps(bbox_list, separators=(",", ": "))

    return {
        "image": media_path,
        "conversations": [
            {"from": "human", "value": f"<image>\n{USER_PROMPT}"},
            {"from": "gpt", "value": gpt_value},
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert AV 3D grounding eval (meta.json + text/) to Qwen annotations.json"
    )
    parser.add_argument(
        "--source_dir",
        type=Path,
        required=True,
        help="Source directory (e.g. cosmos-reason2/.../dataset/eval/) containing meta.json, images/, and text/",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./qwenvl/data/eval"),
        help="Output directory for annotations.json (and images/ if --copy_images)",
    )
    parser.add_argument(
        "--meta_file",
        default="meta.json",
        help="Meta filename (default: meta.json)",
    )
    parser.add_argument(
        "--copy_images",
        action="store_true",
        help="Copy images from source_dir to output_dir/images/ so paths in annotations match.",
    )
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()

    meta_path = source_dir / args.meta_file
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")

    with open(meta_path) as f:
        meta_list = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.copy_images:
        (output_dir / "images").mkdir(parents=True, exist_ok=True)

    out_annotations = []
    for i, entry in enumerate(meta_list):
        rec = convert_entry(source_dir, entry)
        if rec is None:
            continue
        if args.copy_images:
            src_img = source_dir / rec["image"]
            if src_img.exists():
                dst_img = output_dir / rec["image"]
                dst_img.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_img, dst_img)
        out_annotations.append(rec)

    out_path = output_dir / "annotations.json"
    with open(out_path, "w") as f:
        json.dump(out_annotations, f, indent=2)

    print(f"Wrote {len(out_annotations)} entries to {out_path}")
    if args.copy_images:
        print(f"Copied images to {output_dir / 'images'}")


if __name__ == "__main__":
    main()
