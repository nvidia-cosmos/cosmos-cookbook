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

import json
import re
from pathlib import Path


def main():
    batch_path = Path("dream_gen_benchmark/gr1_object/batch_input.json")
    out_path = Path("dream_gen_benchmark/gr1_object/gr1_batch.jsonl")

    with open(batch_path) as f:
        items = json.load(f)

    lines = []
    for i, item in enumerate(items):
        input_video = item["input_video"]
        prompt = item["prompt"]
        # input_path: relative to JSONL's directory (gr1_object), so use basename
        input_path = Path(input_video).name
        # unique name: sanitize and add index to avoid collisions
        base_name = Path(input_path).stem
        safe_name = re.sub(r"[^\w\-.]", "_", base_name)[:80]
        name = f"{i:03d}_{safe_name}"

        sample = {
            "inference_type": "image2world",
            "name": name,
            "prompt": prompt,
            "input_path": input_path,
            "num_output_frames": 93,
            "resolution": "432,768",
            "seed": 0,
            "guidance": 7,
        }
        lines.append(json.dumps(sample))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} samples to {out_path}")


if __name__ == "__main__":
    main()
