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

"""Script to print a score comparison after avha_judge has been run."""


import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def read_json_file(filename: Path):
    """Load evaluation configuration from JSON file."""
    with open(filename, "r") as f:
        return json.load(f)


files = [
    "scores/baseline",
    "scores/sft",
]

summaries = []
for fd in files:
    fname = SCRIPT_DIR / fd / "summary.json"
    summaries.append(read_json_file(fname))

keys = summaries[0].keys()
for k in keys:
    vals = [f"{(s[k]['mean_score']):.3}" for s in summaries]
    print(f"{k}, " + ", ".join(vals))
