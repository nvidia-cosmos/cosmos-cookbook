#!/usr/bin/env bash
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


# Convert markdown files to ipynb files.

set -euo pipefail

for md_file in "$@"; do
    # Skip if the file does not contain `jupyter:`
    if ! grep -qE '^jupyter:$' "$md_file"; then
        continue
    fi
    ipynb_file="${md_file/%.md/.ipynb}"
    if [[ ! -f "$ipynb_file" ]]; then
        uvx jupytext -q --to ipynb --update "$md_file"
    else
        uvx jupytext -q --set-formats ipynb,md "$ipynb_file"
        uvx jupytext -q --sync "$ipynb_file"
    fi
done
