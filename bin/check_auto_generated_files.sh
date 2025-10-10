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


set -euo pipefail

echo "ERROR: Do not commit auto-generated files. Please gitignore them." >&2
echo "The following auto-generated files are staged for commit:" >&2
for FILE in "$@"; do
    echo "  - $FILE" >&2
done
echo "" >&2
exit 1
