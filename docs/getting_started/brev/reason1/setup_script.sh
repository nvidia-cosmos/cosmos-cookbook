#!/bin/bash
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

echo "[*] Updating apt and installing git + curl..."
sudo apt-get update
sudo apt-get install -y git curl

echo "[*] Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh

echo "[*] Adding uv to PATH..."
if [ -f "$HOME/.local/bin/env" ]; then
    # POSIX-compatible "source"
    . "$HOME/.local/bin/env"
else
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "[*] Installing Hugging Face CLI via uv..."
uv tool install -U "huggingface_hub[cli]"

echo
echo "[*] Setup finished."
echo "Next step (manual, interactive): run hf auth login and paste your HF token."
