#!/bin/bash
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

echo "[*] Cloning cosmos-reason1 repo..."
cd $HOME
if [ ! -d cosmos-reason1 ]; then
    git clone https://github.com/nvidia-cosmos/cosmos-reason1.git
else
    echo "    cosmos-reason1 already exists, skipping clone."
fi

echo
echo "[*] Setup finished."
echo "Next step (manual, interactive): run hf auth login and paste your HF token."