#!/bin/bash
# Download checkpoints script for NVIDIA Cosmos models
# Usage: ./download_checkpoints.sh <HF_TOKEN>

set -e  # Exit on any error

# Check if HF_TOKEN is provided
if [ $# -eq 0 ]; then
    echo "Error: HF_TOKEN is required"
    echo "Usage: $0 <HF_TOKEN>"
    echo "Example: $0 hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    exit 1
fi

HF_TOKEN="$1"

# Validate token format (basic check)
if [[ ! "$HF_TOKEN" =~ ^hf_[a-zA-Z0-9]{34}$ ]]; then
    echo "Warning: HF_TOKEN format may be invalid. Expected format: hf_xxx"
    echo "Continuing anyway..."
fi

echo "Starting checkpoint download process..."

# Install huggingface-hub if not already installed
echo "Installing/updating huggingface-hub..."
pip install --upgrade huggingface-hub

# Verify huggingface-cli is available
if ! command -v huggingface-cli &> /dev/null; then
    echo "Error: huggingface-cli not found after installation"
    exit 1
fi

echo "huggingface-cli installed successfully"

# Create checkpoints directory if it doesn't exist
CHECKPOINT_DIR="./checkpoints"
mkdir -p "$CHECKPOINT_DIR"

echo "Downloading nvidia/Cosmos-Reason1-7B-Reward model..."
echo "Target directory: $CHECKPOINT_DIR"

# Download the model using huggingface-cli
huggingface-cli download \
    nvidia/Cosmos-Reason1-7B-Reward \
    --local-dir "$CHECKPOINT_DIR" \
    --token "$HF_TOKEN"

echo "Download completed successfully!"
echo "Model saved to: $CHECKPOINT_DIR"

# List downloaded files
echo ""
echo "Downloaded files:"
find "$CHECKPOINT_DIR" -type f | head -10
if [ $(find "$CHECKPOINT_DIR" -type f | wc -l) -gt 10 ]; then
    echo "... and $(( $(find "$CHECKPOINT_DIR" -type f | wc -l) - 10 )) more files"
fi

echo ""
echo "Checkpoint download complete!"
