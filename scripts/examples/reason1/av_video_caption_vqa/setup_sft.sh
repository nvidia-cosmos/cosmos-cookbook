#!/bin/bash

# Check if at least one argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 /path/to/cosmos-reason"
  exit 1
fi

mkdir -v "$1"/examples/post_training_hf/prompts
cp -v prompts/*.txt "$1"/examples/post_training_hf/prompts
cp -v configs/*.toml "$1"/examples/post_training_hf/configs
cp -v scripts/*.py "$1"/examples/post_training_hf/scripts
