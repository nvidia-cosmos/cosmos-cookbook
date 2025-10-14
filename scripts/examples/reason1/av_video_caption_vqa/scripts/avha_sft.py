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

"""Supervised Fine-Tuning (SFT) dataset with plain text caption loading."""
# ruff: noqa: E402


# Intended for use with cosmos-rl and cosmos-reason1.
# This file requires a .toml configuration file, and associated prompt files.
# To run, do the following:
#
#   git clone https://github.com/nvidia-cosmos/cosmos-reason1.git
#   cd cosmos-reason1/examples/post_training_hf
#
#   cp path/to/avha_sft.py scripts
#   cp paht/to/avha_sft.toml configs
#   mkdir prompts
#   cp path/to/system_prompt.txt path/to/user_prompt.txt prompts
#
#   just install
#   source .venv/bin/activate
#   cosmos-rl --config configs/avha_sft.toml scripts/avha_sft.py


from re import S

from cosmos_reason1_utils.script import init_script

init_script()

import argparse
import json
from pathlib import Path

import toml
from cosmos_rl.launcher.worker_entry import main as launch_worker
from cosmos_rl.policy.config import Config
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.utils.util import basename_from_modelpath
from torch.utils.data import Dataset
from transformers import AutoTokenizer

FPS = 2
# Default patch size is 28 x 28 -- maintain aspect ratio.
# + 1 to account for floating point error.
MAX_PIXELS = (16 * 9 * 28 * 28) + 1
SCRIPT_DIR = Path(__file__).parent


def read_text_file(filename: Path) -> str:
    """Read a text file and return its contents as a string."""
    with open(filename, "r") as f:
        return f.read()


def read_caption(caption_file) -> str:
    with open(caption_file, "r") as f:
        caption_json = json.load(f)
        # Remove items that are not in the prompt.
        if "critical_object" in caption_json:
            del caption_json["critical_object"]
        if "noninteractive_expanded_metaaction" in caption_json:
            del caption_json["noninteractive_expanded_metaaction"]
        # Pretty-print json to string.
        caption_json_str = json.dumps(caption_json, indent=2)
        # Format caption in Qwen's default json output format.
        caption_data = f"```json\n{caption_json_str}\n```"
        return caption_data


class CosmosSFTDataset(Dataset):
    def __init__(self, dataset_path: str, system_prompt: str, user_prompt: str):
        # Prompts
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

        # Paths to video files and caption directories
        self.dataset_path = Path(dataset_path)
        self.videos_dir = self.dataset_path / "videos"
        self.captions_dir = self.dataset_path / "metas"

        # Read directories and construct list of entries
        self.data_entries = self._load_dataset_entries()

    def _load_dataset_entries(self):
        """Load dataset entries by scanning video files and matching each file to a JSON caption."""
        entries = []

        # Validate directories exist
        for dir_path in [self.videos_dir, self.captions_dir]:
            if not dir_path.exists():
                raise FileNotFoundError(
                    f"Directory {dir_path} does not exist. Please check the dataset path."
                )

        # Scan video files with the specific naming pattern
        video_files = list(self.videos_dir.glob("*.camera_front_wide_120fov.mp4"))
        print(f"Found {len(video_files)} video files")

        for video_file in video_files:
            # Extract base filename (remove .camera_front_wide_120fov.mp4 suffix)
            video_filename = video_file.name
            if video_filename.endswith(".camera_front_wide_120fov.mp4"):
                base_id = video_filename[: -len(".camera_front_wide_120fov.mp4")]
            else:
                print(
                    f"Warning: Unexpected video filename format: {video_filename}, skipping"
                )
                continue

            # Look for corresponding caption file
            caption_file = self.captions_dir / f"{base_id}.label.json"

            # Check if caption file exists
            if not caption_file.exists():
                print(f"Warning: Caption file not found for {base_id}, skipping")
                continue

            # Load caption data from JSON
            try:
                caption_data = read_caption(caption_file)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Warning: Could not load caption for {base_id}: {e}, skipping")
                continue

            entries.append(
                {
                    "video_id": base_id,
                    "video_path": str(video_file),
                    "caption_data": caption_data,
                }
            )

        print(f"Successfully loaded {len(entries)} dataset entries")
        return entries

    def setup(self, config: Config, tokenizer: AutoTokenizer, *args, **kwargs):
        """Called by launcher after being mounted."""
        self.config = config
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, idx: int):
        """Return a properly formatted conversation from the video file and caption."""
        try:
            entry = self.data_entries[idx]
            caption_data = entry["caption_data"]
        except Exception as e:
            print(f"Error accessing entry {idx}: {e}")
            raise

        # Use the entire caption file content as the description
        full_description = caption_data if caption_data.strip() else "{}"

        # Create conversations in the expected format with system and user prompts
        conversations = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": entry["video_path"],
                        "max_pixels": MAX_PIXELS,
                        "fps": FPS,
                    },
                    {
                        "type": "text",
                        "text": self.user_prompt,
                    },
                ],
            },
            {"role": "assistant", "content": full_description},
        ]
        return conversations

    def get_video_path(self, idx: int) -> str:
        """Get video path for a specific sample"""
        entry = self.data_entries[idx]
        return entry["video_path"]

    def get_video_id(self, idx: int) -> str:
        """Get video ID for a specific sample"""
        entry = self.data_entries[idx]
        return entry["video_id"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    args = parser.parse_known_args()[0]
    with open(args.config) as config_file:
        config = toml.load(config_file)
    config = Config.from_dict(config)

    # Read prompts.
    prompt_dir = SCRIPT_DIR / ".." / "prompts"
    system_prompt_str = read_text_file(prompt_dir / "system_prompt.txt")
    user_prompt_str = read_text_file(prompt_dir / "user_prompt.txt")

    def get_dataset(comos_config: CosmosConfig) -> Dataset:
        # Get dataset path from config, with fallback to default path
        dataset_config = comos_config.train.train_policy.dataset
        if hasattr(dataset_config, "name") and dataset_config.name.startswith("/"):
            # If name is a path, use it directly.
            dataset_path = dataset_config.name
        else:
            raise ValueError(f"No dataset path specified {dataset_config}.")

        print(f"Loading local dataset from: {dataset_path}")
        return CosmosSFTDataset(dataset_path, system_prompt_str, user_prompt_str)

    # It is best practice to pass the dataset as a factory function
    # so that the dataset can be loaded on demand. (Not all workers need it)
    launch_worker(
        dataset=get_dataset,
    )
