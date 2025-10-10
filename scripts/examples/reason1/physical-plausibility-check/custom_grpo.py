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

"""Custom Reinforcement Learning (RL) dataset."""
# ruff: noqa: E402

import re

from cosmos_reason1_utils.script import init_script

init_script()

import argparse
import json
import warnings
from typing import Union

import toml
from cosmos_reason1_utils.text import set_vision_kwargs
from cosmos_rl.dispatcher.algo.reward import format_reward_fn, single_choice_reward_fn
from cosmos_rl.launcher.worker_entry import main as launch_worker
from cosmos_rl.policy.config import Config as CosmosConfig
from datasets import load_from_disk
from torch.utils.data import Dataset
from transformers import AutoTokenizer

FPS = 1
MAX_PIXELS = 81920


class CustomGRPODataset(Dataset):
    def setup(self, config: CosmosConfig, tokenizer: AutoTokenizer, *args, **kwargs):
        self.config = config
        self.tokenizer = tokenizer

        # Load dataset from disk (like custom_sft.py)
        dataset_path = config.train.train_policy.dataset.name
        self.dataset = load_from_disk(dataset_path)

        # Set up vision kwargs
        self.vision_kwargs = {
            "max_pixels": MAX_PIXELS,
            "fps": FPS,
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        """
        Return conversation structure for GRPO (prompt only, no assistant response)
        """
        sample = self.dataset[idx]

        # Parse the JSON conversation (like custom_sft.py)
        full_conversation = json.loads(sample["conversations"])

        # For GRPO, we use system prompt and user prompt as input
        prompt_conversation = full_conversation[:2]

        # Set vision kwargs
        set_vision_kwargs(prompt_conversation, self.vision_kwargs)

        return prompt_conversation

    def get_reference_answer(self, idx: int) -> str:
        """
        Extract reference answer from the assistant response in the conversation
        """
        sample = self.dataset[idx]
        full_conversation = json.loads(sample["conversations"])

        # Reference answer looks like "<answer>[ground truth score]</answer>"
        # Reward function will parse the string inside the <answer> tag
        return full_conversation[2]["content"]


def single_choice_dense_reward_fn(
    to_be_evaluated: str, reference: Union[str, None], **kwargs
) -> float:
    """Dense reward function based on |student_answer - ground_truth|."""
    try:
        # Extract answer from solution if it has answer tags
        sol_match = re.search(r"<answer>(.*?)</answer>", reference, re.DOTALL)
        ground_truth_str = (
            sol_match.group(1).strip() if sol_match else reference.strip()
        )

        # Extract answer from content if it has answer tags
        content_match = re.search(r"<answer>(.*?)</answer>", to_be_evaluated, re.DOTALL)
        student_answer_str = (
            content_match.group(1).strip() if content_match else to_be_evaluated.strip()
        )

        # Try to convert to float for numerical comparison
        try:
            ground_truth = int(ground_truth_str)
            student_answer = int(student_answer_str)

            if student_answer == ground_truth:
                reward = 1.0
            elif abs(student_answer - ground_truth) <= 1:
                reward = 0.5
            else:
                reward = 0.0

        except ValueError:
            # If conversion to float fails, fall back to string comparison
            if student_answer_str.lower() == ground_truth_str.lower():
                reward = 1.0
            else:
                reward = 0.0

    except Exception:
        reward = 0.0

    return reward


def custom_reward_fn(
    to_be_evaluated: str, reference: str | None = None, *args, **kwargs
) -> float:
    return sum(
        [
            single_choice_dense_reward_fn(to_be_evaluated, reference, *args, **kwargs),
            format_reward_fn(to_be_evaluated, reference, *args, **kwargs),
        ]
    )


if __name__ == "__main__":
    # Suppress warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_known_args()[0]
    with open(args.config) as f:
        config = toml.load(f)
    config = CosmosConfig.from_dict(config)

    # It is best practice to pass the dataset as factory functions
    # so that the dataset can be loaded on demand. (Not all workers need them)
    def get_dataset(config: CosmosConfig) -> Dataset:
        return CustomGRPODataset()

    launch_worker(
        dataset=get_dataset,
        reward_fns=[custom_reward_fn],
        val_dataset=None,  # No validation dataset as requested
        val_data_packer=None,
    )
