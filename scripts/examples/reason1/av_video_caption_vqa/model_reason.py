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

"""Convenient wrapper class to run the Reason model locally."""

from typing import Any, Optional

import qwen_vl_utils
import torch
import transformers

FPS = 2
# Max pixels per frame for Qwen models
# Default patch size is 28 x 28 -- maintain 16:9 aspect ratio.
# +1 to account for floating point error.
MAX_PIXELS = (16 * 9 * 28 * 28) + 1


class LocalModel:
    """A thin wrapper class around a local instance of the Reason 1 model.

    Other models can be used by overriding this simple API.
    """

    def __init__(self, model_path: Optional[str], gpu_id: Optional[int] = None):
        """Initialize the model.

        Args:
            model_path: Either a hugginface model name, or a path to the pretrained weights.
                The model must be fine-tuned variant of Qwen 2.5 VL.
            gpu_id: The GPU to instantiate the model on.
            dryrun: If true,
        """

        # Create a mock of the model for testing purposes.
        if model_path is None:
            print("Dry run.")
            self.model = None
            self.processor = None
            self.gpu_id = gpu_id
            return

        self.system_prompt = ""
        self.gpu_id = gpu_id

        print(f"Loading Cosmos Reason model on GPU {gpu_id}...")
        try:
            if gpu_id is not None:
                torch.cuda.set_device(gpu_id)
                device_map = f"cuda:{gpu_id}"
            else:
                device_map = "auto"

            self.model = (
                transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path,
                    dtype="auto",
                    device_map=device_map,
                    trust_remote_code=True,
                )
            )
            # The processor is a transformers.Qwen2_5_VLProcessor
            self.processor = transformers.AutoProcessor.from_pretrained(model_path)
            print(f"Model loaded successfully on GPU {gpu_id}")
        except Exception as e:
            print(f"Error loading model on GPU {gpu_id}: {e}")
            raise

    def set_system_prompt(self, s: str):
        """Set the system prompt to use for requests."""
        self.system_prompt = s

    def generate(self, prompt: str, video_path: Any) -> str:
        """Query the model.

        Args:
            prompt: The user prompt, e.g. "Caption this video."
            video_path: The path to a video, for VLM tasks.

        Returns:
            The model response, as a string.
        """

        gpu_id = str(self.gpu_id) if self.gpu_id is not None else ""
        if video_path is not None:
            video_path = str(video_path)  # Convert to string... may be a Path object.
            print(f"Processing video {video_path} on GPU {gpu_id}")

        if self.model is None:
            # dry run for testing purposes
            return '```json\n{\n  "weather": "meatballs"\n}\n```'

        content = []
        # Add video path if available.
        if video_path is not None:
            content.append(
                {
                    "type": "video",
                    "video": video_path,
                    "fps": FPS,
                    "max_pixels": MAX_PIXELS,
                }
            )
        # Add user prompt.
        content.append(
            {
                "type": "text",
                "text": prompt,
            }
        )

        # Create a conversation in the expected format.
        conversation = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {"role": "user", "content": content},
        ]

        # Process conversation and load videos.
        text_prompt = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        (image_inputs, video_inputs) = qwen_vl_utils.process_vision_info(conversation)
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Run inference.
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=True,
            temperature=0.1,  # Low temperature for more consistent reasoning
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )

        # Decode the results.
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        result_text = output_text[0].strip()
        return result_text
