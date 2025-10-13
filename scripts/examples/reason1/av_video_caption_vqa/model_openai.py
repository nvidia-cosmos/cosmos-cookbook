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

"""Convenient wrapper class to run ChatGPT5 via API call."""

import os
import uuid
import time

from openai import OpenAI


class OpenAIModel:
    """A thin wrapper class around the OpenAI API.

    Other models, either local or API, can be used instead by implementing
    this simple interface.
    """

    def __init__(self):
        # Get the bearer token from the OPENAI_API_KEY environment variable.
        bearer_token = os.getenv("OPENAI_API_KEY")
        if not bearer_token:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        print("Connecting to ChatGPT...")
        correlation_id = str(uuid.uuid4())
        client = OpenAI(
            api_key=bearer_token,
            base_url="https://prod.api.nvidia.com/llm/v1/azure",
            default_headers={
                "correlationId": correlation_id,
                "dataClassification": "sensitive",
                "dataSource": "internet"
            }
        )
        self.system_prompt = ""
        self.client = client

    def set_system_prompt(self, s: str):
        """Set the system prompt to use for requests."""
        self.system_prompt = s

    def generate(self, prompt: str) -> str:
        """Query the model."""

        print("Querying LLM...")
        time.sleep(0.025)  # Keep us under the query limit.
        messages = []
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt,
            })
        messages.append({
            "role": "user",
            "content": prompt,
        })

        response = self.client.chat.completions.create(
            model="gpt-5",
            messages=messages,
        )
        response_text = response.choices[0].message.content
        return response_text


def test_model():
    """Simple test to see if things are working."""

    model = OpenAIModel()
    model.set_system_prompt("You are a helpful but loud assistant.  Reply in ALL_CAPS.")
    response = model.generate("List five countries in Europe.")
    print(f"{response=}")
