#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets",
#   "langchain[openai]",
#   "pydantic",
#   "rich",
#   "tqdm",
# ]
# [tool.uv]
# exclude-newer = "2025-07-31T00:00:00Z"
# ///

"""Generate QA pairs from VLM captions using LLM.

Example:

* [Authenticate](https://gitlab-master.nvidia.com/dir/nvidia-cosmos/cosmos-infra/-/blob/main/scripts/enterprise-api/README.md?ref_type=heads#authentication)

```shell
./generate_qa.py <input_dir> <output_dir> --prompt 'Zhu Li, do the thing!' [--max-samples 1 -v]
```
"""

import argparse
import json
import os
import uuid
from pathlib import Path

import datasets
import langchain_openai
import pydantic
import tqdm
from rich import print


class QA(pydantic.BaseModel):
    question: str = pydantic.Field(description="A question about the image")
    answer: str = pydantic.Field(description="The answer to the question")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        type=str,
        help="Input huggingface dataset directory",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output huggingface dataset directory",
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="Prompt to generate QA pairs"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("LLM_TOKEN"),
        help="API access token",
    )
    parser.add_argument("--model", default="gpt-4o-mini", type=str, help="LLM model")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    correlation_id = str(uuid.uuid4())
    llm = langchain_openai.ChatOpenAI(
        api_key=args.token,
        base_url="https://prod.api.nvidia.com/llm/v1/azure",
        model=args.model,
        default_headers={
            "correlationId": correlation_id,
            "dataClassification": "sensitive",
            "dataSource": "internet",
        },
    )
    structured_llm = llm.with_structured_output(QA)

    dataset = datasets.load_from_disk(args.input_dir)
    if args.max_samples is not None:
        dataset = dataset.select(range(args.max_samples))
    num_samples = len(dataset)

    def process_sample(sample: dict) -> dict:
        key = sample["__key__"]

        metas = json.loads(sample["metas"])
        caption = metas["windows"][0]["qwen_caption"]
        input = f"{args.prompt}\n\n{caption}"
        if args.verbose:
            print("Input:", [input])
        output = structured_llm.invoke(input)
        if args.verbose:
            print("Output:", output)

        conversations = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Please answer the questions.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": "video",
                    },
                    {
                        "type": "text",
                        "text": output.question,
                    },
                ],
            },
            {
                "role": "assistant",
                "content": output.answer,
            },
        ]

        return {
            "__key__": key,
            "conversations": json.dumps(conversations),
            "video": sample["video"],
        }

    dataset = map(process_sample, dataset)
    if not args.verbose:
        dataset = tqdm.tqdm(dataset, total=num_samples, desc="Processing")
    dataset = list(dataset)
    dataset = datasets.Dataset.from_generator(lambda: dataset)
    print(dataset)
    dataset.save_to_disk(str(output_dir))


if __name__ == "__main__":
    main()
