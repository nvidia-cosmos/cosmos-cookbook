#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets",
#   "rich",
#   "webdataset",
# ]
# [tool.uv]
# exclude-newer = "2025-07-31T00:00:00Z"
# ///

"""Extract QA pairs from VLM captions.

Example:

```shell
./extract_qa.py <input_dir> <output_dir>
```
"""

import argparse
import json
from pathlib import Path

import datasets
from rich import print

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=str, help="Input huggingface dataset directory.")
    parser.add_argument(
        "output", type=str, help="Output huggingface dataset directory."
    )
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = datasets.load_from_disk(args.input)

    def process_sample(sample: dict) -> dict:
        key = sample["__key__"]

        # remove ```json and ``` from the sample
        caption = sample["caption"].replace("```json", "").replace("```", "")
        caption = json.loads(caption)

        caption = caption
        # TODO: Split question/answer
        qa_list = []
        for question, answer in caption.items():
            question = question
            answer = answer["answer"] if isinstance(answer, dict) else answer
            qa_list.append({"question": question, "answer": answer})

        # conversations = []
        # for qa in qa_list:
        #     conversations.append([
        #         {
        #             "role": "system",
        #             "content": "You are a helpful assistant. Please answer the questions.",
        #         },
        #         {
        #             "role": "user",
        #             "content": [
        #                 {
        #                     "type": "video",
        #                     "video": "video",
        #                 },
        #                 {
        #                     "type": "text",
        #                     "text": qa["question"],
        #                 },
        #             ],
        #         },
        #         {
        #             "role": "assistant",
        #             "content": qa["answer"],
        #         },
        #     ]
        # )

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
                        "video": sample["video"],
                    },
                    {
                        "type": "text",
                        "text": qa_list[0]["question"],
                    },
                    {
                        "type": "text",
                        "text": qa_list[0]["answer"],
                    }
                ],
            },
        ]

        return {
            "__key__": key,
            "conversations": json.dumps(conversations),
            "video": sample["video"],
        }

    dataset = list(map(process_sample, dataset))
    dataset = datasets.Dataset.from_generator(lambda: dataset)
    print(dataset)
    dataset.save_to_disk(str(output_dir))


if __name__ == "__main__":
    main()
