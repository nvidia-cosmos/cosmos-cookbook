#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "accelerate",
#   "cosmos-guardrail",
#   "diffusers>=0.34.0",
#   "rich",
#   "torch",
#   "transformers",
# ]
# [tool.uv]
# exclude-newer = "2025-07-31T00:00:00Z"
# override-dependencies = ["peft==0.15.0"]
# ///

import argparse

import torch
from diffusers import Cosmos2TextToImagePipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str, help="Output path")
    parser.add_argument("--prompt", type=str, help="User prompt message")
    parser.add_argument("--negative_prompt", type=str, help="Negative prompt message")
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/Cosmos-Predict2-2B-Text2Image",
        help="Model name (https://huggingface.co/collections/nvidia/cosmos-predict2-68028efc052239369a0f2959)",
    )

    args = parser.parse_args()

    pipe = Cosmos2TextToImagePipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    )
    pipe.to("cuda")

    output = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        generator=torch.Generator().manual_seed(1),
    ).images[0]
    output.save(args.output)


if __name__ == "__main__":
    main()
