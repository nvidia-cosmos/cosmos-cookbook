#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets",
#   "qwen-vl-utils",
#   "rich",
#   "torchcodec",
#   "torchvision",
#   "tqdm",
#   "transformers",
#   "vllm",
# ]
# [tool.uv]
# exclude-newer = "2025-08-04T00:00:00Z"
# ///

"""Generate captions from media using VLM.

Example:

```shell
./generate_caption.py <input_dir> <output_dir> --prompt <prompt.yaml> [--max-samples 1 -v]
```
"""

import argparse
import pathlib

import datasets
import msgspec
import pydantic
import qwen_vl_utils
import tqdm
import transformers
import vllm
import yaml
from rich import print

class Prompt(pydantic.BaseModel):
    """Config for prompt."""

    model_config = pydantic.ConfigDict(extra="forbid")

    system_prompt: str | None = pydantic.Field(description="System prompt")
    user_prompt: str | None = pydantic.Field(description="User prompt")


class VisionConfig(pydantic.BaseModel):
    """Config for vision processing.

    Source: https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    nframes: int | None = pydantic.Field(
        None, description="Number of frames of the video"
    )

    fps: float | None = pydantic.Field(None, description="FPS of the video")
    min_frames: int | None = pydantic.Field(None, description="Min frames of the video")
    max_frames: int | None = pydantic.Field(None, description="Max frames of the video")

    min_pixels: int | None = pydantic.Field(
        None, description="Min pixels of the image/video"
    )
    max_pixels: int | None = pydantic.Field(
        None, description="Max pixels of the image/video"
    )
    total_pixels: int | None = pydantic.Field(
        None, description="Total pixels of the video"
    )

    resized_height: int | None = pydantic.Field(
        None, description="Resized height of the video"
    )
    resized_width: int | None = pydantic.Field(
        None, description="Resized width of the video"
    )

    video_start: float | None = pydantic.Field(
        None, description="Start time of the video"
    )
    video_end: float | None = pydantic.Field(None, description="End time of the video")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=str, help="Input directory")
    parser.add_argument("output", type=str, help="Output directory")
    parser.add_argument(
        "--prompt", type=str, required=True, help="Path to prompt yaml file"
    )
    parser.add_argument("--vision-config", type=str, default=, help="Path to vision config json file")
    parser.add_argument("--generation-config", type=str, help="Path to generation config json file")
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/Cosmos-Reason1-7B",
        help="Model to evaluate",
    )
    parser.add_argument(
        "--max-samples", type=int, help="Maximum number of samples to evaluate"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_config = Prompt.model_validate(yaml.safe_load(open(args.prompt, "rb")))
    if args.vision_config is not None:
        vision_config = VisionConfig.model_validate_json(open(args.vision_config, "rb").read())
    else:
        vision_config = VisionConfig()
    vision_kwargs = vision_config.model_dump(exclude_none=True)
    if args.generation_config is not None:
        sampling_params = msgspec.json.decode(
            open(args.generation_config, "rb").read(), type=vllm.SamplingParams
        )
    else:
        sampling_params = vllm.SamplingParams(
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.05,
            max_tokens=4096,
        )

    dataset = datasets.load_from_disk(args.input)
    if args.max_samples is not None:
        dataset = dataset.select(range(args.max_samples))
    num_samples = len(dataset)

    # Load the model
    llm = vllm.LLM(
        model=args.model,
    )
    processor: transformers.Qwen2_5_VLProcessor = (
        transformers.AutoProcessor.from_pretrained(args.model)
    )

    def create_prompt(sample: dict) -> vllm.TextPrompt:
        messages = [
            {
                "role": "system",
                "content": prompt_config.system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_config.user_prompt,
                    },
                    {
                        "type": "video",
                        "video": sample["video"],
                    }
                    | vision_kwargs,
                ],
            },
        ]
        if args.verbose:
            print("Messages:", messages)
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = qwen_vl_utils.process_vision_info(
            messages, return_video_kwargs=True
        )
        multi_modal_data = {}
        if image_inputs is not None:
            multi_modal_data["image"] = image_inputs
        if video_inputs is not None:
            multi_modal_data["video"] = video_inputs
        return dict(
            prompt=prompt,
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=video_kwargs,
        )

    prompts = list(
        tqdm.tqdm(
            map(create_prompt, dataset), desc="Creating prompts", total=num_samples
        )
    )
    outputs = llm.generate(
        prompts,
        sampling_params=sampling_params,
    )

    def process_output(sample: dict, output: vllm.RequestOutput) -> dict:
        output_text = output.outputs[0].text
        return {
            "__key__": sample["__key__"],
            "caption": output_text,
            "video": sample["video"],
        }

    outputs = list(map(process_output, dataset, outputs))
    dataset = datasets.Dataset.from_list(outputs)
    print(dataset)
    dataset.save_to_disk(output_dir)
    dataset.to_json(output_dir / "data.jsonl")


if __name__ == "__main__":
    main()
