# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import pickle
import sys

import torch
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)

# Model configurations
MODEL_CONFIGS = {
    "cr2-2b": {
        "model_path": "nvidia/Cosmos-Reason2-2B-v1.0",
        "processor_path": "nvidia/Cosmos-Reason2-2B-v1.0",
        "model_class": Qwen3VLForConditionalGeneration,
        "dtype": torch.float16,
        "attn_implementation": "sdpa",
        "max_new_tokens": 128,
        "num_trials": 10,
        "pop_token_type_ids": True,
    },
    "cr2-30b": {
        "model_path": "nvidia/Cosmos-Reason2-30B-A3B-v1.0",
        "processor_path": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "model_class": Qwen3VLMoeForConditionalGeneration,
        "dtype": "auto",
        "attn_implementation": None,
        "max_new_tokens": 4096,
        "num_trials": 10,
        "pop_token_type_ids": False,
    },
    "qwen-2b": {
        "model_path": "Qwen/Qwen3-VL-2B-Instruct",
        "processor_path": "Qwen/Qwen3-VL-2B-Instruct",
        "model_class": Qwen3VLForConditionalGeneration,
        "dtype": "auto",
        "attn_implementation": None,
        "max_new_tokens": 4096,
        "num_trials": 10,
        "pop_token_type_ids": False,
    },
    "qwen-8b": {
        "model_path": "Qwen/Qwen3-VL-8B-Instruct",
        "processor_path": "Qwen/Qwen3-VL-8B-Instruct",
        "model_class": Qwen3VLForConditionalGeneration,
        "dtype": "auto",
        "attn_implementation": None,
        "max_new_tokens": 4096,
        "num_trials": 10,
        "pop_token_type_ids": False,
    },
    "qwen-30b": {
        "model_path": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "processor_path": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "model_class": Qwen3VLMoeForConditionalGeneration,
        "dtype": "auto",
        "attn_implementation": None,
        "max_new_tokens": 4096,
        "num_trials": 10,
        "pop_token_type_ids": False,
    },
}

# User prompts dictionary
USER_PROMPTS = {
    "cube": """You should find the following 3 events in the input video
            Event 1: grasping the red cube.
            Event 2: releasing the red cube.
            Event 3: grasping the green cube.
            Extract the exact timestamps for each event.""",
    "cube_bridge": """You should find the following 2 events in the input video
                    Event 1: grasping the cube.
                    Event 2: releasing the cube.
                    Extract the exact timestamps for each event.""",
    "nut": """You should find the following 3 events in the input video
            Event 1: Picking up the red cylinder from the table.
            Event 2: Placing the red cylinder in the blue tray.
            Event 3: Picking up the yellow bowl from the table.
            Extract the exact timestamps for each event.""",
    "toaster": """You should find the following 4 events in the input video
                Event 1: grasping bread.
                Event 2: releasing bread.
                Event 3: pushing the toaster.
                Event 4: releasing the toaster.
                Extract the exact timestamps for each event.""",
    "chips": """You should find the following 2 events in the input video
            Event 1: grasping a bag of chips.
            Event 2: releasing a bag of chips.
            Extract the exact timestamps for each event.""",
    "fork": """You should find the following 4 events in the input video
            Event 1: grasping a fork.
            Event 2: releasing a fork.
            Event 3: grasping a bowl.
            Event 4: releasing a bowl.
            Extract the exact timestamps for each event.""",
    "cup": """You should find the following 3 events in the input video
            Event 1: grasping a cup.
            Event 2: grasping a rag.
            Event 3: releasing a rag.
            Extract the exact timestamps for each event.""",
}

# System prompt (common for all)
SYSTEM_PROMPT = """You are a specialized behavior analyst. Your task is to analyze the video and identify MULTIPLE discrete events with precise timestamps. At each frame, the timestamp is embedded at the bottom of the video. You need to extract the timestamp and answer the user question
                    CRITICAL REQUIREMENTS:
                    1. Extract timestamps from the bottom of each frame
                    2. Extract timestamps for USER-DEFINED events

                    Answer the question in the following format:
                    <think>
                    I will analyze the video systematically:
                    1. First, identify ALL visible timestamps throughout the video
                    2. Identify USER-DEFINED events
                    3. Extract timestamps for identified USER-DEFINED events. There will be different timestamps for each video.
                    4. Always answer in English

                    Event 1: <start time> - <end time> - Event | reasoning
                    Event 2: <start time> - <end time> - Event | reasoning
                    Event 3: <start time> - <end time> - Event | reasoning

                    [Continue for all events identified]
                    </think>

                    <answer>
                    Event 1: <start time> - <end time> Specific Event | detailed explanation.
                    Event 2: <start time> - <end time> Specific Event | detailed explanation.
                    Event 3: <start time> - <end time> Specific Event | detailed explanation.
                    [Continue for all events identified]
                    </answer>"""


def load_model(config):
    """Load model and processor based on configuration"""
    print(f"Loading model: {config['model_path']}")
    print(f"Loading processor: {config['processor_path']}")

    model_kwargs = {
        "dtype": config["dtype"],
        "device_map": "auto",
    }

    if config["attn_implementation"]:
        model_kwargs["attn_implementation"] = config["attn_implementation"]

    model = config["model_class"].from_pretrained(config["model_path"], **model_kwargs)
    processor = AutoProcessor.from_pretrained(config["processor_path"])

    return model, processor


def get_video_files(video_dir, extensions=(".mp4", ".avi", ".mov", ".mkv")):
    """Get all video files from the directory"""
    video_files = []
    for file in os.listdir(video_dir):
        if file.lower().endswith(extensions):
            video_files.append(os.path.join(video_dir, file))
    return sorted(video_files)


def process_video(model, processor, config, video_path, fps, user_prompt):
    """Process a single video"""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "fps": fps,
                },
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    if config["pop_token_type_ids"]:
        inputs.pop("token_type_ids", None)

    generated_ids = model.generate(**inputs, max_new_tokens=config["max_new_tokens"])
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text


def main():
    """Main function to process all demo videos"""
    parser = argparse.ArgumentParser(
        description="Process videos with different vision-language models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Demo-based mode:
  python process_video_vllm_qwen3_uni.py --model cr2-2b --prompt cube --output_dir results
  python process_video_vllm_qwen3_uni.py --model qwen-30b --prompt chips --fps 8 --num_trials 5 --output_dir results
  python process_video_vllm_qwen3_uni.py --model qwen-8b --prompt pour --start_demo 0 --end_demo 5 --output_dir results

  # Video directory mode:
  python process_video_vllm_qwen3_uni.py --model cr2-2b --prompt cube --video_dir /path/to/videos --output_dir results
  python process_video_vllm_qwen3_uni.py --model qwen-8b --prompt chips --video_dir /path/to/videos --fps 8 --output_dir results
        """,
    )

    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for results"
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        required=True,
        help="Model to use for processing",
    )
    parser.add_argument(
        "--prompt",
        choices=list(USER_PROMPTS.keys()),
        required=True,
        help="User prompt type to use",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="Directory containing video files to process (alternative to demo-based mode)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="Frames per second for video processing (default: 8)",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=None,
        help="Number of trials per demo (default: from model config)",
    )
    parser.add_argument(
        "--start_demo",
        type=int,
        default=0,
        help="Starting demo number (default: 0) - only for demo-based mode",
    )
    parser.add_argument(
        "--end_demo",
        type=int,
        default=10,
        help="Ending demo number (exclusive, default: 10) - only for demo-based mode",
    )
    parser.add_argument(
        "--video_path_template",
        type=str,
        default="/mnt/pvc/datasets/videos_bridge_small/02_2023-05-05_10-36-06_traj_group0_traj{demo_num}_images0.mp4",
        help="Video path template with {demo_num} placeholder - only for demo-based mode",
    )

    args = parser.parse_args()

    # Get configuration
    config = MODEL_CONFIGS[args.model]
    user_prompt = USER_PROMPTS[args.prompt]
    num_trials = (
        args.num_trials if args.num_trials is not None else config["num_trials"]
    )

    # Load model
    model, processor = load_model(config)

    print(f"\n{'='*60}")
    print(f"Configuration:")
    print(f"  Model: {args.model} ({config['model_path']})")
    print(f"  Prompt: {args.prompt}")
    print(f"  FPS: {args.fps}")
    print(f"  Trials per video: {num_trials}")
    print(f"  Output directory: {args.output_dir}")

    # Determine which mode to use
    if args.video_dir:
        print(f"  Mode: Video Directory")
        print(f"  Video directory: {args.video_dir}")

        # Get all video files from the directory
        video_files = get_video_files(args.video_dir)

        if not video_files:
            print(f"\nError: No video files found in {args.video_dir}")
            return

        print(f"  Found {len(video_files)} video files")
        videos_to_process = [
            (os.path.basename(vp).rsplit(".", 1)[0], vp) for vp in video_files
        ]
    else:
        print(f"  Mode: Demo-based")
        print(f"  Demo range: {args.start_demo} to {args.end_demo-1}")
        print(f"  Video template: {args.video_path_template}")

        # Generate demo-based video paths
        videos_to_process = []
        for demo_num in range(args.start_demo, args.end_demo):
            video_path = args.video_path_template.format(demo_num=demo_num)
            if os.path.exists(video_path):
                videos_to_process.append((f"demo{demo_num}", video_path))
            else:
                print(f"  Warning: Video not found: {video_path}")

    print(f"{'='*60}\n")

    # Process videos
    for video_name, video_path in videos_to_process:
        result_dict = {}

        print(f"\nProcessing: {video_name}")
        print(f"  Video: {video_path}")

        fps_key = f"fps{args.fps}"
        print(f"  FPS {args.fps}:\n")
        result_dict[fps_key] = []

        for trial_num in range(1, num_trials + 1):
            print(f"  Trial {trial_num}/{num_trials}")

            try:
                # Process the video
                output_text = process_video(
                    model, processor, config, video_path, args.fps, user_prompt
                )

                result_dict[fps_key].append(output_text)
                result_dict["video_path"] = video_path
                result_dict["model"] = args.model
                result_dict["prompt_type"] = args.prompt

                # Save result_dict after each trial
                os.makedirs(args.output_dir, exist_ok=True)
                output_file = f"{args.output_dir}/results_{video_name}.pkl"
                with open(output_file, "wb") as f:
                    pickle.dump(result_dict, f)

                print(f"    Saved to: {output_file}")

            except Exception as e:
                print(f"    Error in trial {trial_num}: {str(e)}")
                continue

    print("\nAll processing completed!")


if __name__ == "__main__":
    main()
